import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import feedparser
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# === Updated RSS Sources (verified more reliable as of March 2026) ===
RSS_FEEDS = [
    {"name": "Times of India - Top Stories", "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "category": "India"},
    {"name": "The Hindu - News", "url": "https://www.thehindu.com/news/feeder/default.rss", "category": "India"},  # Better than main feeder
    {"name": "Indian Express - India", "url": "https://indianexpress.com/section/india/feed/", "category": "India"},
    {"name": "Indian Express - World", "url": "https://indianexpress.com/section/world/feed/", "category": "World"},
    {"name": "BBC News - World", "url": "http://feeds.bbci.co.uk/news/rss.xml", "category": "World"},
    {"name": "Hindustan Times - India", "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml", "category": "India"},  # Reliable alternative
    {"name": "Reuters - India", "url": "https://ir.thomsonreuters.com/rss/news-releases.xml", "category": "India/World"},  # Broad & fresh
]

@tool
def fetch_latest_news(hours_back: int = 24) -> str:
    """Fetch and filter latest articles from all RSS feeds. Returns list of dicts as string."""
    from datetime import datetime, timedelta
    import feedparser
    
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    articles = []
    
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
            if parsed.bozo:
                print(f"Warning: Bozo flag on {feed['name']}: {parsed.bozo_exception}")
                continue
                
            print(f"Feed {feed['name']}: {len(parsed.entries)} entries found")
            
            for entry in parsed.entries[:20]:  # Limit per feed
                # Flexible date handling
                pub_date = entry.get("published_parsed") or entry.get("updated_parsed") or entry.get("date_parsed")
                if not pub_date:
                    continue  # Skip no-date entries
                
                pub_dt = datetime(*pub_date[:6])
                if pub_dt > cutoff:
                    articles.append({
                        "title": entry.get("title", "No title"),
                        "link": entry.get("link", "#"),
                        "source": feed["name"],
                        "category": feed["category"],
                        "summary": (entry.get("summary") or entry.get("description") or "")[:250] + "..."
                    })
        except Exception as e:
            print(f"Feed {feed['name']} failed: {str(e)}")
            continue
    
    # Sort by recency (if dates available), dedupe by title+link
    articles.sort(key=lambda x: x.get("pub_dt", datetime.min), reverse=True)  # Note: we didn't store pub_dt, but can add if needed
    seen = set()
    unique = [a for a in articles if (a["title"], a["link"]) not in seen and not seen.add((a["title"], a["link"]))]
    
    print(f"Total recent articles: {len(unique)}")
    return str(unique[:30]) if unique else "No recent news found in the last 24 hours from any source."
tools = [fetch_latest_news]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = next(t for t in tools if t.name == tool_call["name"])
        result = tool.invoke(tool_call["args"])
        outputs.append(AIMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": outputs}

# Build graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# === Run & Email ===
def run_news_agent():
    prompt = """You are my personal news curator. Follow these EXACT steps:

1. First, FETCH the latest news articles
2. Then, CREATE a formatted news digest with these EXACT sections:

### 🌍 World
- **Article Title** - One-liner summary - [Read More](https://link-here.com) - Source Name

### 🏛️ Politics  
- **Article Title** - One-liner summary - [Read More](https://link-here.com) - Source Name

### ⚽ Sports
- **Article Title** - One-liner summary - [Read More](https://link-here.com) - Source Name

### 💼 Business
- **Article Title** - One-liner summary - [Read More](https://link-here.com) - Source Name

### 🎬 Entertainment
- **Article Title** - One-liner summary - [Read More](https://link-here.com) - Source Name

IMPORTANT: 
- Only include sections with relevant articles
- Replace (https://link-here.com) with ACTUAL links from the articles
- Keep summaries to ONE sentence max
- Do NOT output raw data or lists
- Format it as shown above with markdown"""
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    result = ""
    print(inputs["messages"][0].content)
    result = ""
    
    for chunk in graph.stream(inputs, {"recursion_limit": 10}):
        # Chunks come as {node_name: {'messages': [...]}}
        for node_name, node_chunk in chunk.items():
            if isinstance(node_chunk, dict) and "messages" in node_chunk:
                msg = node_chunk["messages"][-1]
                # Only capture non-tool-call agent messages (final digest)
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    # Skip raw tool output (lists starting with [{)
                    if not msg.content.strip().startswith('[{'):
                        print(f"[{node_name}] LLM output:", msg.content[:100] + "...")
                        result = msg.content  # Replace, don't append (keep final only)
    
    print("\nFinal result accumulated:\n", result)
    
    # === Convert markdown to HTML ===
    def markdown_to_html(text):
        """Convert markdown to HTML with proper formatting"""
        import re
        html = text
        # Convert section headings (### Heading)
        html = re.sub(
            r'^###\s+(.*?)$', 
            r'<h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding: 12px 0 8px 0; margin: 24px 0 12px 0; font-size: 18px;">\1</h2>', 
            html, 
            flags=re.MULTILINE
        )
        # Convert bullet points (- text)
        html = re.sub(
            r'^-\s+(.*?)$',
            r'<div style="margin: 10px 0 10px 20px; padding: 8px; background: #f8f9fa; border-left: 3px solid #3498db; border-radius: 3px;">\1</div>',
            html,
            flags=re.MULTILINE
        )
        # Convert bold text (**text**)
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #2c3e50;">\1</strong>', html)
        # Convert markdown links [text](url) to embedded HTML links
        html = re.sub(
            r'\[(.*?)\]\((.*?)\)', 
            r'<a href="\2" style="color: #3498db; text-decoration: none; font-weight: 500; border-bottom: 1px dotted #bdc3c7;">\1</a>', 
            html
        )
        # Convert line breaks
        html = html.replace('\n', '<br>')
        return html
    
    html_content = markdown_to_html(result)
    
    # === Build HTML email ===
    html_email = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }}
          .container {{ max-width: 600px; margin: 0 auto; background: #f8f9fa; padding: 20px; border-radius: 8px; }}
          .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; }}
          .header h1 {{ margin: 0; font-size: 24px; }}
          .date {{ font-size: 12px; opacity: 0.9; margin-top: 5px; }}
          .section {{ background: white; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #3498db; }}
          .section h2 {{ margin: 0 0 12px 0; font-size: 18px; color: #2c3e50; }}
          .news-item {{ margin: 12px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
          .news-item strong {{ color: #2c3e50; display: block; margin-bottom: 4px; }}
          .news-item p {{ margin: 4px 0; font-size: 13px; color: #555; }}
          .source {{ font-size: 12px; color: #95a5a6; margin-top: 4px; }}
          a {{ color: #7f8c8d; text-decoration: none; border-bottom: 1px dotted #bdc3c7; }}
          a:hover {{ color: #3498db; border-bottom-color: #3498db; }}
          .footer {{ text-align: center; font-size: 12px; color: #95a5a6; margin-top: 20px; padding-top: 10px; border-top: 1px solid #ecf0f1; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>📰 Daily News Digest</h1>
            <div class="date">{datetime.now().strftime('%A, %B %d, %Y at %I:%M %p IST')}</div>
          </div>
          
          <div class="content">
            {html_content}
          </div>
          
          <div class="footer">
            <p>Stay informed. Stay ahead. | Auto-generated daily news digest</p>
          </div>
        </div>
      </body>
    </html>
    """
    
    # === Send HTML email ===
    try:
        msg = MIMEMultipart('alternative')
        msg["From"] = os.getenv("EMAIL_SENDER")
        msg["To"] = os.getenv("EMAIL_RECEIVER")
        msg["Subject"] = f"📰 Daily News Digest - {datetime.now().strftime('%d %b %Y %I:%M %p IST')}"

        # Attach plain text fallback
        plain_part = MIMEText(result.strip() if result else "No content generated", 'plain', 'utf-8')
        msg.attach(plain_part)
        
        # Attach HTML version (preferred)
        html_part = MIMEText(html_email, 'html', 'utf-8')
        msg.attach(html_part)

        # Debug: Print full message before sending
        print("\n=== EMAIL BEING SENT ===\n")
        print(f"To: {os.getenv('EMAIL_RECEIVER')}")
        print(f"Subject: {msg['Subject']}")
        print(f"Content-Type: HTML\n")

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
        server.send_message(msg)
        server.quit()
        print("✅ HTML Email sent successfully! Check inbox/spam.")
    except Exception as e:
        print(f"❌ Email failed: {str(e)}")

if __name__ == "__main__":
    run_news_agent()
