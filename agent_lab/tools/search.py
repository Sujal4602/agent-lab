from langchain_core.tools import tool
from tavily import TavilyClient
import os

@tool
def search_tool(query: str) -> str:
    """Search the web for current information. Use this for any question about real world facts, news, or anything requiring up to date information."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query, max_results=3)
        print("TAVILY RAW:", result)  # debug line
        outputs = [r["content"] for r in result["results"]]
        return "\n\n".join(outputs)
    except Exception as e:
        print("TAVILY ERROR:", str(e))  # debug line
        return f"Search failed: {str(e)}"