from langchain_core.tools import tool
from tavily import TavilyClient
import os

@tool
def search_tool(query: str) -> str:
    """Search the web for current information. Use this for any question about real world facts, news, or anything requiring up to date information."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query, max_results=3)

        outputs = []
        for i, r in enumerate(result["results"], 1):
            title = r.get("title", "No title")
            snippet = r.get("content", "")
            clean_lines = [l for l in snippet.split('\n')
                           if l.strip() and not l.strip().startswith(('#', '* [', '[!['))]
            clean = ' '.join(clean_lines)[:300]
            url = r.get("url", "No URL")
            outputs.append(f"[{i}] {title}\n    {clean}\n    {url}")

        formatted = "\n\n".join(outputs)

        answer = result.get("answer", "")
        if answer:
            formatted = f"Direct answer: {answer}\n\nSources:\n{formatted}"        
        return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"