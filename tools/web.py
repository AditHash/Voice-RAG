import logging
import asyncio
from ddgs import DDGS
from strands import tool

logger = logging.getLogger(__name__)

@tool(description="Perform a deep web search to find real-time news, current events, or general info when not in documents.")
async def web_search(query: str) -> str:
    """Perform a comprehensive web search using DuckDuckGo."""
    logger.info(f"Tool: Deep searching the web for: {query}")
    
    def run_search():
        with DDGS() as ddgs:
            return ddgs.text(query, region='wt-wt', safesearch='moderate', max_results=10)

    try:
        search_results = await asyncio.to_thread(run_search)
        
        results = []
        for r in search_results:
            results.append(f"Source: {r['title']}
Snippet: {r['body']}
URL: {r['href']}")
        
        if not results:
            return f"No detailed results found for '{query}'."
        
        return "

".join(results)[:2000] 
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return "The web search service is currently unavailable."
