import logging
import boto3
import json
import asyncio
from datetime import datetime
from ddgs import DDGS
from strands import tool
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator
from knowledge_base import KnowledgeBase
from config import Config

logger = logging.getLogger(__name__)

def create_voice_agent(session: boto3.Session, kb: KnowledgeBase) -> BidiAgent:
    """Initialize a fresh BidiAgent with tools and knowledge retrieval capabilities."""
    
    # Get current date for the agent's awareness
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    @tool(description="Search the internal knowledge base for specific information, company documents, or technical info.")
    async def search_knowledge_base(query: str) -> str:
        logger.info(f"Agent tool: Searching knowledge base with query: {query}")
        return kb.retrieve(query)

    @tool(description="Perform a deep web search to find real-time news, current events, or general info when not in documents.")
    async def web_search(query: str) -> str:
        """Perform a comprehensive web search using DuckDuckGo."""
        logger.info(f"Agent tool: Deep searching the web for: {query}")
        
        def run_search():
            # Use broader settings for better results
            with DDGS() as ddgs:
                # Get more results (10) to give AI more context
                return ddgs.text(query, region='wt-wt', safesearch='moderate', max_results=10)

        try:
            search_results = await asyncio.to_thread(run_search)
            
            results = []
            for r in search_results:
                results.append(f"Source: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}")
            
            if not results:
                return f"No detailed results found for '{query}'. Try a different search term."
            
            # Combine more content for better reasoning
            combined_results = "\n\n".join(results)
            return combined_results[:2000] 
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "The web search service is currently unavailable."

    # Initialize a fresh model instance
    model = BidiNovaSonicModel(
        model_id=Config.NOVA_SONIC_MODEL_ID,
        provider_config={
            "audio": {
                "voice": Config.VOICE_ID,
                "output_rate": Config.OUTPUT_SAMPLE_RATE
            }
        },
        client_config={
            "boto_session": session
        }
    )

    # Each connection gets its own BidiAgent instance
    agent = BidiAgent(
        model=model,
        system_prompt=f"""You are 'Voice-RAG', a highly intelligent and proactive voice assistant.
        Today's date is {current_date}. 
        
        CORE INSTRUCTIONS:
        1. PERSPECTIVE: You are an expert researcher. If you don't know something, use your tools. 
        2. RAG FIRST: For any questions about documents or specific personal/company info, use 'search_knowledge_base'.
        3. WEB SECOND: For real-time news, upcoming events (like the 2026 World Cup), or general knowledge, use 'web_search'.
        4. REASONING: When using 'web_search', synthesize the information. If search results are vague, say what you found but also offer to search for a more specific detail.
        5. CONCISENESS: Since this is a voice interaction, be extremely concise. Get to the point quickly.
        
        If the user asks "what's happening in 2026", perform a broad web search and summarize the major events.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base, web_search]
    )
    
    return agent
