import logging
import boto3
import json
import asyncio
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
    
    @tool(description="Search the internal knowledge base for specific information, company documents, or technical info.")
    async def search_knowledge_base(query: str) -> str:
        logger.info(f"Agent tool: Searching knowledge base with query: {query}")
        return kb.retrieve(query)

    @tool(description="Perform a real-time web search to find the latest news, current events, or general information from the internet.")
    async def web_search(query: str) -> str:
        """Perform a full web search using DuckDuckGo."""
        logger.info(f"Agent tool: Searching the web for: {query}")
        
        def run_search():
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=5)

        try:
            # Run the synchronous search in a thread to avoid blocking the voice stream
            search_results = await asyncio.to_thread(run_search)
            
            results = []
            for r in search_results:
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}")
            
            if not results:
                return f"No web search results found for '{query}'."
            
            combined_results = "\n\n".join(results)
            return combined_results[:1500] 
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "The web search service is currently unavailable. Please try again later."

    # Initialize a fresh model instance for THIS specific connection
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
        system_prompt="""You are a professional and helpful voice assistant for Voice-RAG. 
        IMPORTANT: You have two primary tools for finding information:
        
        1. 'search_knowledge_base': Use this FIRST for any questions about internal documents, uploaded files (PDFs/Text), or company-specific info.
        2. 'web_search': Use this ONLY if the user explicitly asks to 'search the web' or if they ask about real-time events, news, or general knowledge NOT found in the documents.
        
        Keep your responses very concise and conversational for real-time audio interaction.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base, web_search]
    )
    
    return agent
