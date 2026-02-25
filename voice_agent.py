import logging
import boto3
import json
import httpx
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

    @tool(description="Search Wikipedia for general knowledge, historical facts, or technical concepts when the user asks to search the web.")
    async def web_search(query: str) -> str:
        """Search Wikipedia for general knowledge."""
        logger.info(f"Agent tool: Searching Wikipedia for: {query}")
        
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return f"No Wikipedia results found for '{query}'."
            
            page_id = next(iter(pages))
            if page_id == "-1":
                return f"No Wikipedia results found for '{query}'."
            
            extract = pages[page_id].get("extract", "")
            if not extract:
                return f"No summary available for '{query}'."
                
            return extract[:800]
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return "Web search failed. Please try again later."

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
        IMPORTANT: You have an internal knowledge base containing documents uploaded by the user.
        
        1. If the user refers to "this document", "the PDF", "the file", or asks "what is this about?", you MUST use the 'search_knowledge_base' tool first to see what's inside. Do not say you don't see any content until you have tried searching.
        2. Use 'search_knowledge_base' for any specific questions where the answer might be in the uploaded documents.
        3. If the user explicitly asks to 'search the web', use the 'web_search' tool.
        
        Keep your responses very concise and conversational for real-time audio.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base, web_search]
    )
    
    return agent
