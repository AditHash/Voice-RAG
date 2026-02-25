import logging
import boto3
import json
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator, http_request
from knowledge_base import KnowledgeBase
from config import Config

logger = logging.getLogger(__name__)

def create_voice_agent(session: boto3.Session, kb: KnowledgeBase) -> BidiAgent:
    """Initialize a fresh BidiAgent with tools and knowledge retrieval capabilities."""
    
    # Define the search tool for the knowledge base
    async def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for specific information, company documents, or technical info."""
        logger.info(f"Agent tool: Searching knowledge base with query: {query}")
        return kb.retrieve(query)

    # Define a free web search tool using Wikipedia API via http_request
    async def web_search(query: str) -> str:
        """Search Wikipedia for general knowledge, historical facts, or technical concepts when the user asks to search the web."""
        logger.info(f"Agent tool: Searching Wikipedia for: {query}")
        
        # We use http_request to call the Wikipedia API
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
            # http_request tool expects (url, method, params, headers, etc.)
            response_str = await http_request(
                url="https://en.wikipedia.org/w/api.php",
                method="GET",
                params=params
            )
            data = json.loads(response_str)
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return f"No Wikipedia results found for '{query}'."
            
            # Extract the first page result
            page_id = next(iter(pages))
            if page_id == "-1":
                return f"No Wikipedia results found for '{query}'."
            
            extract = pages[page_id].get("extract", "")
            return extract[:800] # Return first 800 chars for voice brevity
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
        1. Use 'search_knowledge_base' for questions about internal documents, policies, or specific company info.
        2. If the user explicitly asks you to 'search the web' or look for general knowledge, use the 'web_search' tool.
        Keep your responses very concise and conversational, suitable for real-time audio interaction.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base, web_search]
    )
    
    return agent
