import logging
import boto3
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator
from strands_tools.tavily import tavily_search
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
        2. ONLY if the user explicitly asks you to 'search the web' or look for real-time news/info outside your knowledge base, use the 'tavily_search' tool.
        Keep your responses very concise and conversational, suitable for real-time audio interaction.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base, tavily_search]
    )
    
    return agent
