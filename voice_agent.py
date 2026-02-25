import logging
import boto3
from strands.experimental.bidi import BidiAgent, BidiModel
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator
from knowledge_base import KnowledgeBase

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
        model_id="amazon.nova-2-sonic-v1:0",
        provider_config={
            "audio": {
                "voice": "matthew",
                "output_rate": 24000
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
        You have access to an internal knowledge base. 
        If a user asks a question about company documents, technical info, or specific policies, use the 'search_knowledge_base' tool. 
        Keep your responses concise and conversational for real-time audio.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base]
    )
    
    return agent
