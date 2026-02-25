import logging
import boto3
from datetime import datetime
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator

# Import local functional modules
from knowledge_base import KnowledgeBase
from config import Config
from knowledge_base_tool import get_knowledge_base_tool
from web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

def create_voice_agent(session: boto3.Session, kb: KnowledgeBase) -> BidiAgent:
    """Initialize a fresh BidiAgent with tools and knowledge retrieval capabilities."""
    
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # Initialize tools
    search_knowledge_base = get_knowledge_base_tool(kb)

    # Initialize the Nova Sonic model
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

    # Assemble the BidiAgent with specific instructions for the tools
    agent = BidiAgent(
        model=model,
        system_prompt=f"""You are 'Voice-RAG', a highly intelligent and proactive voice assistant.
        Today's date is {current_date}. 
        
        CORE INSTRUCTIONS:
        1. PERSPECTIVE: You are an expert researcher. If you don't know something, use your tools. 
        2. RAG FIRST: For any questions about documents or specific personal/company info, use 'search_knowledge_base'.
        3. WEB SECOND: For real-time news, upcoming events, or general knowledge, use 'web_search_tool'.
        4. REASONING: Synthesize the information from your tools into a concise response.
        5. CONCISENESS: Be extremely brief and get to the point quickly for voice interaction.
        """,
        tools=[calculator, stop_conversation, search_knowledge_base, web_search_tool]
    )
    
    return agent
