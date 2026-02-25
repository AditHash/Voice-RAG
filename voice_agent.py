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
    search_documents = get_knowledge_base_tool(kb)

    # Initialize the Nova Sonic model with lower interruption sensitivity
    model = BidiNovaSonicModel(
        model_id=Config.NOVA_SONIC_MODEL_ID,
        provider_config={
            "audio": {
                "voice": Config.VOICE_ID,
                "output_rate": Config.OUTPUT_SAMPLE_RATE
            },
            "turn_detection": {
                "endpointingSensitivity": "LOW" # Makes it less likely to stop on minor noise
            }
        },
        client_config={
            "boto_session": session
        }
    )

    # Assemble the BidiAgent with extremely strict tool-use instructions
    agent = BidiAgent(
        model=model,
        system_prompt=f"""You are 'Voice-RAG', a highly intelligent AI assistant with access to a local knowledge base.
        Today's date is {current_date}. 

        CRITICAL INSTRUCTIONS:
        1. YOU HAVE ACCESS TO FILES: If the user says "what is this pdf", "tell me about the document", "what did I upload", or any question about specific content, YOU MUST CALL the 'search_documents' tool.
        2. NEVER SAY "I don't have access to your files". You DO have access through the 'search_documents' tool.
        3. AUTOMATIC SEARCH: If the user asks a question and you don't know the answer, search the knowledge base first, then search the web.
        4. WEB SEARCH: Use 'web_search_tool' only for real-time news, current events, or general knowledge NOT in the uploaded files.
        5. CONCISENESS: Since this is a voice interaction, be extremely brief.
        
        Example:
        User: "What is this PDF?"
        Action: Call 'search_documents' with query "summary of the document"
        """,
        tools=[calculator, stop_conversation, search_documents, web_search_tool]
    )
    
    return agent
