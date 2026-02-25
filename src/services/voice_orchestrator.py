import logging
import boto3
from datetime import datetime
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator

from src.core.config import settings
from src.core.prompts import get_system_prompt
from src.services.knowledge_base import KnowledgeBaseService
from src.tools.rag import get_rag_tool
from src.tools.web import get_web_search_tool

logger = logging.getLogger(__name__)

class VoiceOrchestrator:
    def __init__(self, session: boto3.Session, kb: KnowledgeBaseService):
        self.session = session
        self.kb = kb
        self.current_date = datetime.now().strftime("%A, %B %d, %Y")

    def create_agent(self) -> BidiAgent:
        """Assembles a specialized BidiAgent instance."""
        
        # Tools initialized with session for Nova Lite reasoning
        search_internal_documents = get_rag_tool(self.kb, self.session)
        web_search = get_web_search_tool(self.session)

        model = BidiNovaSonicModel(
            model_id=settings.NOVA_SONIC_MODEL_ID,
            provider_config={
                "audio": {
                    "voice": settings.VOICE_ID,
                    "output_rate": settings.OUTPUT_SAMPLE_RATE
                },
                "turn_detection": {"endpointingSensitivity": "LOW"}
            },
            client_config={"boto_session": self.session}
        )

        return BidiAgent(
            model=model,
            system_prompt=get_system_prompt(self.current_date),
            tools=[calculator, stop_conversation, search_internal_documents, web_search]
        )
