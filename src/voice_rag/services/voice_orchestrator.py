import logging
import boto3
from datetime import datetime
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation
from strands_tools import calculator

from src.voice_rag.core.config import settings
from src.voice_rag.services.knowledge_base import KnowledgeBaseService
from src.voice_rag.tools.rag import get_rag_tool
from src.voice_rag.tools.web import get_web_search_tool

logger = logging.getLogger(__name__)

class VoiceOrchestrator:
    def __init__(self, session: boto3.Session, kb: KnowledgeBaseService):
        self.session = session
        self.kb = kb
        self.current_date = datetime.now().strftime("%A, %B %d, %Y")

    def create_agent(self) -> BidiAgent:
        """Assembles a specialized BidiAgent instance."""
        
        # Tools initialized with session for Nova Lite reasoning
        search_docs = get_rag_tool(self.kb, self.session)
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
            system_prompt=self._get_system_prompt(),
            tools=[calculator, stop_conversation, search_docs, web_search]
        )

    def _get_system_prompt(self) -> str:
        return f"""You are 'Voice-RAG', an advanced AI assistant.
        Today's date is {self.current_date}. 

        CRITICAL INSTRUCTIONS:
        1. PERSPECTIVE: You are an expert researcher. Use your tools for any specific or factual queries.
        2. ACCESS: You DO have access to uploaded files via 'search_documents'.
        3. PRIORITY: Search local documents first, then the web.
        4. BREVITY: Keep voice responses extremely concise (1-2 sentences max).
        """
