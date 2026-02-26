import logging
import boto3
from datetime import datetime
from typing import Any
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

    def create_agent(
        self,
        chat_id: str,
        *,
        assistant_lang: str | None = None,
        allow_code_switch: bool = True,
        voice: str | None = None,
        input_rate: int | None = None,
        output_rate: int | None = None,
        channels: int | None = None,
        audio_format: str | None = None,
        endpointing_sensitivity: str | None = None,
        inference: dict[str, Any] | None = None,
    ) -> BidiAgent:
        """Assembles a specialized BidiAgent instance."""
         
        # Tools initialized with session for Nova Lite reasoning
        search_internal_documents = get_rag_tool(self.kb, self.session, chat_id=chat_id)
        web_search = get_web_search_tool(self.session)

        audio_config: dict[str, Any] = {
            "voice": voice or settings.VOICE_ID,
            "output_rate": output_rate or settings.OUTPUT_SAMPLE_RATE,
        }
        if input_rate is not None:
            audio_config["input_rate"] = input_rate
        if channels is not None:
            audio_config["channels"] = channels
        if audio_format is not None:
            audio_config["format"] = audio_format

        provider_config: dict[str, Any] = {"audio": audio_config}
        if endpointing_sensitivity is not None:
            provider_config["turn_detection"] = {"endpointingSensitivity": endpointing_sensitivity}
        if inference is not None:
            provider_config["inference"] = inference

        model = BidiNovaSonicModel(
            model_id=settings.NOVA_SONIC_MODEL_ID,
            provider_config=provider_config,
            client_config={"boto_session": self.session}
        )

        return BidiAgent(
            model=model,
            system_prompt=get_system_prompt(self.current_date, assistant_lang=assistant_lang, allow_code_switch=allow_code_switch),
            tools=[calculator, stop_conversation, search_internal_documents, web_search]
        )
