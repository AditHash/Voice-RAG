import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Voice-RAG"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN: str = os.getenv("AWS_SESSION_TOKEN")
    BEDROCK_API_KEY: str = os.getenv("BEDROCK_API_KEY") or os.getenv("BEDRCOK_API_KEY")
    
    # Models
    NOVA_SONIC_MODEL_ID: str = "amazon.nova-2-sonic-v1:0"
    NOVA_LITE_MODEL_ID: str = "amazon.nova-lite-v1:0"
    # Web grounding (Nova system tool "nova_grounding" via Bedrock Converse toolConfig).
    # Defaults mirror AWS docs: grounding is currently US-region only and may require a "us." model prefix.
    NOVA_GROUNDING_MODEL_ID: str = os.getenv("NOVA_GROUNDING_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
    TITAN_EMBED_MODEL_ID: str = "amazon.titan-embed-text-v2:0"

    # Web search behavior
    # - "auto": try nova_grounding, then fall back to DDGS+Nova Lite synthesis
    # - "grounding": only nova_grounding (no external web calls)
    # - "ddgs": only DDGS+Nova Lite synthesis
    WEB_SEARCH_BACKEND: str = os.getenv("WEB_SEARCH_BACKEND", "auto").lower()
    WEB_SEARCH_MAX_SOURCES: int = int(os.getenv("WEB_SEARCH_MAX_SOURCES", "3"))
    
    # Audio
    INPUT_SAMPLE_RATE: int = 16000
    OUTPUT_SAMPLE_RATE: int = 24000
    CHANNELS: int = 1
    VOICE_ID: str = "matthew"
    
    # Vector DB
    CHROMA_DB_PATH: str = str(BASE_DIR / "chroma_db")
    COLLECTION_NAME: str = "voice_rag_knowledge"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

settings = Settings()
