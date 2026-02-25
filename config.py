import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # AWS Configuration
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
    
    # Bedrock Models
    NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"
    TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
    
    # Audio Configuration
    INPUT_SAMPLE_RATE = 16000
    OUTPUT_SAMPLE_RATE = 24000
    CHANNELS = 1
    VOICE_ID = "matthew"
    
    # RAG Configuration
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "voice_rag_knowledge"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Server Configuration
    HOST = "127.0.0.1"
    PORT = 8001
