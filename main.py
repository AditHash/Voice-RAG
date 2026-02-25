import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.voice_rag.core.config import settings
from src.voice_rag.core.auth import get_aws_session
from src.voice_rag.services.knowledge_base import KnowledgeBaseService
from src.voice_rag.services.voice_orchestrator import VoiceOrchestrator
from src.voice_rag.api.routes import ingest, websocket

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title=settings.PROJECT_NAME)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dependency Initialization
    session = get_aws_session()
    kb_service = KnowledgeBaseService(session)
    orchestrator = VoiceOrchestrator(session, kb_service)

    # Store in app state for route access
    app.state.kb = kb_service
    app.state.orchestrator = orchestrator

    # Include Routes
    app.include_router(ingest.router)
    app.include_router(websocket.router)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open("index.html", "r") as f:
            return f.read()

    # Legacy compatibility redirects
    @app.post("/ingest")
    async def legacy_ingest(file=None): return await ingest.ingest_document(None, file)
    
    @app.post("/reset")
    async def legacy_reset(): return await ingest.reset_kb(None)

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
