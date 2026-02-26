import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import settings
from src.core.auth import get_aws_session
from src.core.sessions import SessionStore
from src.services.knowledge_base import KnowledgeBaseService
from src.services.voice_orchestrator import VoiceOrchestrator
from src.api.routes import ingest, websocket, media

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
    sessions = SessionStore()
    orchestrator = VoiceOrchestrator(session, kb_service, sessions)

    # Store in app state for route access
    app.state.kb = kb_service
    app.state.orchestrator = orchestrator
    app.state.sessions = sessions

    # Mount Static Files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Include Routes
    app.include_router(ingest.router)
    app.include_router(websocket.router)
    app.include_router(media.router)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open("index.html", "r") as f:
            return f.read()

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
