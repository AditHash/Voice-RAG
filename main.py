import os
import asyncio
import logging
import json
import base64
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Import BidiAgent and its dependencies
from strands.experimental.bidi import (
    BidiTranscriptStreamEvent,
    BidiAudioStreamEvent,
    BidiInterruptionEvent,
    BidiAudioInputEvent
)

# Import local modules
from knowledge_base import KnowledgeBase
from voice_agent import create_voice_agent

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Bedrock API key (for standard models)
api_key = os.getenv("BEDROCK_API_KEY") or os.getenv("BEDRCOK_API_KEY")
if api_key:
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key

# Clear SSO profiles to prevent boto3 refresh crashes
os.environ.pop("AWS_PROFILE", None)
os.environ.pop("AWS_DEFAULT_PROFILE", None)

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AWS Session (Requires SigV4)
region = os.getenv("AWS_REGION", "us-east-1")
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=region
)

# Initialize Knowledge Base
kb = KnowledgeBase(session, region)

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """API to ingest documents (PDF or Text) into the Knowledge Base."""
    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        num_chunks = kb.ingest_pdf(content, file.filename)
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        num_chunks = kb.ingest_text(text, {"filename": file.filename, "type": "text"})
    return {"status": "success", "filename": file.filename, "chunks_ingested": num_chunks}

@app.post("/retrieve")
async def retrieve_info(query: str):
    """Diagnostic API to search the Knowledge Base manually."""
    context = kb.retrieve(query)
    return {"query": query, "context": context}

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the single-page web UI."""
    with open("index.html", "r") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Bridge the WebSocket to the BidiAgent."""
    await websocket.accept()
    
    # Each connection gets its own BidiAgent instance
    agent = create_voice_agent(session, kb)

    await agent.start()
    logger.info("BidiAgent session started")

    async def receive_from_agent():
        """Handle output events from the agent and forward to the client."""
        current_transcript = ""
        try:
            async for event in agent.receive():
                if isinstance(event, BidiTranscriptStreamEvent):
                    if event.text:
                        current_transcript += event.text
                        await websocket.send_text(json.dumps({
                            "event": {"textOutput": {"content": current_transcript}}
                        }))
                    if event.is_final:
                        current_transcript = ""
                elif isinstance(event, BidiAudioStreamEvent):
                    if event.audio:
                        audio_bytes = base64.b64decode(event.audio)
                        await websocket.send_bytes(audio_bytes)
                elif isinstance(event, BidiInterruptionEvent):
                    logger.info("User interrupted assistant")
        except Exception as e:
            logger.error(f"Error receiving from agent: {e}")

    # Start a background task for agent outputs
    receiver_task = asyncio.create_task(receive_from_agent())

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_b64 = base64.b64encode(message["bytes"]).decode('utf-8')
                await agent.send(BidiAudioInputEvent(
                    audio=audio_b64, 
                    format="pcm",
                    sample_rate=16000,
                    channels=1
                ))
            elif "text" in message:
                await agent.send(message["text"])
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await agent.stop()
        receiver_task.cancel()
        logger.info("BidiAgent session stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
