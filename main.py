import os
import asyncio
import logging
import json
import base64
import boto3
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
from config import Config

# Initialize environment and logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Universal Authentication Resolver ---
# This ensures it works with 1) Bedrock API Key, 2) SigV4 Keys, or 3) AWS SSO Sessions

if Config.BEDROCK_API_KEY:
    # Set it as a bearer token for Bedrock clients (e.g. Converse, Embeddings)
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = Config.BEDROCK_API_KEY
    logger.info("Auth: Found BEDROCK_API_KEY. Configured as Bearer Token.")

# If we have keys in the .env, we clear SSO profile to avoid refresh crashes
if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
    os.environ.pop("AWS_PROFILE", None)
    os.environ.pop("AWS_DEFAULT_PROFILE", None)
    logger.info("Auth: Found SigV4 keys in environment. Prioritizing over SSO profiles.")

# Initialize the AWS Session with available credentials
session = boto3.Session(
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
    aws_session_token=Config.AWS_SESSION_TOKEN,
    region_name=Config.AWS_REGION
)
# --- End of Auth Resolver ---

app = FastAPI()

# Initialize Knowledge Base
kb = KnowledgeBase(session, Config.AWS_REGION)

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
                    sample_rate=Config.INPUT_SAMPLE_RATE,
                    channels=Config.CHANNELS
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
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
