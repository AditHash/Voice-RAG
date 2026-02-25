import json
import base64
import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from strands.experimental.bidi import (
    BidiTranscriptStreamEvent, BidiAudioStreamEvent, 
    BidiInterruptionEvent, BidiAudioInputEvent, ToolUseStreamEvent
)
from src.voice_rag.core.config import settings
from src.voice_rag.services.voice_orchestrator import VoiceOrchestrator

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)

@router.websocket("/ws")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    
    orchestrator: VoiceOrchestrator = websocket.app.state.orchestrator
    agent = orchestrator.create_agent()
    kb = websocket.app.state.kb

    await agent.start()
    
    async def agent_receiver():
        current_bot_text = ""
        try:
            async for event in agent.receive():
                if isinstance(event, BidiTranscriptStreamEvent):
                    if event.role == "user" and event.text:
                        await websocket.send_text(json.dumps({"event": {"userTranscript": event.text}}))
                    
                    if event.role == "assistant" and event.text:
                        current_bot_text += event.text
                        await websocket.send_text(json.dumps({"event": {"textOutput": {"content": current_bot_text}}}))
                    
                    if event.is_final and event.role == "assistant":
                        logger.info(f"Agent: {current_bot_text}")
                        current_bot_text = ""

                elif isinstance(event, BidiAudioStreamEvent) and event.audio:
                    audio_data = base64.b64decode(event.audio)
                    # Relay in small chunks to prevent browser buffering/stuttering
                    chunk_size = 512
                    for i in range(0, len(audio_data), chunk_size):
                        await websocket.send_bytes(audio_data[i:i+chunk_size])
                        # Tiny yield to allow the event loop to breathe
                        await asyncio.sleep(0.001)

                elif isinstance(event, ToolUseStreamEvent):
                    tool_name = event.get("current_tool_use", {}).get("name", "tool")
                    await websocket.send_text(json.dumps({"event": {"statusUpdate": f"Using {tool_name}..."}}))

        except Exception as e:
            logger.error(f"Agent receiver error: {e}")

    receiver_task = asyncio.create_task(agent_receiver())

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg:
                await agent.send(BidiAudioInputEvent(
                    audio=base64.b64encode(msg["bytes"]).decode('utf-8'),
                    format="pcm",
                    sample_rate=settings.INPUT_SAMPLE_RATE,
                    channels=settings.CHANNELS
                ))
            elif "text" in msg:
                await agent.send(msg["text"])
    except WebSocketDisconnect:
        pass
    finally:
        await agent.stop()
        receiver_task.cancel()
        logger.info("Voice session ended")
