import json
import base64
import logging
import asyncio
import uuid
import contextlib
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from strands.experimental.bidi import (
    BidiTranscriptStreamEvent, BidiAudioStreamEvent, 
    BidiInterruptionEvent, BidiAudioInputEvent, ToolUseStreamEvent
)
from src.core.config import settings
from src.core.sessions import SessionStore
from src.services.voice_orchestrator import VoiceOrchestrator

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)

_AUDIO_SAMPLE_RATES = {16000, 24000, 48000}
_AUDIO_CHANNELS = {1, 2}
_ENDPOINTING = {"HIGH", "MEDIUM", "LOW"}
_VOICES = {
    "matthew",
    "tiffany",
    "amy",
    "olivia",
    "kiara",
    "arjun",
    "ambre",
    "florian",
    "beatrice",
    "lorenzo",
    "tina",
    "lennart",
    "lupe",
    "carlos",
    "carolina",
    "leo",
}
_LOCALES = {
    "en-US",
    "en-GB",
    "en-AU",
    "en-IN",
    "fr-FR",
    "it-IT",
    "de-DE",
    "es-US",
    "pt-BR",
    "hi-IN",
}


def _parse_int(value: str | None, *, default: int, allowed: set[int]) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed in allowed else default


def _parse_bounded_int(value: str | None, *, default: int, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


def _parse_float(value: str | None, *, default: float, min_value: float, max_value: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


@router.websocket("/ws")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    
    orchestrator: VoiceOrchestrator = websocket.app.state.orchestrator
    kb = websocket.app.state.kb
    sessions: SessionStore = websocket.app.state.sessions

    qp = websocket.query_params
    voice = qp.get("voice") or None
    if voice is not None and voice not in _VOICES:
        voice = None

    assistant_lang = qp.get("assistant_lang") or "auto"
    if assistant_lang != "auto" and assistant_lang not in _LOCALES:
        assistant_lang = "auto"
    allow_code_switch = (qp.get("code_switch") or "1") not in {"0", "false", "False"}
    endpointing = qp.get("endpointing")
    if endpointing is not None and endpointing not in _ENDPOINTING:
        endpointing = None

    input_rate = _parse_int(qp.get("input_rate"), default=settings.INPUT_SAMPLE_RATE, allowed=_AUDIO_SAMPLE_RATES)
    output_rate = _parse_int(qp.get("output_rate"), default=settings.OUTPUT_SAMPLE_RATE, allowed=_AUDIO_SAMPLE_RATES)
    channels = _parse_int(qp.get("channels"), default=settings.CHANNELS, allowed=_AUDIO_CHANNELS)

    temperature = _parse_float(qp.get("temperature"), default=0.2, min_value=0.0, max_value=1.0)
    top_p = _parse_float(qp.get("top_p"), default=0.9, min_value=0.0, max_value=1.0)
    max_tokens = _parse_bounded_int(qp.get("max_tokens"), default=256, min_value=1, max_value=8192)

    chat_id = uuid.uuid4().hex
    agent = orchestrator.create_agent(
        chat_id=chat_id,
        assistant_lang=assistant_lang,
        allow_code_switch=allow_code_switch,
        voice=voice,
        input_rate=input_rate,
        output_rate=output_rate,
        channels=channels,
        audio_format="pcm",
        endpointing_sensitivity=endpointing or "LOW",
        inference={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
    )
    await sessions.add(chat_id, agent)

    await websocket.send_text(json.dumps({"event": {"chatInit": {"chatId": chat_id}}}))

    await agent.start()

    async def safe_send_text(payload: dict) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            # Client disconnected or transport error; let receiver unwind.
            raise

    async def safe_send_bytes(data: bytes) -> None:
        try:
            await websocket.send_bytes(data)
        except Exception:
            raise
    
    async def agent_receiver():
        current_bot_text = ""
        try:
            async for event in agent.receive():
                if isinstance(event, BidiTranscriptStreamEvent):
                    if event.role == "user" and event.text:
                        await safe_send_text({"event": {"userTranscript": event.text}})
                    
                    if event.role == "assistant" and event.text:
                        # Some providers emit either deltas or full cumulative text; handle both.
                        if event.text.startswith(current_bot_text):
                            current_bot_text = event.text
                        else:
                            current_bot_text += event.text
                        await safe_send_text(
                            {"event": {"textOutput": {"content": current_bot_text, "isFinal": bool(event.is_final)}}}
                        )
                    
                    if event.is_final and event.role == "assistant":
                        logger.info(f"Agent: {current_bot_text}")
                        current_bot_text = ""
                        await safe_send_text({"event": {"assistantFinal": True}})

                elif isinstance(event, BidiAudioStreamEvent) and event.audio:
                    audio_data = base64.b64decode(event.audio)
                    await safe_send_bytes(audio_data)

                elif isinstance(event, ToolUseStreamEvent):
                    tool_name = event.get("current_tool_use", {}).get("name", "tool")
                    # Send a structured event to frontend for better handling
                    await safe_send_text(
                        {
                            "event": {
                                "toolEvent": {
                                    "name": tool_name,
                                    "status": "started",  # Can be extended to "progress", "completed", "error"
                                }
                            }
                        }
                    )

        except Exception as e:
            logger.error(f"Agent receiver error: {e}")

    receiver_task = asyncio.create_task(agent_receiver())

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg:
                await agent.send(BidiAudioInputEvent(
                    audio=base64.b64encode(msg["bytes"]).decode('utf-8'),
                    format="pcm",
                    sample_rate=input_rate,
                    channels=channels,
                ))
            elif "text" in msg:
                await agent.send(msg["text"])
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # Starlette can raise if receive() is called after disconnect.
        pass
    finally:
        receiver_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await receiver_task
        with contextlib.suppress(Exception):
            await agent.stop()
        await sessions.remove(chat_id)
        kb.clear_chat(chat_id)
        logger.info("Voice session ended")
