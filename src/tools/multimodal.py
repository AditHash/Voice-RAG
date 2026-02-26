import asyncio
import json
import logging
from typing import Any, Literal, Optional

import boto3
from botocore.config import Config
from strands import tool

from src.core.config import settings
from src.core.sessions import SessionStore, ChatAttachment

logger = logging.getLogger(__name__)


def _guess_format_from_content_type(content_type: str) -> str:
    ct = (content_type or "").lower().strip()
    if "/" in ct:
        subtype = ct.split("/", 1)[1]
    else:
        subtype = ct
    subtype = subtype.split(";", 1)[0].strip()

    if subtype in {"jpg"}:
        return "jpeg"
    if subtype in {"jpeg", "png", "webp", "gif"}:
        return subtype
    if subtype in {"mp4", "webm", "mov", "mkv"}:
        return subtype
    if subtype in {"pdf"}:
        return "pdf"
    return subtype or "unknown"


def _strip_outer_code_fences(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_text_from_converse(response: dict) -> str:
    content_list = response.get("output", {}).get("message", {}).get("content", [])
    parts: list[str] = []
    for item in content_list:
        t = item.get("text")
        if isinstance(t, str) and t:
            parts.append(t)
    return "".join(parts).strip()


async def _get_attachment(
    sessions: SessionStore,
    *,
    chat_id: str,
    attachment_id: str | None,
    media_type: Literal["image", "video", "document", "audio"] | None,
) -> ChatAttachment | None:
    if attachment_id:
        return await sessions.get_attachment(chat_id, attachment_id)
    if media_type:
        return await sessions.get_latest_attachment(chat_id, media_type=media_type)
    return await sessions.get_latest_attachment(chat_id)


def get_multimodal_tools(sessions: SessionStore, session: boto3.Session, *, chat_id: str):
    bedrock = session.client(
        "bedrock-runtime",
        region_name=settings.AWS_REGION,
        config=Config(read_timeout=3600),
    )

    model_id = settings.NOVA_MULTIMODAL_MODEL_ID

    @tool(
        name="extract_image_text",
        description="Extracts text from the most recently uploaded IMAGE in this chat (OCR). Upload an image first in the UI.",
    )
    async def extract_image_text(attachment_id: Optional[str] = None, text_formatting: str = "markdown") -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="image",
        )
        if attachment is None:
            return "No image uploaded for this chat. Upload an image first."

        img_format = _guess_format_from_content_type(attachment.content_type)
        prompt = f"""## Instructions
Extract all information from this page using only {text_formatting} formatting. Retain the original layout and structure including lists, tables, charts and math formulae.

## Rules
1. For math formulae, always use LaTeX syntax.
2. Describe images using only text.
3. NEVER use HTML image tags <img> in the output.
4. NEVER use Markdown image tags ![]() in the output.
5. Always wrap the entire output in ``` tags.
"""

        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": img_format, "source": {"bytes": attachment.data}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.7, "topP": 0.9},
        )
        text = _extract_text_from_converse(response)
        return _strip_outer_code_fences(text)

    @tool(
        name="extract_image_json",
        description="Extracts structured information from the most recently uploaded IMAGE in this chat using a JSON Schema you provide.",
    )
    async def extract_image_json(json_schema: str, attachment_id: Optional[str] = None) -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="image",
        )
        if attachment is None:
            return "No image uploaded for this chat. Upload an image first."

        img_format = _guess_format_from_content_type(attachment.content_type)
        prompt = f"""Given the image representation of a document, extract information in JSON format according to the given schema.

Follow these guidelines:
- Ensure that every field is populated, provided the document includes the corresponding value. Only use null when the value is absent from the document.
- When instructed to read tables or lists, read each row from every page. Ensure every field in each row is populated if the document contains the field.

JSON Schema:
{json_schema}
"""
        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": img_format, "source": {"bytes": attachment.data}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 2048, "temperature": 0},
        )
        text = _extract_text_from_converse(response)
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return text

    @tool(
        name="locate_in_image",
        description="Finds UI/object locations in the most recently uploaded IMAGE. Returns bounding boxes scaled 0-1000. Provide what you want to locate.",
    )
    async def locate_in_image(target_description: str, attachment_id: Optional[str] = None) -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="image",
        )
        if attachment is None:
            return "No image uploaded for this chat. Upload an image first."

        img_format = _guess_format_from_content_type(attachment.content_type)
        prompt = f"""Detect all objects with their bounding boxes in the image for: {target_description}

Represent bounding boxes as [x1, y1, x2, y2] scaled between 0 and 1000 to the image width and height.
Return JSON as a list:
[
  {{"class": "<label>", "bbox": [x1, y1, x2, y2]}},
  ...
]
"""

        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": img_format, "source": {"bytes": attachment.data}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 1024, "temperature": 0},
        )
        return _extract_text_from_converse(response)

    @tool(
        name="summarize_video",
        description="Summarizes the most recently uploaded VIDEO in this chat. Upload a video first in the UI.",
    )
    async def summarize_video(user_prompt: str = "Create an executive summary of this video's content.", attachment_id: Optional[str] = None) -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="video",
        )
        if attachment is None:
            return "No video uploaded for this chat. Upload a video first."

        vid_format = _guess_format_from_content_type(attachment.content_type)
        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"video": {"format": vid_format, "source": {"bytes": attachment.data}}},
                        {"text": user_prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 1024, "temperature": 0},
        )
        return _extract_text_from_converse(response)

    @tool(
        name="dense_caption_video",
        description="Generates detailed captions for the most recently uploaded VIDEO in this chat.",
    )
    async def dense_caption_video(user_prompt: str = "Describe the video scene-by-scene with key details.", attachment_id: Optional[str] = None) -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="video",
        )
        if attachment is None:
            return "No video uploaded for this chat. Upload a video first."

        vid_format = _guess_format_from_content_type(attachment.content_type)
        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"video": {"format": vid_format, "source": {"bytes": attachment.data}}},
                        {"text": user_prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 2048, "temperature": 0},
        )
        return _extract_text_from_converse(response)

    @tool(
        name="find_video_event_times",
        description="Localize the start/end timestamps of an event in the most recently uploaded VIDEO. Returns list like [[72, 82]].",
    )
    async def find_video_event_times(event_description: str, attachment_id: Optional[str] = None) -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="video",
        )
        if attachment is None:
            return "No video uploaded for this chat. Upload a video first."

        vid_format = _guess_format_from_content_type(attachment.content_type)
        prompt = (
            f'Please localize the moment that the event "{event_description}" happens in the video. '
            "Answer with the starting and ending time of the event in seconds, such as [[72, 82]]. "
            "If the event happens multiple times, list all of them like [[40, 50], [72, 82]]."
        )
        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"video": {"format": vid_format, "source": {"bytes": attachment.data}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        return _extract_text_from_converse(response)

    @tool(
        name="classify_video",
        description="Classify the most recently uploaded VIDEO into one of the provided categories (one per line).",
    )
    async def classify_video(categories: str, attachment_id: Optional[str] = None) -> str:
        attachment = await _get_attachment(
            sessions,
            chat_id=chat_id,
            attachment_id=attachment_id,
            media_type="video",
        )
        if attachment is None:
            return "No video uploaded for this chat. Upload a video first."

        vid_format = _guess_format_from_content_type(attachment.content_type)
        prompt = "What is the most appropriate category for this video? Select your answer from the options provided:\n" + categories.strip()
        response = await asyncio.to_thread(
            bedrock.converse,
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"video": {"format": vid_format, "source": {"bytes": attachment.data}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 256, "temperature": 0},
        )
        return _extract_text_from_converse(response)

    tools = [
        extract_image_text,
        extract_image_json,
        locate_in_image,
        summarize_video,
        dense_caption_video,
        find_video_event_times,
        classify_video,
    ]

    # Log once at construction time; helps diagnose model/region issues.
    logger.info(f"Multimodal tools enabled (modelId={model_id}).")
    return tools

