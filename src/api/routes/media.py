import uuid
from fastapi import APIRouter, UploadFile, File, Request, Query, HTTPException

from src.core.sessions import SessionStore
from src.core.config import settings

router = APIRouter(prefix="/api/media", tags=["media"])


def _guess_media_type(content_type: str) -> str:
    ct = (content_type or "").lower()
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("video/"):
        return "video"
    if ct.startswith("audio/"):
        return "audio"
    if ct in {"application/pdf"}:
        return "document"
    return "unknown"


@router.post("/upload")
async def upload_media(
    request: Request,
    chat_id: str = Query(..., description="Unique chat ID for scoping uploads"),
    file: UploadFile = File(...),
):
    sessions: SessionStore = request.app.state.sessions
    if not await sessions.exists(chat_id):
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")

    content_type = file.content_type or "application/octet-stream"
    media_type = _guess_media_type(content_type)
    if media_type not in {"image", "video", "document", "audio"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content_type: {content_type}")

    max_mb = int(getattr(settings, "MEDIA_UPLOAD_MAX_MB", 25))
    max_bytes = max_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {max_mb}MB)")

    attachment_id = uuid.uuid4().hex
    attachment = await sessions.add_attachment(
        chat_id,
        attachment_id=attachment_id,
        filename=file.filename or attachment_id,
        content_type=content_type,
        media_type=media_type,  # type: ignore[arg-type]
        data=data,
    )
    if attachment is None:
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")

    return {
        "status": "success",
        "attachment": {
            "id": attachment.attachment_id,
            "filename": attachment.filename,
            "content_type": attachment.content_type,
            "media_type": attachment.media_type,
            "bytes": len(attachment.data),
        },
    }


@router.get("/list")
async def list_media(
    request: Request,
    chat_id: str = Query(..., description="Unique chat ID for listing uploads"),
):
    sessions: SessionStore = request.app.state.sessions
    if not await sessions.exists(chat_id):
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")

    items = await sessions.list_attachments(chat_id)
    return {
        "status": "success",
        "attachments": [
            {
                "id": a.attachment_id,
                "filename": a.filename,
                "content_type": a.content_type,
                "media_type": a.media_type,
                "bytes": len(a.data),
                "created_at": a.created_at.isoformat(),
            }
            for a in items
        ],
    }


@router.post("/clear")
async def clear_media(
    request: Request,
    chat_id: str = Query(..., description="Unique chat ID to clear uploads"),
):
    sessions: SessionStore = request.app.state.sessions
    if not await sessions.exists(chat_id):
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")

    ok = await sessions.clear_attachments(chat_id)
    return {"status": "success" if ok else "error"}

