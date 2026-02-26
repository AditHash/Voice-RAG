import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Literal

from strands.experimental.bidi import BidiAgent


@dataclass
class ChatSession:
    chat_id: str
    agent: BidiAgent
    created_at: datetime
    attachments: Dict[str, "ChatAttachment"]


@dataclass
class ChatAttachment:
    attachment_id: str
    filename: str
    content_type: str
    media_type: Literal["image", "video", "document", "audio", "unknown"]
    data: bytes
    created_at: datetime


class SessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, ChatSession] = {}

    async def add(self, chat_id: str, agent: BidiAgent) -> ChatSession:
        session = ChatSession(
            chat_id=chat_id,
            agent=agent,
            created_at=datetime.now(timezone.utc),
            attachments={},
        )
        async with self._lock:
            self._sessions[chat_id] = session
        return session

    async def get(self, chat_id: str) -> Optional[ChatSession]:
        async with self._lock:
            return self._sessions.get(chat_id)

    async def exists(self, chat_id: str) -> bool:
        async with self._lock:
            return chat_id in self._sessions

    async def remove(self, chat_id: str) -> Optional[ChatSession]:
        async with self._lock:
            return self._sessions.pop(chat_id, None)

    async def add_attachment(
        self,
        chat_id: str,
        *,
        attachment_id: str,
        filename: str,
        content_type: str,
        media_type: Literal["image", "video", "document", "audio", "unknown"],
        data: bytes,
    ) -> Optional[ChatAttachment]:
        attachment = ChatAttachment(
            attachment_id=attachment_id,
            filename=filename,
            content_type=content_type,
            media_type=media_type,
            data=data,
            created_at=datetime.now(timezone.utc),
        )
        async with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return None
            session.attachments[attachment_id] = attachment
        return attachment

    async def list_attachments(self, chat_id: str) -> list[ChatAttachment]:
        async with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return []
            return list(session.attachments.values())

    async def get_attachment(self, chat_id: str, attachment_id: str) -> Optional[ChatAttachment]:
        async with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return None
            return session.attachments.get(attachment_id)

    async def get_latest_attachment(
        self,
        chat_id: str,
        *,
        media_type: Literal["image", "video", "document", "audio", "unknown"] | None = None,
    ) -> Optional[ChatAttachment]:
        async with self._lock:
            session = self._sessions.get(chat_id)
            if session is None or not session.attachments:
                return None
            attachments = list(session.attachments.values())
            if media_type is not None:
                attachments = [a for a in attachments if a.media_type == media_type]
            if not attachments:
                return None
            attachments.sort(key=lambda a: a.created_at, reverse=True)
            return attachments[0]

    async def clear_attachments(self, chat_id: str) -> bool:
        async with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return False
            session.attachments.clear()
            return True
