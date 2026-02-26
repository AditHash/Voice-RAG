import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from strands.experimental.bidi import BidiAgent


@dataclass
class ChatSession:
    chat_id: str
    agent: BidiAgent
    created_at: datetime


class SessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, ChatSession] = {}

    async def add(self, chat_id: str, agent: BidiAgent) -> ChatSession:
        session = ChatSession(chat_id=chat_id, agent=agent, created_at=datetime.now(timezone.utc))
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

