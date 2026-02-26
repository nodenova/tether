"""In-memory session store — same behavior as the original SessionManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tether.core.session import Session


class MemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, Session] = {}

    def _key(self, user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    async def save(self, session: Session) -> None:
        self._data[self._key(session.user_id, session.chat_id)] = session

    async def load(self, user_id: str, chat_id: str) -> Session | None:
        session = self._data.get(self._key(user_id, chat_id))
        if session and session.is_active:
            return session
        return None

    async def delete(self, user_id: str, chat_id: str) -> None:
        self._data.pop(self._key(user_id, chat_id), None)

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass
