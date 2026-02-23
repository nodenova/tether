"""Session manager with optional persistent storage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tether.storage.base import SessionStore

logger = structlog.get_logger()


class Session(BaseModel):
    session_id: str
    user_id: str
    chat_id: str
    working_directory: str
    claude_session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_cost: float = 0.0
    message_count: int = 0
    mode: str = "default"
    mode_instruction: str | None = None
    is_active: bool = True


class SessionManager:
    def __init__(self, store: SessionStore | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self._store = store

    def _key(self, user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    async def get_or_create(
        self, user_id: str, chat_id: str, working_directory: str
    ) -> Session:
        key = self._key(user_id, chat_id)

        # Memory cache first
        session = self._sessions.get(key)
        if session and session.is_active:
            session.last_used = datetime.now(UTC)
            logger.debug(
                "session_cache_hit",
                user_id=user_id,
                chat_id=chat_id,
                session_id=session.session_id,
            )
            return session

        # Try persistent store
        if self._store:
            session = await self._store.load(user_id, chat_id)
            if session and session.is_active:
                session.last_used = datetime.now(UTC)
                self._sessions[key] = session
                logger.info(
                    "session_restored",
                    user_id=user_id,
                    chat_id=chat_id,
                    session_id=session.session_id,
                )
                return session

        # Create new
        session = Session(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            chat_id=chat_id,
            working_directory=working_directory,
        )
        self._sessions[key] = session
        logger.info(
            "session_created",
            user_id=user_id,
            chat_id=chat_id,
            session_id=session.session_id,
        )
        return session

    def get(self, user_id: str, chat_id: str) -> Session | None:
        key = self._key(user_id, chat_id)
        return self._sessions.get(key)

    async def update_from_result(
        self,
        session: Session,
        claude_session_id: str | None = None,
        cost: float = 0.0,
    ) -> None:
        session.last_used = datetime.now(UTC)
        session.message_count += 1
        session.total_cost += cost
        if claude_session_id:
            session.claude_session_id = claude_session_id

        if self._store:
            await self._store.save(session)

        logger.debug(
            "session_updated",
            session_id=session.session_id,
            message_count=session.message_count,
            total_cost=session.total_cost,
            has_claude_session=session.claude_session_id is not None,
        )

    async def deactivate(self, user_id: str, chat_id: str) -> None:
        key = self._key(user_id, chat_id)
        session = self._sessions.get(key)
        if session:
            session.is_active = False
        if self._store:
            await self._store.delete(user_id, chat_id)
        logger.info("session_deactivated", user_id=user_id, chat_id=chat_id)

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        now = datetime.now(UTC)
        expired_keys = [
            k
            for k, s in self._sessions.items()
            if (now - s.last_used).total_seconds() > max_age_hours * 3600
        ]
        for key in expired_keys:
            del self._sessions[key]
        if expired_keys:
            logger.info(
                "sessions_expired", count=len(expired_keys), max_age_hours=max_age_hours
            )
        return len(expired_keys)
