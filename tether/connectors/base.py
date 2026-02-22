"""Abstract connector protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, ConfigDict


class InlineButton(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    callback_data: str


class BaseConnector(ABC):
    def __init__(self) -> None:
        self._message_handler: (
            Callable[[str, str, str], Coroutine[Any, Any, str]] | None
        ) = None
        self._approval_resolver: (
            Callable[[str, bool], Coroutine[Any, Any, bool]] | None
        ) = None
        self._interaction_resolver: (
            Callable[[str, str], Coroutine[Any, Any, bool]] | None
        ) = None
        self._auto_approve_handler: Callable[[str, str], None] | None = None
        self._command_handler: (
            Callable[[str, str, str, str], Coroutine[Any, Any, str]] | None
        ) = None

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        buttons: list[list[InlineButton]] | None = None,
    ) -> None: ...

    @abstractmethod
    async def send_typing_indicator(self, chat_id: str) -> None: ...

    @abstractmethod
    async def request_approval(
        self, chat_id: str, approval_id: str, description: str, tool_name: str = ""
    ) -> str | None: ...

    @abstractmethod
    async def send_file(self, chat_id: str, file_path: str) -> None: ...

    async def send_message_with_id(
        self,
        chat_id: str,  # noqa: ARG002
        text: str,  # noqa: ARG002
    ) -> str | None:
        """Send a message and return its platform ID for later editing.

        Returns None if not supported (streaming will be disabled).
        """
        return None

    async def edit_message(  # noqa: B027
        self, chat_id: str, message_id: str, text: str
    ) -> None:
        """Edit an existing message. Default: no-op."""

    async def delete_message(  # noqa: B027
        self, chat_id: str, message_id: str
    ) -> None:
        """Delete a message by ID. Default: no-op."""

    def set_message_handler(
        self,
        handler: Callable[[str, str, str], Coroutine[Any, Any, str]],
    ) -> None:
        """Register handler(user_id, text, chat_id) for incoming messages."""
        self._message_handler = handler

    def set_approval_resolver(
        self,
        resolver: Callable[[str, bool], Coroutine[Any, Any, bool]],
    ) -> None:
        """Register resolver(approval_id, approved) for approval callbacks."""
        self._approval_resolver = resolver

    def set_interaction_resolver(
        self,
        resolver: Callable[[str, str], Coroutine[Any, Any, bool]],
    ) -> None:
        """Register resolver(interaction_id, answer) for interaction callbacks."""
        self._interaction_resolver = resolver

    def set_auto_approve_handler(
        self,
        handler: Callable[[str, str], None],
    ) -> None:
        """Register handler(chat_id, tool_name) to enable auto-approve for a tool type."""
        self._auto_approve_handler = handler

    def set_command_handler(
        self,
        handler: Callable[[str, str, str, str], Coroutine[Any, Any, str]],
    ) -> None:
        """Register handler(user_id, command, args, chat_id) for slash commands."""
        self._command_handler = handler

    async def send_question(  # noqa: B027
        self,
        chat_id: str,
        interaction_id: str,
        question_text: str,
        header: str,
        options: list[dict[str, str]],
    ) -> None:
        """Send a question with option buttons. Default: no-op."""

    async def send_activity(
        self,
        chat_id: str,  # noqa: ARG002
        tool_name: str,  # noqa: ARG002
        description: str,  # noqa: ARG002
    ) -> str | None:
        """Create/update a standalone activity indicator. Returns message ID."""
        return None

    async def clear_activity(  # noqa: B027
        self, chat_id: str
    ) -> None:
        """Delete the current activity indicator for a chat."""

    async def send_plan_messages(
        self,
        chat_id: str,  # noqa: ARG002
        plan_text: str,  # noqa: ARG002
    ) -> list[str]:
        """Send plan as split messages, return message IDs."""
        return []

    async def delete_messages(  # noqa: B027
        self,
        chat_id: str,
        message_ids: list[str],
    ) -> None:
        """Bulk delete messages by ID."""

    async def clear_plan_messages(self, chat_id: str) -> None:  # noqa: B027
        """Delete tracked plan messages for a chat. Default: no-op."""

    async def send_plan_review(  # noqa: B027
        self,
        chat_id: str,
        interaction_id: str,
        description: str,
    ) -> None:
        """Send plan review prompt with proceed/adjust/clean options. Default: no-op."""
