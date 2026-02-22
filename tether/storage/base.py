"""Abstract session store protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tether.core.session import Session


@runtime_checkable
class SessionStore(Protocol):
    async def save(self, session: Session) -> None: ...

    async def load(self, user_id: str, chat_id: str) -> Session | None: ...

    async def delete(self, user_id: str, chat_id: str) -> None: ...

    async def setup(self) -> None: ...

    async def teardown(self) -> None: ...
