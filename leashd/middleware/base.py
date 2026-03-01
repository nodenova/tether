"""Middleware chain — each middleware can pass through or short-circuit."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    chat_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


NextHandler = Callable[[MessageContext], Awaitable[str]]


class Middleware(ABC):
    @abstractmethod
    async def process(self, ctx: MessageContext, call_next: NextHandler) -> str: ...


class MiddlewareChain:
    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    def add(self, middleware: Middleware) -> None:
        self._middleware.append(middleware)

    def has_middleware(self) -> bool:
        return bool(self._middleware)

    async def run(self, ctx: MessageContext, handler: NextHandler) -> str:
        chain = handler
        for mw in reversed(self._middleware):
            chain = _wrap(mw, chain)
        return await chain(ctx)


def _wrap(mw: Middleware, nxt: NextHandler) -> NextHandler:
    async def _next(c: MessageContext) -> str:
        return await mw.process(c, nxt)

    return _next
