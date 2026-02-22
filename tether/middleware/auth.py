"""Whitelist authentication middleware."""

import structlog

from tether.middleware.base import MessageContext, Middleware, NextHandler

logger = structlog.get_logger()


class AuthMiddleware(Middleware):
    def __init__(self, allowed_user_ids: set[str], *, allow_all: bool = False) -> None:
        self._allowed = allowed_user_ids
        self._allow_all = allow_all

    async def process(self, ctx: MessageContext, call_next: NextHandler) -> str:
        if self._allow_all or ctx.user_id in self._allowed:
            return await call_next(ctx)

        logger.warning("auth_rejected", user_id=ctx.user_id, chat_id=ctx.chat_id)
        return "Unauthorized: you are not allowed to use this bot."
