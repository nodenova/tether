"""Token-bucket rate limiting middleware."""

import time

import structlog

from leashd.middleware.base import MessageContext, Middleware, NextHandler

logger = structlog.get_logger()


class TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

    def consume(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now


class RateLimitMiddleware(Middleware):
    def __init__(self, requests_per_minute: int, burst: int = 5) -> None:
        self._rate = requests_per_minute / 60.0
        self._burst = burst
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, user_id: str) -> TokenBucket:
        if user_id not in self._buckets:
            self._buckets[user_id] = TokenBucket(self._rate, self._burst)
        return self._buckets[user_id]

    async def process(self, ctx: MessageContext, call_next: NextHandler) -> str:
        bucket = self._get_bucket(ctx.user_id)
        if bucket.consume():
            return await call_next(ctx)

        logger.warning("rate_limited", user_id=ctx.user_id)
        return "Rate limited. Please wait before sending another message."
