"""Plugin protocol for extending leashd."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from leashd.core.config import LeashdConfig
from leashd.core.events import EventBus


class PluginMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str = ""


class PluginContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    event_bus: EventBus
    config: LeashdConfig


@runtime_checkable
class LeashdPlugin(Protocol):
    meta: PluginMeta

    async def initialize(self, context: PluginContext) -> None:
        """Called once — subscribe to events here."""
        ...

    async def start(self) -> None:
        """Optional lifecycle hook — called after all plugins are initialized."""
        ...

    async def stop(self) -> None:
        """Optional lifecycle hook — called on shutdown."""
        ...
