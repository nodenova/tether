"""Feature registry — explicit plugin registration, no magic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from tether.exceptions import PluginError

if TYPE_CHECKING:
    from tether.plugins.base import PluginContext, TetherPlugin

logger = structlog.get_logger()


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, TetherPlugin] = {}

    def register(self, plugin: TetherPlugin) -> None:
        name = plugin.meta.name
        if name in self._plugins:
            raise PluginError(f"Plugin already registered: {name}")
        self._plugins[name] = plugin
        logger.info("plugin_registered", name=name, version=plugin.meta.version)

    def get(self, name: str) -> TetherPlugin | None:
        return self._plugins.get(name)

    @property
    def plugins(self) -> list[TetherPlugin]:
        return list(self._plugins.values())

    async def init_all(self, context: PluginContext) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.initialize(context)
                logger.info("plugin_initialized", name=plugin.meta.name)
            except Exception as e:
                logger.error("plugin_init_failed", name=plugin.meta.name, error=str(e))
                raise PluginError(
                    f"Plugin {plugin.meta.name} failed to initialize: {e}"
                ) from e

    async def start_all(self) -> None:
        for plugin in self._plugins.values():
            await plugin.start()

    async def stop_all(self) -> None:
        for plugin in reversed(self._plugins.values()):
            try:
                await plugin.stop()
            except Exception:
                logger.exception("plugin_stop_failed", name=plugin.meta.name)
