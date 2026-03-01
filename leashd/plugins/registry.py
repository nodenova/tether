"""Feature registry — explicit plugin registration, no magic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from leashd.exceptions import PluginError

if TYPE_CHECKING:
    from leashd.plugins.base import LeashdPlugin, PluginContext

logger = structlog.get_logger()


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, LeashdPlugin] = {}

    def register(self, plugin: LeashdPlugin) -> None:
        name = plugin.meta.name
        if name in self._plugins:
            raise PluginError(f"Plugin already registered: {name}")
        self._plugins[name] = plugin
        logger.info("plugin_registered", name=name, version=plugin.meta.version)

    def get(self, name: str) -> LeashdPlugin | None:
        return self._plugins.get(name)

    @property
    def plugins(self) -> list[LeashdPlugin]:
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
            try:
                await plugin.start()
                logger.info("plugin_started", name=plugin.meta.name)
            except Exception as e:
                logger.error("plugin_start_failed", name=plugin.meta.name, error=str(e))
                raise PluginError(
                    f"Plugin {plugin.meta.name} failed to start: {e}"
                ) from e

    async def stop_all(self) -> None:
        for plugin in reversed(self._plugins.values()):
            try:
                await plugin.stop()
            except Exception:
                logger.exception("plugin_stop_failed", name=plugin.meta.name)
