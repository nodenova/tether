"""Browser tools plugin — observability for Playwright MCP browser tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from tether.core.events import TOOL_ALLOWED, TOOL_DENIED, TOOL_GATED
from tether.core.safety.gatekeeper import _normalize_tool_name
from tether.plugins.base import PluginMeta, TetherPlugin

if TYPE_CHECKING:
    from tether.core.events import Event
    from tether.plugins.base import PluginContext

logger = structlog.get_logger()

BROWSER_READONLY_TOOLS: frozenset[str] = frozenset(
    {
        "browser_snapshot",
        "browser_take_screenshot",
        "browser_console_messages",
        "browser_network_requests",
        "browser_tab_list",
        "browser_wait_for",
        "browser_generate_playwright_test",
    }
)

BROWSER_MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "browser_navigate",
        "browser_navigate_back",
        "browser_navigate_forward",
        "browser_click",
        "browser_type",
        "browser_hover",
        "browser_drag",
        "browser_press_key",
        "browser_select_option",
        "browser_file_upload",
        "browser_handle_dialog",
        "browser_fill_form",
        "browser_evaluate",
        "browser_tabs",
        "browser_tab_new",
        "browser_tab_select",
        "browser_tab_close",
        "browser_resize",
        "browser_pdf_save",
        "browser_close",
        "browser_install",
    }
)

ALL_BROWSER_TOOLS: frozenset[str] = BROWSER_READONLY_TOOLS | BROWSER_MUTATION_TOOLS


def is_browser_tool(tool_name: str) -> bool:
    """Check if a tool is a browser tool, normalizing MCP prefixes."""
    return _normalize_tool_name(tool_name) in ALL_BROWSER_TOOLS


class BrowserToolsPlugin(TetherPlugin):
    meta = PluginMeta(
        name="browser_tools",
        version="0.1.0",
        description="Observability for Playwright MCP browser tools",
    )

    async def initialize(self, context: PluginContext) -> None:
        context.event_bus.subscribe(TOOL_GATED, self._on_tool_gated)
        context.event_bus.subscribe(TOOL_ALLOWED, self._on_tool_allowed)
        context.event_bus.subscribe(TOOL_DENIED, self._on_tool_denied)

    async def start(self) -> None:
        logger.info(
            "browser_tools_plugin_ready",
            readonly_count=len(BROWSER_READONLY_TOOLS),
            mutation_count=len(BROWSER_MUTATION_TOOLS),
            total_count=len(ALL_BROWSER_TOOLS),
        )

    async def stop(self) -> None:
        pass

    async def _on_tool_gated(self, event: Event) -> None:
        tool_name = event.data.get("tool_name", "")
        if not is_browser_tool(tool_name):
            return
        normalized = _normalize_tool_name(tool_name)
        logger.info(
            "browser_tool_gated",
            tool_name=tool_name,
            is_mutation=normalized in BROWSER_MUTATION_TOOLS,
            session_id=event.data.get("session_id", "unknown"),
        )

    async def _on_tool_allowed(self, event: Event) -> None:
        tool_name = event.data.get("tool_name", "")
        if not is_browser_tool(tool_name):
            return
        logger.info(
            "browser_tool_allowed",
            tool_name=tool_name,
            session_id=event.data.get("session_id", "unknown"),
        )

    async def _on_tool_denied(self, event: Event) -> None:
        tool_name = event.data.get("tool_name", "")
        if not is_browser_tool(tool_name):
            return
        logger.warning(
            "browser_tool_denied",
            tool_name=tool_name,
            reason=event.data.get("reason", ""),
            session_id=event.data.get("session_id", "unknown"),
        )
