"""Tests for the BrowserToolsPlugin."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tether.core.events import (
    TOOL_ALLOWED,
    TOOL_DENIED,
    TOOL_GATED,
    Event,
    EventBus,
)
from tether.plugins.base import PluginContext
from tether.plugins.builtin.browser_tools import (
    ALL_BROWSER_TOOLS,
    BROWSER_MUTATION_TOOLS,
    BROWSER_READONLY_TOOLS,
    BrowserToolsPlugin,
    is_browser_tool,
)


class TestBrowserToolConstants:
    def test_all_browser_tools_count(self):
        assert len(ALL_BROWSER_TOOLS) == 28

    def test_readonly_count(self):
        assert len(BROWSER_READONLY_TOOLS) == 7

    def test_mutation_count(self):
        assert len(BROWSER_MUTATION_TOOLS) == 21

    def test_no_overlap(self):
        assert frozenset() == BROWSER_READONLY_TOOLS & BROWSER_MUTATION_TOOLS

    def test_union_equals_all(self):
        assert BROWSER_READONLY_TOOLS | BROWSER_MUTATION_TOOLS == ALL_BROWSER_TOOLS

    def test_is_browser_tool_readonly(self):
        assert is_browser_tool("browser_snapshot") is True

    def test_is_browser_tool_mutation(self):
        assert is_browser_tool("browser_click") is True

    def test_is_browser_tool_negative(self):
        assert is_browser_tool("Read") is False

    def test_is_browser_tool_empty(self):
        assert is_browser_tool("") is False

    def test_is_browser_tool_mcp_prefixed(self):
        assert is_browser_tool("mcp__playwright__browser_navigate") is True
        assert is_browser_tool("mcp__playwright__browser_snapshot") is True

    def test_is_browser_tool_mcp_prefixed_negative(self):
        assert is_browser_tool("mcp__playwright__some_other_tool") is False

    def test_new_mutation_tools_present(self):
        assert "browser_fill_form" in BROWSER_MUTATION_TOOLS
        assert "browser_evaluate" in BROWSER_MUTATION_TOOLS
        assert "browser_tabs" in BROWSER_MUTATION_TOOLS


class TestBrowserToolsPlugin:
    @pytest.mark.asyncio
    async def test_subscribes_to_events(self, config):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        assert len(bus._handlers.get(TOOL_GATED, [])) == 1
        assert len(bus._handlers.get(TOOL_ALLOWED, [])) == 1
        assert len(bus._handlers.get(TOOL_DENIED, [])) == 1

    @pytest.mark.asyncio
    async def test_gated_handler_fires_for_browser_tool(self, config):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        with patch.object(plugin, "_on_tool_gated", wraps=plugin._on_tool_gated) as m:
            bus.unsubscribe(TOOL_GATED, plugin._on_tool_gated)
            bus.subscribe(TOOL_GATED, m)
            await bus.emit(
                Event(
                    name=TOOL_GATED,
                    data={"tool_name": "browser_click", "session_id": "s1"},
                )
            )
            m.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gated_handler_skips_non_browser_tool(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_GATED,
                data={"tool_name": "Read", "session_id": "s1"},
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_gated" not in captured.out

    @pytest.mark.asyncio
    async def test_gated_mutation_flag(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_GATED,
                data={"tool_name": "browser_navigate", "session_id": "s1"},
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_gated" in captured.out
        assert "is_mutation=True" in captured.out

    @pytest.mark.asyncio
    async def test_gated_readonly_flag(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_GATED,
                data={"tool_name": "browser_snapshot", "session_id": "s1"},
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_gated" in captured.out
        assert "is_mutation=False" in captured.out

    @pytest.mark.asyncio
    async def test_allowed_handler_fires_for_browser_tool(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_ALLOWED,
                data={"tool_name": "browser_snapshot", "session_id": "s1"},
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_allowed" in captured.out

    @pytest.mark.asyncio
    async def test_allowed_handler_skips_non_browser_tool(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_ALLOWED,
                data={"tool_name": "Bash", "session_id": "s1"},
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_allowed" not in captured.out

    @pytest.mark.asyncio
    async def test_denied_handler_fires_for_browser_tool(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_DENIED,
                data={
                    "tool_name": "browser_navigate",
                    "session_id": "s1",
                    "reason": "policy denied",
                },
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_denied" in captured.out

    @pytest.mark.asyncio
    async def test_denied_handler_skips_non_browser_tool(self, config, capsys):
        bus = EventBus()
        ctx = PluginContext(event_bus=bus, config=config)
        plugin = BrowserToolsPlugin()
        await plugin.initialize(ctx)

        await bus.emit(
            Event(
                name=TOOL_DENIED,
                data={"tool_name": "Write", "session_id": "s1", "reason": "blocked"},
            )
        )

        captured = capsys.readouterr()
        assert "browser_tool_denied" not in captured.out

    @pytest.mark.asyncio
    async def test_start_completes(self):
        plugin = BrowserToolsPlugin()
        await plugin.start()

    @pytest.mark.asyncio
    async def test_stop_completes(self):
        plugin = BrowserToolsPlugin()
        await plugin.stop()

    def test_meta(self):
        plugin = BrowserToolsPlugin()
        assert plugin.meta.name == "browser_tools"
        assert plugin.meta.version == "0.1.0"
