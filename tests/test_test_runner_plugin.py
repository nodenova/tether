"""Tests for the TestRunnerPlugin."""

from unittest.mock import MagicMock

import pytest

from tether.core.events import COMMAND_TEST, Event
from tether.core.session import Session
from tether.plugins.base import PluginContext
from tether.plugins.builtin.browser_tools import BROWSER_MUTATION_TOOLS
from tether.plugins.builtin.test_runner import TEST_MODE_INSTRUCTION, TestRunnerPlugin


@pytest.fixture
def plugin():
    return TestRunnerPlugin()


@pytest.fixture
async def initialized_plugin(plugin, config, event_bus):
    ctx = PluginContext(event_bus=event_bus, config=config)
    await plugin.initialize(ctx)
    return plugin


@pytest.fixture
def session(tmp_path):
    return Session(
        session_id="test-session",
        user_id="user1",
        chat_id="chat1",
        working_directory=str(tmp_path),
    )


@pytest.fixture
def gatekeeper():
    mock = MagicMock()
    mock.enable_tool_auto_approve = MagicMock()
    return mock


class TestTestRunnerPlugin:
    @pytest.mark.asyncio
    async def test_plugin_sets_mode_and_instruction(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "verify login",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        assert session.mode == "test"
        assert session.mode_instruction == TEST_MODE_INSTRUCTION

    @pytest.mark.asyncio
    async def test_plugin_auto_approves_browser_tools(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        for tool in BROWSER_MUTATION_TOOLS:
            gatekeeper.enable_tool_auto_approve.assert_any_call("chat1", tool)

    @pytest.mark.asyncio
    async def test_plugin_builds_prompt_from_args(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "verify login flow",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        assert event.data["prompt"] == "verify login flow"

    @pytest.mark.asyncio
    async def test_plugin_builds_default_prompt(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        assert (
            event.data["prompt"] == "Run comprehensive tests for the current codebase."
        )

    @pytest.mark.asyncio
    async def test_plugin_meta(self, plugin):
        assert plugin.meta.name == "test_runner"
        assert plugin.meta.version == "0.1.0"

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self, plugin):
        await plugin.start()
        await plugin.stop()
