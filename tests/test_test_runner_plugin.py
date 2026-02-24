"""Tests for the TestRunnerPlugin."""

from unittest.mock import MagicMock

import pytest

from tether.core.events import COMMAND_TEST, TEST_STARTED, Event, EventBus
from tether.core.session import Session
from tether.plugins.base import PluginContext
from tether.plugins.builtin.browser_tools import BROWSER_MUTATION_TOOLS
from tether.plugins.builtin.test_runner import (
    TEST_BASH_AUTO_APPROVE,
    TEST_MODE_INSTRUCTION,
    TestConfig,
    TestRunnerPlugin,
    _build_test_prompt,
    build_test_instruction,
    parse_test_args,
)


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


# --- TestConfig ---


class TestTestConfig:
    def test_defaults(self):
        c = TestConfig()
        assert c.app_url is None
        assert c.dev_server_command is None
        assert c.test_directory is None
        assert c.framework is None
        assert c.focus is None
        assert c.include_e2e is True
        assert c.include_unit is True
        assert c.include_backend is True

    def test_frozen(self):
        from pydantic import ValidationError

        c = TestConfig(app_url="http://localhost:3000")
        with pytest.raises(ValidationError, match="frozen"):
            c.app_url = "http://other"  # type: ignore[misc]

    def test_model_dump(self):
        c = TestConfig(app_url="http://localhost:3000", framework="next")
        d = c.model_dump()
        assert d["app_url"] == "http://localhost:3000"
        assert d["framework"] == "next"
        assert d["include_e2e"] is True


# --- parse_test_args ---


class TestParseTestArgs:
    def test_empty_args(self):
        c = parse_test_args("")
        assert c == TestConfig()

    def test_whitespace_only(self):
        c = parse_test_args("   ")
        assert c == TestConfig()

    def test_url_long_flag(self):
        c = parse_test_args("--url http://localhost:3000")
        assert c.app_url == "http://localhost:3000"

    def test_url_short_flag(self):
        c = parse_test_args("-u http://localhost:8080")
        assert c.app_url == "http://localhost:8080"

    def test_server_flag(self):
        c = parse_test_args("--server 'npm run dev'")
        assert c.dev_server_command == "npm run dev"

    def test_server_short_flag(self):
        c = parse_test_args("-s 'yarn start'")
        assert c.dev_server_command == "yarn start"

    def test_dir_flag(self):
        c = parse_test_args("--dir tests/e2e")
        assert c.test_directory == "tests/e2e"

    def test_framework_flag(self):
        c = parse_test_args("--framework next")
        assert c.framework == "next"

    def test_framework_short_flag(self):
        c = parse_test_args("-f react")
        assert c.framework == "react"

    def test_no_e2e_flag(self):
        c = parse_test_args("--no-e2e")
        assert c.include_e2e is False
        assert c.include_unit is True

    def test_no_unit_flag(self):
        c = parse_test_args("--no-unit")
        assert c.include_unit is False
        assert c.include_e2e is True

    def test_no_backend_flag(self):
        c = parse_test_args("--no-backend")
        assert c.include_backend is False

    def test_focus_text(self):
        c = parse_test_args("verify checkout flow")
        assert c.focus == "verify checkout flow"

    def test_mixed_flags_and_focus(self):
        c = parse_test_args(
            "--url http://localhost:3000 --framework next verify checkout"
        )
        assert c.app_url == "http://localhost:3000"
        assert c.framework == "next"
        assert c.focus == "verify checkout"

    def test_all_flags(self):
        c = parse_test_args(
            "--url http://localhost:3000 "
            "--server 'npm run dev' "
            "--dir tests/ "
            "--framework next "
            "--no-backend "
            "check login"
        )
        assert c.app_url == "http://localhost:3000"
        assert c.dev_server_command == "npm run dev"
        assert c.test_directory == "tests/"
        assert c.framework == "next"
        assert c.include_backend is False
        assert c.include_e2e is True
        assert c.include_unit is True
        assert c.focus == "check login"

    def test_malformed_quotes_fallback(self):
        c = parse_test_args("verify 'unclosed quote")
        assert c.focus == "verify 'unclosed quote"


# --- build_test_instruction ---


class TestBuildTestInstruction:
    def test_default_has_all_phases(self):
        instruction = build_test_instruction(TestConfig())
        assert "PHASE 1" in instruction
        assert "PHASE 2" in instruction
        assert "PHASE 3" in instruction
        assert "PHASE 4" in instruction
        assert "PHASE 5" in instruction
        assert "PHASE 6" in instruction
        assert "PHASE 7" in instruction
        assert "PHASE 8" in instruction
        assert "PHASE 9" in instruction

    def test_no_e2e_skips_server_smoke_browser(self):
        instruction = build_test_instruction(TestConfig(include_e2e=False))
        assert "PHASE 1" in instruction  # discovery always present
        assert "SERVER STARTUP" not in instruction
        assert "SMOKE TEST" not in instruction
        assert "E2E BROWSER" not in instruction
        assert "UNIT" in instruction
        assert "BACKEND" in instruction

    def test_no_unit_skips_unit_phase(self):
        instruction = build_test_instruction(TestConfig(include_unit=False))
        assert "UNIT & INTEGRATION" not in instruction
        assert "SERVER STARTUP" in instruction

    def test_no_backend_skips_backend_phase(self):
        instruction = build_test_instruction(TestConfig(include_backend=False))
        assert "BACKEND VERIFICATION" not in instruction
        assert "UNIT" in instruction

    def test_url_in_hints(self):
        instruction = build_test_instruction(
            TestConfig(app_url="http://localhost:3000")
        )
        assert "http://localhost:3000" in instruction
        assert "USER HINTS" in instruction

    def test_framework_in_hints(self):
        instruction = build_test_instruction(TestConfig(framework="next"))
        assert "Framework: next" in instruction

    def test_focus_in_hints(self):
        instruction = build_test_instruction(TestConfig(focus="verify login"))
        assert "Focus area: verify login" in instruction

    def test_server_command_in_hints(self):
        instruction = build_test_instruction(
            TestConfig(dev_server_command="npm run dev")
        )
        assert "Dev server command: npm run dev" in instruction

    def test_test_directory_in_hints(self):
        instruction = build_test_instruction(TestConfig(test_directory="tests/e2e"))
        assert "Test directory: tests/e2e" in instruction

    def test_no_hints_when_default(self):
        instruction = build_test_instruction(TestConfig())
        assert "USER HINTS" not in instruction


# --- _build_test_prompt ---


class TestBuildTestPrompt:
    def test_default_prompt(self):
        prompt = _build_test_prompt(TestConfig())
        assert prompt == "Run comprehensive tests for the current codebase."

    def test_focus_becomes_prompt(self):
        prompt = _build_test_prompt(TestConfig(focus="verify login flow"))
        assert "verify login flow" in prompt

    def test_url_appended(self):
        prompt = _build_test_prompt(TestConfig(app_url="http://localhost:3000"))
        assert "http://localhost:3000" in prompt

    def test_framework_appended(self):
        prompt = _build_test_prompt(TestConfig(framework="next"))
        assert "next" in prompt

    def test_focus_with_url_and_framework(self):
        prompt = _build_test_prompt(
            TestConfig(
                focus="check checkout",
                app_url="http://localhost:3000",
                framework="next",
            )
        )
        assert "check checkout" in prompt
        assert "http://localhost:3000" in prompt
        assert "next" in prompt

    def test_no_focus_with_url(self):
        prompt = _build_test_prompt(TestConfig(app_url="http://localhost:3000"))
        assert prompt.startswith("Run comprehensive tests")
        assert "http://localhost:3000" in prompt


# --- TestRunnerPlugin ---


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
        assert "TEST MODE" in session.mode_instruction
        assert "PHASE 1" in session.mode_instruction

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

        assert "verify login flow" in event.data["prompt"]

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
        assert plugin.meta.version == "0.2.0"

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self, plugin):
        await plugin.start()
        await plugin.stop()

    @pytest.mark.asyncio
    async def test_plugin_auto_approves_bash_commands(
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

        for key in TEST_BASH_AUTO_APPROVE:
            gatekeeper.enable_tool_auto_approve.assert_any_call("chat1", key)

    @pytest.mark.asyncio
    async def test_plugin_auto_approves_write_edit(
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

        gatekeeper.enable_tool_auto_approve.assert_any_call("chat1", "Write")
        gatekeeper.enable_tool_auto_approve.assert_any_call("chat1", "Edit")

    @pytest.mark.asyncio
    async def test_plugin_parses_url_from_args(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "--url http://localhost:3000 verify checkout",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        assert "http://localhost:3000" in event.data["prompt"]
        assert "verify checkout" in event.data["prompt"]
        assert "http://localhost:3000" in session.mode_instruction

    @pytest.mark.asyncio
    async def test_plugin_emits_test_started(self, plugin, config, session, gatekeeper):
        event_bus = EventBus()
        ctx = PluginContext(event_bus=event_bus, config=config)
        await plugin.initialize(ctx)

        received: list[Event] = []

        async def capture(ev: Event) -> None:
            received.append(ev)

        event_bus.subscribe(TEST_STARTED, capture)

        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "--url http://localhost:3000",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        assert len(received) == 1
        assert received[0].name == TEST_STARTED
        assert received[0].data["chat_id"] == "chat1"
        assert received[0].data["config"]["app_url"] == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_plugin_instruction_uses_config(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "--no-e2e --framework django",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        # E2E phases should be absent
        assert "SERVER STARTUP" not in session.mode_instruction
        assert "SMOKE TEST" not in session.mode_instruction
        assert "E2E BROWSER" not in session.mode_instruction
        # Framework hint should be present
        assert "Framework: django" in session.mode_instruction


class TestBackwardCompat:
    def test_test_mode_instruction_is_default(self):
        """TEST_MODE_INSTRUCTION matches default config output for backward compat."""
        assert build_test_instruction(TestConfig()) == TEST_MODE_INSTRUCTION
        assert "TEST MODE" in TEST_MODE_INSTRUCTION


class TestParseTestArgsEdgeCases:
    """Edge cases for argument parsing that guard against misuse."""

    def test_flag_value_looks_like_flag(self):
        """--url followed by another flag should not consume it as the value."""
        c = parse_test_args("--url --framework next")
        assert c.app_url is None
        assert c.framework == "next"

    def test_all_phases_disabled_resets(self):
        """Disabling all test phases resets them all to enabled with a warning."""
        c = parse_test_args("--no-e2e --no-unit --no-backend")
        assert c.include_e2e is True
        assert c.include_unit is True
        assert c.include_backend is True

    def test_bare_flag_at_end(self):
        """--url at end of string with no value should not crash."""
        c = parse_test_args("check login --url")
        # --url has no next token, so it becomes part of focus
        assert c.app_url is None
        assert "--url" in (c.focus or "")

    def test_duplicate_flags_last_wins(self):
        """When the same flag is given twice, the last value wins."""
        c = parse_test_args("--url http://a --url http://b")
        assert c.app_url == "http://b"

    def test_server_flag_value_looks_like_flag(self):
        """--server followed by another flag should not consume it."""
        c = parse_test_args("--server --no-e2e")
        assert c.dev_server_command is None
        assert c.include_e2e is False

    def test_dir_flag_value_looks_like_flag(self):
        """--dir followed by another flag should not consume it."""
        c = parse_test_args("--dir --framework react")
        assert c.test_directory is None
        assert c.framework == "react"


class TestAutoApproveTotalCount:
    """Verify the exact number of auto-approve calls in test mode."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def gatekeeper(self):
        mock = MagicMock()
        mock.enable_tool_auto_approve = MagicMock()
        return mock

    @pytest.fixture
    def session(self):
        return Session(
            session_id="count-session",
            user_id="u1",
            chat_id="chat1",
            working_directory="/tmp",
        )

    @pytest.fixture
    async def initialized_plugin(self, event_bus):
        from tether.core.config import TetherConfig

        config = TetherConfig(approved_directories=["/tmp"])
        p = TestRunnerPlugin()
        ctx = PluginContext(event_bus=event_bus, config=config)
        await p.initialize(ctx)
        return p

    @pytest.mark.asyncio
    async def test_auto_approve_total_call_count(
        self, initialized_plugin, event_bus, session, gatekeeper
    ):
        """Total auto-approve calls must match browser + bash + Write + Edit."""
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

        expected = len(BROWSER_MUTATION_TOOLS) + len(TEST_BASH_AUTO_APPROVE) + 2
        assert gatekeeper.enable_tool_auto_approve.call_count == expected


class TestTestStartedEventData:
    """Verify TEST_STARTED event contains complete config data."""

    @pytest.mark.asyncio
    async def test_full_config_in_event(self):
        from tether.core.config import TetherConfig

        event_bus = EventBus()
        config = TetherConfig(approved_directories=["/tmp"])
        p = TestRunnerPlugin()
        ctx = PluginContext(event_bus=event_bus, config=config)
        await p.initialize(ctx)

        received: list[Event] = []

        async def capture(ev: Event) -> None:
            received.append(ev)

        event_bus.subscribe(TEST_STARTED, capture)

        session = Session(
            session_id="event-session",
            user_id="u1",
            chat_id="chat1",
            working_directory="/tmp",
        )
        gatekeeper = MagicMock()
        gatekeeper.enable_tool_auto_approve = MagicMock()

        event = Event(
            name=COMMAND_TEST,
            data={
                "session": session,
                "chat_id": "chat1",
                "args": "--url http://localhost:3000 --framework next --no-backend check login",
                "gatekeeper": gatekeeper,
                "prompt": "",
            },
        )
        await event_bus.emit(event)

        assert len(received) == 1
        cfg = received[0].data["config"]
        assert cfg["app_url"] == "http://localhost:3000"
        assert cfg["framework"] == "next"
        assert cfg["include_backend"] is False
        assert cfg["include_e2e"] is True
        assert cfg["include_unit"] is True
        assert cfg["focus"] == "check login"
        assert cfg["dev_server_command"] is None
        assert cfg["test_directory"] is None
