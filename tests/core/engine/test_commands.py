"""Engine tests — /test, /dir, /commit, /plan, /edit commands."""

from unittest.mock import AsyncMock

import pytest

from tests.core.engine.conftest import FakeAgent, _make_git_handler_mock
from tether.core.config import TetherConfig
from tether.core.engine import Engine
from tether.core.events import EventBus
from tether.core.session import SessionManager


class TestHandleCommand:
    @pytest.mark.asyncio
    async def test_plan_command_sets_mode(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "plan", "", "chat1")

        assert "plan mode" in result.lower()
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "plan"
        assert "chat1" not in eng._gatekeeper._auto_approved_chats

    @pytest.mark.asyncio
    async def test_accept_command_sets_mode_and_auto_approve(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "edit", "", "chat1")

        assert "accept edits" in result.lower() or "auto-approve" in result.lower()
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "auto"
        auto_tools = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert auto_tools == {"Write", "Edit", "NotebookEdit"}
        assert "chat1" not in eng._gatekeeper._auto_approved_chats

    @pytest.mark.asyncio
    async def test_status_command_shows_info(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # First send a message to create session state
        await eng.handle_message("user1", "hello", "chat1")

        result = await eng.handle_command("user1", "status", "", "chat1")

        assert "Mode:" in result
        assert "Messages:" in result
        assert "Total cost:" in result
        assert "Auto-approve:" in result

    @pytest.mark.asyncio
    async def test_status_shows_per_tool_auto_approve(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        eng._gatekeeper.enable_tool_auto_approve("chat1", "Write")
        eng._gatekeeper.enable_tool_auto_approve("chat1", "Edit")

        result = await eng.handle_command("user1", "status", "", "chat1")

        assert "Auto-approve: Edit, Write" in result

    @pytest.mark.asyncio
    async def test_status_shows_blanket_auto_approve(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        eng._gatekeeper.enable_auto_approve("chat1")

        result = await eng.handle_command("user1", "status", "", "chat1")

        assert "Auto-approve: on (all tools)" in result

    @pytest.mark.asyncio
    async def test_default_command_sets_mode(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "default", "", "chat1")

        assert "default" in result.lower()
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "default"
        assert "chat1" not in eng._gatekeeper._auto_approved_chats
        assert "chat1" not in eng._gatekeeper._auto_approved_tools

    @pytest.mark.asyncio
    async def test_default_disables_auto_approve(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # Enable auto-approve first via /edit
        await eng.handle_command("user1", "edit", "", "chat1")
        assert eng._gatekeeper._auto_approved_tools.get("chat1") == {
            "Write",
            "Edit",
            "NotebookEdit",
        }

        # Switch to default mode
        await eng.handle_command("user1", "default", "", "chat1")
        assert "chat1" not in eng._gatekeeper._auto_approved_tools

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "foo", "", "chat1")
        assert "Unknown command" in result

    @pytest.mark.asyncio
    async def test_plan_disables_auto_approve(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # Enable auto-approve first
        await eng.handle_command("user1", "edit", "", "chat1")
        assert eng._gatekeeper._auto_approved_tools.get("chat1") == {
            "Write",
            "Edit",
            "NotebookEdit",
        }

        # Switch to plan mode
        await eng.handle_command("user1", "plan", "", "chat1")
        assert "chat1" not in eng._gatekeeper._auto_approved_chats
        assert "chat1" not in eng._gatekeeper._auto_approved_tools

    @pytest.mark.asyncio
    async def test_command_handler_wired_to_connector(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        assert mock_connector._command_handler is not None
        assert mock_connector._auto_approve_handler is not None

    @pytest.mark.asyncio
    async def test_simulate_command_via_connector(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await mock_connector.simulate_command("user1", "status", "", "chat1")
        assert "Mode:" in result

    @pytest.mark.asyncio
    async def test_clear_command_resets_session(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # Establish session state
        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        original_id = session.session_id
        assert session.claude_session_id == "test-session-123"

        # Enable auto-approve so we can verify it gets disabled
        eng._gatekeeper.enable_auto_approve("chat1")
        assert "chat1" in eng._gatekeeper._auto_approved_chats

        # Run /clear
        result = await eng.handle_command("user1", "clear", "", "chat1")

        assert "cleared" in result.lower()
        assert "fresh" in result.lower()
        assert session.is_active is True
        assert session.session_id != original_id
        assert session.claude_session_id is None
        assert session.message_count == 0
        assert session.total_cost == 0.0
        assert "chat1" not in eng._gatekeeper._auto_approved_chats

    @pytest.mark.asyncio
    async def test_clear_preserves_working_directory(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # Establish session, switch to d2
        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        assert session.working_directory == str(d2.resolve())

        # Clear and send a new message
        await eng.handle_command("user1", "clear", "", "chat1")
        await eng.handle_message("user1", "hi again", "chat1")

        session = eng.session_manager.get("user1", "chat1")
        assert session.working_directory == str(d2.resolve())

    @pytest.mark.asyncio
    async def test_clear_preserves_directory_with_sqlite(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        from tether.storage.sqlite import SqliteSessionStore

        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        store = SqliteSessionStore(tmp_path / "test.db")
        await store.setup()
        try:
            eng = Engine(
                connector=mock_connector,
                agent=FakeAgent(),
                config=config,
                session_manager=SessionManager(store=store),
                policy_engine=policy_engine,
                audit=audit_logger,
            )

            await eng.handle_message("user1", "hello", "chat1")
            await eng.handle_command("user1", "dir", "api", "chat1")

            await eng.handle_command("user1", "clear", "", "chat1")

            loaded = await store.load("user1", "chat1")
            assert loaded is not None
            assert loaded.working_directory == str(d2.resolve())
            assert loaded.is_active is True
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_clear_preserves_directory_across_multiple_clears(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")

        # Multiple clears
        await eng.handle_command("user1", "clear", "", "chat1")
        await eng.handle_command("user1", "clear", "", "chat1")

        await eng.handle_message("user1", "still here", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        assert session.working_directory == str(d2.resolve())


class TestTestCommand:
    async def _make_engine_with_plugin(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        from tether.plugins.base import PluginContext
        from tether.plugins.builtin.test_runner import TestRunnerPlugin

        agent = FakeAgent()
        bus = EventBus()
        plugin = TestRunnerPlugin()

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            event_bus=bus,
        )

        ctx = PluginContext(event_bus=bus, config=config)
        await plugin.initialize(ctx)

        return eng, agent

    @pytest.mark.asyncio
    async def test_test_command_sets_mode(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        eng, _ = await self._make_engine_with_plugin(
            config, audit_logger, policy_engine, mock_connector
        )

        await eng.handle_command("user1", "test", "verify login", "chat1")

        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "test"
        assert session.mode_instruction is not None

    @pytest.mark.asyncio
    async def test_test_command_auto_approves_browser_tools(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        from tether.plugins.builtin.browser_tools import BROWSER_MUTATION_TOOLS

        eng, _ = await self._make_engine_with_plugin(
            config, audit_logger, policy_engine, mock_connector
        )

        await eng.handle_command("user1", "test", "", "chat1")

        auto_tools = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        for tool in BROWSER_MUTATION_TOOLS:
            assert tool in auto_tools

    @pytest.mark.asyncio
    async def test_test_command_routes_args_to_agent(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        eng, _ = await self._make_engine_with_plugin(
            config, audit_logger, policy_engine, mock_connector
        )

        await eng.handle_command("user1", "test", "verify login", "chat1")

        assert len(mock_connector.sent_messages) == 1
        assert "verify login" in mock_connector.sent_messages[0]["text"]

    @pytest.mark.asyncio
    async def test_test_command_routes_default_prompt(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        eng, _ = await self._make_engine_with_plugin(
            config, audit_logger, policy_engine, mock_connector
        )

        await eng.handle_command("user1", "test", "", "chat1")

        assert len(mock_connector.sent_messages) == 1
        assert "comprehensive tests" in mock_connector.sent_messages[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_test_command_returns_empty(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        eng, _ = await self._make_engine_with_plugin(
            config, audit_logger, policy_engine, mock_connector
        )

        result = await eng.handle_command("user1", "test", "check it", "chat1")

        assert result == ""

    @pytest.mark.asyncio
    async def test_default_command_clears_test_mode(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        from tether.plugins.builtin.browser_tools import BROWSER_MUTATION_TOOLS

        eng, _ = await self._make_engine_with_plugin(
            config, audit_logger, policy_engine, mock_connector
        )

        await eng.handle_command("user1", "test", "", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "test"
        assert session.mode_instruction is not None
        auto_tools = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert auto_tools >= BROWSER_MUTATION_TOOLS

        await eng.handle_command("user1", "default", "", "chat1")
        assert session.mode == "default"
        assert session.mode_instruction is None
        assert "chat1" not in eng._gatekeeper._auto_approved_tools


class TestDirCommand:
    @pytest.mark.asyncio
    async def test_dir_lists_directories(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "dir", "", "chat1")

        assert result == ""
        assert len(mock_connector.sent_messages) == 1
        msg = mock_connector.sent_messages[0]
        assert msg["text"] == "Select directory:"
        assert msg["buttons"] is not None
        button_texts = [row[0].text for row in msg["buttons"]]
        assert any("tether" in t for t in button_texts)
        assert any("api" in t for t in button_texts)
        assert any("✅" in t for t in button_texts)

    @pytest.mark.asyncio
    async def test_dir_switches_directory(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # Create a session first
        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        session.claude_session_id = "old-session"

        result = await eng.handle_command("user1", "dir", "api", "chat1")

        assert "Switched to api" in result
        assert session.working_directory == str(d2.resolve())
        assert session.claude_session_id is None

    @pytest.mark.asyncio
    async def test_dir_unknown_name_returns_error(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "dir", "nonexistent", "chat1")

        assert "Unknown directory" in result
        assert "Available:" in result

    @pytest.mark.asyncio
    async def test_dir_already_active(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "dir", "tether", "chat1")

        assert "Already in" in result

    @pytest.mark.asyncio
    async def test_status_shows_directory(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        result = await eng.handle_command("user1", "status", "", "chat1")

        assert "Directory:" in result
        assert "tether" in result

    @pytest.mark.asyncio
    async def test_dir_switch_disables_auto_approve(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        eng._gatekeeper.enable_tool_auto_approve("chat1", "Write")

        await eng.handle_command("user1", "dir", "api", "chat1")

        assert "chat1" not in eng._gatekeeper._auto_approved_tools


class TestDirSwitchDataPaths:
    """Verify /dir switch moves audit and storage paths for unpinned configs."""

    @pytest.mark.asyncio
    async def test_audit_path_switches_on_dir_change(
        self, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()
        audit_path = d1 / ".tether" / "audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        from tether.core.safety.audit import AuditLogger

        audit = AuditLogger(audit_path)
        config = TetherConfig(approved_directories=[d1, d2])
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit,
            audit_path_pinned=False,
            storage_path_pinned=True,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "proj2", "chat1")

        expected = d2 / ".tether" / "audit.jsonl"
        assert eng.audit._path == expected

    @pytest.mark.asyncio
    async def test_sqlite_switches_on_dir_change(
        self, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()
        from tether.storage.sqlite import SqliteSessionStore

        db_path = d1 / ".tether" / "tether.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SqliteSessionStore(db_path)
        await store.setup()
        config = TetherConfig(approved_directories=[d1, d2])
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            store=store,
            storage_path_pinned=False,
            audit_path_pinned=True,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "proj2", "chat1")

        expected = str(d2 / ".tether" / "tether.db")
        assert eng._message_store._db_path == expected
        await store.teardown()

    @pytest.mark.asyncio
    async def test_pinned_audit_path_not_switched(
        self, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()
        pinned_path = tmp_path / "global_audit.jsonl"
        from tether.core.safety.audit import AuditLogger

        audit = AuditLogger(pinned_path)
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=pinned_path,
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit,
            audit_path_pinned=True,
            storage_path_pinned=True,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "proj2", "chat1")

        assert eng.audit._path == pinned_path

    @pytest.mark.asyncio
    async def test_dir_switch_creates_tether_dir(
        self, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        from tether.core.safety.audit import AuditLogger

        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=AuditLogger(tmp_path / "audit.jsonl"),
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "proj2", "chat1")

        assert (d2 / ".tether").is_dir()
        assert (d2 / ".tether" / ".gitignore").is_file()


class TestDirButtons:
    @pytest.mark.asyncio
    async def test_dir_sends_buttons_with_connector(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "dir", "", "chat1")

        assert result == ""
        assert len(mock_connector.sent_messages) == 1
        msg = mock_connector.sent_messages[0]
        assert msg["text"] == "Select directory:"
        assert msg["buttons"] is not None
        button_texts = [row[0].text for row in msg["buttons"]]
        assert any("tether" in t for t in button_texts)
        assert any("api" in t for t in button_texts)

    @pytest.mark.asyncio
    async def test_dir_shows_active_marker_on_button(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_command("user1", "dir", "", "chat1")

        msg = mock_connector.sent_messages[0]
        # First directory is the default (active)
        active_buttons = [row[0].text for row in msg["buttons"] if "✅" in row[0].text]
        assert len(active_buttons) == 1

    @pytest.mark.asyncio
    async def test_dir_callback_data_uses_prefix(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_command("user1", "dir", "", "chat1")

        msg = mock_connector.sent_messages[0]
        callback_datas = [row[0].callback_data for row in msg["buttons"]]
        assert all(cd.startswith("dir:") for cd in callback_datas)

    @pytest.mark.asyncio
    async def test_dir_falls_back_to_text_without_connector(
        self, audit_logger, policy_engine, tmp_path
    ):
        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "dir", "", "chat1")

        assert "Directories:" in result
        assert "tether" in result
        assert "api" in result

    @pytest.mark.asyncio
    async def test_dir_single_directory_falls_back_to_text(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "tether"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "dir", "", "chat1")

        assert "Directories:" in result
        assert "tether" in result


class TestGitCommandWithoutHandler:
    """Verify /git returns friendly message when no git handler is configured."""

    @pytest.mark.asyncio
    async def test_git_command_without_handler_returns_not_available(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "git", "status", "chat1")
        assert result == "Git commands not available."

    @pytest.mark.asyncio
    async def test_git_command_without_handler_no_args(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "git", "", "chat1")
        assert result == "Git commands not available."


class TestActiveDirNameFallback:
    """Verify _active_dir_name falls back to basename when no match."""

    @pytest.mark.asyncio
    async def test_active_dir_name_returns_known_name(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "myproject"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")

        name = eng._active_dir_name(session)
        assert name == "myproject"

    @pytest.mark.asyncio
    async def test_active_dir_name_unknown_dir_shows_basename(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "myproject"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        # Point session to a directory not in the config
        session.working_directory = str(tmp_path / "unknown_project")

        name = eng._active_dir_name(session)
        assert name == "unknown_project"


class TestSmartCommit:
    @pytest.mark.asyncio
    async def test_bare_git_commit_triggers_smart_flow(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        git_handler = _make_git_handler_mock()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            git_handler=git_handler,
        )

        result = await eng.handle_command("user1", "git", "commit", "chat1")

        assert result == ""
        analyzing_msgs = [
            m
            for m in mock_connector.sent_messages
            if "analyzing" in m.get("text", "").lower()
        ]
        assert len(analyzing_msgs) == 1
        git_handler.handle_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_git_commit_with_message_goes_through_handler(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        git_handler = _make_git_handler_mock()
        git_handler.handle_command = AsyncMock(return_value="committed")
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            git_handler=git_handler,
        )

        result = await eng.handle_command("user1", "git", "commit fix typo", "chat1")

        assert result == "committed"
        git_handler.handle_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_smart_commit_auto_approves_git_commands(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        git_handler = _make_git_handler_mock()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            git_handler=git_handler,
        )

        await eng.handle_command("user1", "git", "commit", "chat1")

        auto_tools = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Bash::git diff" in auto_tools
        assert "Bash::git status" in auto_tools
        assert "Bash::git commit" in auto_tools

    @pytest.mark.asyncio
    async def test_smart_commit_without_git_handler(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "git", "commit", "chat1")

        assert result == "Git commands not available."

    @pytest.mark.asyncio
    async def test_smart_commit_sends_prompt_to_agent(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        git_handler = _make_git_handler_mock()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            git_handler=git_handler,
        )

        await eng.handle_command("user1", "git", "commit", "chat1")

        # Agent should have been called — check the response was streamed
        agent_msgs = [
            m
            for m in mock_connector.sent_messages
            if "conventional commit" in m.get("text", "").lower()
            or "echo:" in m.get("text", "").lower()
        ]
        assert len(agent_msgs) >= 1


class TestPlanWithArgs:
    @pytest.mark.asyncio
    async def test_plan_with_args_forwards_to_agent(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command(
            "user1", "plan", "create a login page", "chat1"
        )

        assert result == ""
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "plan"
        confirmation_msgs = [
            m
            for m in mock_connector.sent_messages
            if "plan mode" in m.get("text", "").lower()
        ]
        assert len(confirmation_msgs) == 1
        agent_msgs = [
            m
            for m in mock_connector.sent_messages
            if "create a login page" in m.get("text", "").lower()
        ]
        assert len(agent_msgs) >= 1

    @pytest.mark.asyncio
    async def test_plan_without_args_returns_confirmation(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "plan", "", "chat1")

        assert "plan mode" in result.lower()
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "plan"

    @pytest.mark.asyncio
    async def test_plan_with_whitespace_only_args_returns_confirmation(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "plan", "   ", "chat1")

        assert "plan mode" in result.lower()

    @pytest.mark.asyncio
    async def test_plan_with_args_no_connector(
        self, config, audit_logger, policy_engine
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=None,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "plan", "build feature", "chat1")

        assert result == ""


class TestEditWithArgs:
    @pytest.mark.asyncio
    async def test_edit_with_args_forwards_to_agent(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "edit", "fix the auth bug", "chat1")

        assert result == ""
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "auto"
        confirmation_msgs = [
            m
            for m in mock_connector.sent_messages
            if "accept edits" in m.get("text", "").lower()
        ]
        assert len(confirmation_msgs) == 1
        agent_msgs = [
            m
            for m in mock_connector.sent_messages
            if "fix the auth bug" in m.get("text", "").lower()
        ]
        assert len(agent_msgs) >= 1

    @pytest.mark.asyncio
    async def test_edit_without_args_returns_confirmation(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "edit", "", "chat1")

        assert "accept edits" in result.lower() or "auto-approve" in result.lower()
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "auto"

    @pytest.mark.asyncio
    async def test_edit_with_args_auto_approves_tools(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_command("user1", "edit", "fix bug", "chat1")

        auto_tools = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Write" in auto_tools
        assert "Edit" in auto_tools
        assert "NotebookEdit" in auto_tools

    @pytest.mark.asyncio
    async def test_edit_with_whitespace_only_args_returns_confirmation(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_command("user1", "edit", "   ", "chat1")

        assert "accept edits" in result.lower() or "auto-approve" in result.lower()
