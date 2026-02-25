"""Tests for the central engine — safety hook wiring."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from tether.agents.base import AgentResponse, BaseAgent
from tether.core.config import TetherConfig
from tether.core.engine import Engine
from tether.core.events import (
    MESSAGE_IN,
    MESSAGE_OUT,
    TOOL_ALLOWED,
    TOOL_DENIED,
    EventBus,
)
from tether.core.interactions import InteractionCoordinator
from tether.core.safety.approvals import ApprovalCoordinator
from tether.core.session import SessionManager
from tether.exceptions import AgentError
from tether.middleware.base import MessageContext, MiddlewareChain


class FakeAgent(BaseAgent):
    """Agent that captures the can_use_tool callback for inspection."""

    def __init__(self, *, fail=False):
        self.last_can_use_tool = None
        self._fail = fail

    async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
        self.last_can_use_tool = can_use_tool
        if self._fail:
            raise AgentError("Agent crashed")
        return AgentResponse(
            content=f"Echo: {prompt}",
            session_id="test-session-123",
            cost=0.01,
        )

    async def cancel(self, session_id):
        pass

    async def shutdown(self):
        pass


@pytest.fixture
def fake_agent():
    return FakeAgent()


@pytest.fixture
def engine(config, fake_agent, policy_engine, audit_logger):
    return Engine(
        connector=None,
        agent=fake_agent,
        config=config,
        session_manager=SessionManager(),
        policy_engine=policy_engine,
        audit=audit_logger,
    )


class TestEngineMessageHandling:
    @pytest.mark.asyncio
    async def test_handle_message_returns_response(self, engine):
        result = await engine.handle_message("user1", "hello", "chat1")
        assert "Echo: hello" in result

    @pytest.mark.asyncio
    async def test_handle_message_creates_session(self, engine):
        await engine.handle_message("user1", "hello", "chat1")
        session = engine.session_manager.get("user1", "chat1")
        assert session is not None
        assert session.message_count == 1

    @pytest.mark.asyncio
    async def test_handle_message_updates_session_cost(self, engine):
        await engine.handle_message("user1", "hello", "chat1")
        session = engine.session_manager.get("user1", "chat1")
        assert session.total_cost == 0.01


class TestSafetyHookWiring:
    @pytest.mark.asyncio
    async def test_can_use_tool_callback_provided(self, engine, fake_agent):
        await engine.handle_message("user1", "hello", "chat1")
        assert fake_agent.last_can_use_tool is not None

    @pytest.mark.asyncio
    async def test_read_tool_allowed(self, engine, fake_agent, tmp_dir):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        result = await hook("Read", {"file_path": str(tmp_dir / "foo.py")}, None)
        assert result.behavior == "allow"

    @pytest.mark.asyncio
    async def test_sandbox_violation_denied(self, engine, fake_agent):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        result = await hook("Read", {"file_path": "/etc/passwd"}, None)
        assert result.behavior == "deny"
        assert "outside allowed" in result.message

    @pytest.mark.asyncio
    async def test_destructive_bash_denied(self, engine, fake_agent):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        result = await hook("Bash", {"command": "rm -rf /"}, None)
        assert result.behavior == "deny"

    @pytest.mark.asyncio
    async def test_credential_read_denied(self, engine, fake_agent, tmp_dir):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        result = await hook(
            "Read",
            {"file_path": str(tmp_dir / ".env")},
            None,
        )
        assert result.behavior == "deny"

    @pytest.mark.asyncio
    async def test_file_write_requires_approval_denied_without_coordinator(
        self, engine, fake_agent, tmp_dir
    ):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        result = await hook(
            "Write",
            {"file_path": str(tmp_dir / "main.py")},
            None,
        )
        # No approval coordinator — denied as safe default
        assert result.behavior == "deny"
        assert "approval" in result.message.lower()

    @pytest.mark.asyncio
    async def test_git_status_bash_allowed(self, engine, fake_agent):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        result = await hook("Bash", {"command": "git status"}, None)
        assert result.behavior == "allow"

    @pytest.mark.asyncio
    async def test_audit_log_written(self, engine, fake_agent, tmp_dir):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        await hook("Read", {"file_path": str(tmp_dir / "foo.py")}, None)

        audit_path = engine.audit._path
        assert audit_path.exists()
        content = audit_path.read_text()
        assert "tool_attempt" in content


class TestEngineEvents:
    @pytest.mark.asyncio
    async def test_message_in_event_emitted(self, config, fake_agent, audit_logger):
        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(MESSAGE_IN, capture)

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )
        await eng.handle_message("user1", "hello", "chat1")

        assert len(events) == 1
        assert events[0].data["user_id"] == "user1"

    @pytest.mark.asyncio
    async def test_message_out_event_emitted(self, config, fake_agent, audit_logger):
        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(MESSAGE_OUT, capture)

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )
        await eng.handle_message("user1", "hello", "chat1")

        assert len(events) == 1
        assert "Echo: hello" in events[0].data["content"]

    @pytest.mark.asyncio
    async def test_tool_allowed_event_emitted(
        self, config, fake_agent, audit_logger, tmp_dir
    ):
        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(TOOL_ALLOWED, capture)

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        await hook("Bash", {"command": "git status"}, None)

        assert len(events) == 1
        assert events[0].data["tool_name"] == "Bash"

    @pytest.mark.asyncio
    async def test_tool_denied_event_emitted(self, config, fake_agent, audit_logger):
        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(TOOL_DENIED, capture)

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        await hook("Read", {"file_path": "/etc/passwd"}, None)

        assert len(events) == 1
        assert events[0].data["tool_name"] == "Read"


class TestEngineApprovalFlow:
    @pytest.mark.asyncio
    async def test_approval_granted_allows_tool(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector, tmp_dir
    ):
        coordinator = ApprovalCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        async def approve():
            await asyncio.sleep(0.05)
            req = mock_connector.approval_requests[0]
            await coordinator.resolve_approval(req["approval_id"], True)

        task = asyncio.create_task(approve())
        result = await hook("Write", {"file_path": str(tmp_dir / "main.py")}, None)
        await task
        assert result.behavior == "allow"

    @pytest.mark.asyncio
    async def test_approval_denied_blocks_tool(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector, tmp_dir
    ):
        coordinator = ApprovalCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        async def deny():
            await asyncio.sleep(0.05)
            req = mock_connector.approval_requests[0]
            await coordinator.resolve_approval(req["approval_id"], False)

        task = asyncio.create_task(deny())
        result = await hook("Write", {"file_path": str(tmp_dir / "main.py")}, None)
        await task
        assert result.behavior == "deny"


class TestEngineErrorHandling:
    @pytest.mark.asyncio
    async def test_agent_error_returns_error_message(self, config, audit_logger):
        failing_agent = FakeAgent(fail=True)
        eng = Engine(
            connector=None,
            agent=failing_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )
        result = await eng.handle_message("user1", "hello", "chat1")
        assert "Error:" in result
        assert "Agent crashed" in result

    @pytest.mark.asyncio
    async def test_agent_error_sent_to_connector(
        self, config, audit_logger, mock_connector
    ):
        failing_agent = FakeAgent(fail=True)
        eng = Engine(
            connector=mock_connector,
            agent=failing_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )
        await eng.handle_message("user1", "hello", "chat1")
        assert len(mock_connector.sent_messages) == 1
        assert "Error:" in mock_connector.sent_messages[0]["text"]


class TestEngineMessageCtx:
    @pytest.mark.asyncio
    async def test_handle_message_ctx_delegates(self, engine):
        ctx = MessageContext(user_id="user1", chat_id="chat1", text="hello ctx")
        result = await engine.handle_message_ctx(ctx)
        assert "Echo: hello ctx" in result


class TestEngineWithMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_chain_runs(self, config, fake_agent, audit_logger):
        from tether.middleware.auth import AuthMiddleware

        chain = MiddlewareChain()
        chain.add(AuthMiddleware({"user1"}))

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            middleware_chain=chain,
        )

        ctx = MessageContext(user_id="user1", chat_id="chat1", text="hi")
        result = await chain.run(ctx, eng.handle_message_ctx)
        assert "Echo: hi" in result

    @pytest.mark.asyncio
    async def test_middleware_rejects_unauthorized(
        self, config, fake_agent, audit_logger
    ):
        from tether.middleware.auth import AuthMiddleware

        chain = MiddlewareChain()
        chain.add(AuthMiddleware({"user1"}))

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            middleware_chain=chain,
        )

        ctx = MessageContext(user_id="intruder", chat_id="chat1", text="hi")
        result = await chain.run(ctx, eng.handle_message_ctx)
        assert "Unauthorized" in result

    @pytest.mark.asyncio
    async def test_connector_handler_enforces_middleware(
        self, config, fake_agent, audit_logger, mock_connector
    ):
        from tether.middleware.auth import AuthMiddleware

        chain = MiddlewareChain()
        chain.add(AuthMiddleware({"user1"}))

        Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            middleware_chain=chain,
        )

        # Simulate an unauthorized user sending a message through the connector
        await mock_connector.simulate_message("intruder", "hi", "chat1")

        assert len(mock_connector.sent_messages) == 0
        assert fake_agent.last_can_use_tool is None

    @pytest.mark.asyncio
    async def test_connector_handler_authorized_user_through_middleware(
        self, config, fake_agent, audit_logger, mock_connector
    ):
        from tether.middleware.auth import AuthMiddleware

        chain = MiddlewareChain()
        chain.add(AuthMiddleware({"user1"}))

        Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            middleware_chain=chain,
        )

        await mock_connector.simulate_message("user1", "hello", "chat1")

        assert len(mock_connector.sent_messages) == 1
        assert "Echo: hello" in mock_connector.sent_messages[0]["text"]
        assert fake_agent.last_can_use_tool is not None


class TestEngineLifecycle:
    @pytest.mark.asyncio
    async def test_engine_started_event(self, config, fake_agent, audit_logger):
        from tether.core.events import ENGINE_STARTED

        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(ENGINE_STARTED, capture)

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )
        await eng.startup()
        assert len(events) == 1
        assert events[0].name == ENGINE_STARTED
        await eng.shutdown()

    @pytest.mark.asyncio
    async def test_engine_stopped_event(self, config, fake_agent, audit_logger):
        from tether.core.events import ENGINE_STOPPED

        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(ENGINE_STOPPED, capture)

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )
        await eng.startup()
        await eng.shutdown()
        assert len(events) == 1
        assert events[0].name == ENGINE_STOPPED


class TestEngineStartupShutdown:
    """Tests for engine startup/shutdown lifecycle (lines 84-101)."""

    @pytest.mark.asyncio
    async def test_startup_calls_store_setup(self, config, fake_agent, audit_logger):
        from unittest.mock import AsyncMock

        from tether.storage.memory import MemorySessionStore

        store = MemorySessionStore()
        store.setup = AsyncMock()
        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            store=store,
        )
        await eng.startup()
        store.setup.assert_awaited_once()
        await eng.shutdown()

    @pytest.mark.asyncio
    async def test_startup_calls_plugin_init_and_start(
        self, config, fake_agent, audit_logger
    ):
        from tether.plugins.base import PluginMeta, TetherPlugin
        from tether.plugins.registry import PluginRegistry

        class FakePlugin(TetherPlugin):
            meta = PluginMeta(name="test", version="1.0")
            init_called = False
            start_called = False

            async def initialize(self, context):
                FakePlugin.init_called = True

            async def start(self):
                FakePlugin.start_called = True

        registry = PluginRegistry()
        registry.register(FakePlugin())
        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            plugin_registry=registry,
        )
        await eng.startup()
        assert FakePlugin.init_called
        assert FakePlugin.start_called
        await eng.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_calls_plugin_stop(self, config, fake_agent, audit_logger):
        from tether.plugins.base import PluginMeta, TetherPlugin
        from tether.plugins.registry import PluginRegistry

        class FakePlugin(TetherPlugin):
            meta = PluginMeta(name="test2", version="1.0")
            stop_called = False

            async def initialize(self, context):
                pass

            async def stop(self):
                FakePlugin.stop_called = True

        registry = PluginRegistry()
        registry.register(FakePlugin())
        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            plugin_registry=registry,
        )
        await eng.startup()
        await eng.shutdown()
        assert FakePlugin.stop_called

    @pytest.mark.asyncio
    async def test_shutdown_calls_store_teardown(
        self, config, fake_agent, audit_logger
    ):
        from unittest.mock import AsyncMock

        from tether.storage.memory import MemorySessionStore

        store = MemorySessionStore()
        store.teardown = AsyncMock()
        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            store=store,
        )
        await eng.startup()
        await eng.shutdown()
        store.teardown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_startup_shutdown_full_lifecycle(
        self, config, fake_agent, audit_logger
    ):
        from unittest.mock import AsyncMock

        from tether.plugins.base import PluginMeta, TetherPlugin
        from tether.plugins.registry import PluginRegistry
        from tether.storage.memory import MemorySessionStore

        lifecycle_events = []

        class LP(TetherPlugin):
            meta = PluginMeta(name="lp", version="1.0")

            async def initialize(self, context):
                lifecycle_events.append("init")

            async def start(self):
                lifecycle_events.append("start")

            async def stop(self):
                lifecycle_events.append("stop")

        store = MemorySessionStore()
        store.setup = AsyncMock()
        store.teardown = AsyncMock()
        registry = PluginRegistry()
        registry.register(LP())

        eng = Engine(
            connector=None,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            store=store,
            plugin_registry=registry,
        )
        await eng.startup()
        store.setup.assert_awaited_once()
        assert lifecycle_events == ["init", "start"]

        await eng.shutdown()
        store.teardown.assert_awaited_once()
        assert lifecycle_events == ["init", "start", "stop"]

    @pytest.mark.asyncio
    async def test_connector_sends_response(
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
        assert len(mock_connector.sent_messages) == 1
        assert "Echo: hello" in mock_connector.sent_messages[0]["text"]


class TestAuditLogger:
    """Tests for audit logger edge cases."""

    def test_audit_write_failure_logged_not_raised(self, tmp_path):
        from tether.core.safety.audit import AuditLogger

        logger = AuditLogger(tmp_path / "nonexistent_dir_xyz" / "deep" / "audit.jsonl")
        # _write creates parent dir, but let's make it fail by using a file as parent
        file_as_dir = tmp_path / "blocker"
        file_as_dir.write_text("I am a file")
        logger._path = file_as_dir / "subpath" / "audit.jsonl"
        # Should not raise — write failure is logged, not raised
        logger._write({"event": "test"})

    def test_sanitize_input_truncation(self):
        from tether.core.safety.audit import _sanitize_input

        long_val = "x" * 600
        result = _sanitize_input({"cmd": long_val})
        assert len(result["cmd"]) < 600
        assert "[truncated]" in result["cmd"]

    def test_sanitize_input_passthrough_non_strings(self):
        from tether.core.safety.audit import _sanitize_input

        result = _sanitize_input({"count": 42, "flag": True, "data": None})
        assert result["count"] == 42
        assert result["flag"] is True
        assert result["data"] is None


class TestEngineSessionManagement:
    @pytest.mark.asyncio
    async def test_session_reuse_across_messages(self, engine):
        await engine.handle_message("user1", "hello", "chat1")
        await engine.handle_message("user1", "world", "chat1")
        session = engine.session_manager.get("user1", "chat1")
        assert session.message_count == 2
        assert session.total_cost == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_session_isolation_between_users(self, engine):
        await engine.handle_message("user1", "hello", "chat1")
        await engine.handle_message("user2", "world", "chat2")
        s1 = engine.session_manager.get("user1", "chat1")
        s2 = engine.session_manager.get("user2", "chat2")
        assert s1.session_id != s2.session_id
        assert s1.message_count == 1
        assert s2.message_count == 1

    @pytest.mark.asyncio
    async def test_connector_receives_response(
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
        assert len(mock_connector.sent_messages) == 1
        assert "Echo: hello" in mock_connector.sent_messages[0]["text"]

    @pytest.mark.asyncio
    async def test_no_connector_no_crash(self, engine):
        result = await engine.handle_message("user1", "hello", "chat1")
        assert "Echo: hello" in result

    @pytest.mark.asyncio
    async def test_agent_error_does_not_crash_engine(self, config, audit_logger):
        failing_agent = FakeAgent(fail=True)
        eng = Engine(
            connector=None,
            agent=failing_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )
        result = await eng.handle_message("user1", "hello", "chat1")
        assert "Error:" in result
        # Engine still works after error
        result2 = await eng.handle_message("user1", "hello2", "chat1")
        assert "Error:" in result2

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self, engine, fake_agent, tmp_dir):
        await engine.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        r1 = await hook("Read", {"file_path": str(tmp_dir / "a.py")}, None)
        r2 = await hook("Bash", {"command": "git status"}, None)
        assert r1.behavior == "allow"
        assert r2.behavior == "allow"


class TestEngineAgentCrashCancelsApprovals:
    @pytest.mark.asyncio
    async def test_agent_crash_cancels_pending_approvals(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        coordinator = ApprovalCoordinator(mock_connector, config)
        failing_agent = FakeAgent(fail=True)
        eng = Engine(
            connector=mock_connector,
            agent=failing_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=coordinator,
        )

        # Manually add a pending approval for the chat
        from tether.core.safety.approvals import PendingApproval

        pending = PendingApproval(
            approval_id="test-approval-id",
            chat_id="chat1",
            tool_name="Write",
            tool_input={"file_path": "/a.py"},
        )
        coordinator.pending["test-approval-id"] = pending

        await eng.handle_message("user1", "hello", "chat1")

        # Pending approval should be cancelled (decision set to False, event set)
        assert pending.decision is False
        assert pending.event.is_set()


class TestEngineMessageLogging:
    @pytest.mark.asyncio
    async def test_messages_logged_when_sqlite_store(
        self, config, fake_agent, audit_logger, tmp_path
    ):
        from tether.storage.sqlite import SqliteSessionStore

        store = SqliteSessionStore(tmp_path / "msg.db")
        await store.setup()
        try:
            eng = Engine(
                connector=None,
                agent=fake_agent,
                config=config,
                session_manager=SessionManager(),
                audit=audit_logger,
                store=store,
            )
            await eng.handle_message("user1", "hello", "chat1")
            msgs = await store.get_messages("user1", "chat1")
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[0]["content"] == "hello"
            assert msgs[1]["role"] == "assistant"
            assert "Echo: hello" in msgs[1]["content"]
            assert msgs[1]["cost"] == pytest.approx(0.01)
            assert msgs[1]["duration_ms"] is not None
            assert msgs[1]["session_id"] == "test-session-123"
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_message_log_failure_does_not_break_handling(
        self, config, fake_agent, audit_logger, tmp_path
    ):
        from unittest.mock import AsyncMock

        from tether.exceptions import StorageError
        from tether.storage.sqlite import SqliteSessionStore

        store = SqliteSessionStore(tmp_path / "msg.db")
        await store.setup()
        store.save_message = AsyncMock(side_effect=StorageError("disk full"))
        try:
            eng = Engine(
                connector=None,
                agent=fake_agent,
                config=config,
                session_manager=SessionManager(),
                audit=audit_logger,
                store=store,
            )
            result = await eng.handle_message("user1", "hello", "chat1")
            assert "Echo: hello" in result
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_agent_error_only_logs_user_message(
        self, config, audit_logger, tmp_path
    ):
        from tether.storage.sqlite import SqliteSessionStore

        store = SqliteSessionStore(tmp_path / "msg.db")
        await store.setup()
        try:
            eng = Engine(
                connector=None,
                agent=FakeAgent(fail=True),
                config=config,
                session_manager=SessionManager(),
                audit=audit_logger,
                store=store,
            )
            await eng.handle_message("user1", "hello", "chat1")
            msgs = await store.get_messages("user1", "chat1")
            assert len(msgs) == 1
            assert msgs[0]["role"] == "user"
        finally:
            await store.teardown()


class TestBuildImplementationPrompt:
    def test_long_content_includes_plan(self, engine):
        plan = "A detailed plan with many steps and specifics to implement carefully"
        result = engine._build_implementation_prompt(plan)
        assert result.startswith("Implement the following plan:")
        assert "A detailed plan" in result

    def test_short_content_uses_generic(self, engine):
        result = engine._build_implementation_prompt("Short")
        assert result == "Implement the plan."

    def test_empty_content_uses_generic(self, engine):
        result = engine._build_implementation_prompt("   ")
        assert result == "Implement the plan."


class TestEngineInteractionRouting:
    @pytest.mark.asyncio
    async def test_ask_user_question_intercepted(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        tool_input = {
            "questions": [
                {
                    "question": "Pick?",
                    "header": "Choice",
                    "options": [{"label": "A", "description": "a"}],
                    "multiSelect": False,
                }
            ]
        }

        async def answer():
            await asyncio.sleep(0.05)
            req = mock_connector.question_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "A")

        task = asyncio.create_task(answer())
        result = await hook("AskUserQuestion", tool_input, None)
        await task

        assert result.behavior == "allow"
        assert result.updated_input["answers"]["Pick?"] == "A"

    @pytest.mark.asyncio
    async def test_exit_plan_mode_intercepted(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        async def click_proceed():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "edit")

        task = asyncio.create_task(click_proceed())
        result = await hook("ExitPlanMode", {}, None)
        await task

        assert result.behavior == "allow"
        auto = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Write" in auto
        assert "Edit" in auto

    @pytest.mark.asyncio
    async def test_regular_tool_still_hits_gatekeeper(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector, tmp_dir
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        # Read inside approved dir → still goes through gatekeeper → allowed
        result = await hook("Read", {"file_path": str(tmp_dir / "foo.py")}, None)
        assert result.behavior == "allow"

        # Sandbox violation → still denied
        result = await hook("Read", {"file_path": "/etc/passwd"}, None)
        assert result.behavior == "deny"

    @pytest.mark.asyncio
    async def test_text_routed_to_pending_interaction(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        tool_input = {
            "questions": [
                {
                    "question": "Q?",
                    "header": "H",
                    "options": [{"label": "A", "description": "a"}],
                    "multiSelect": False,
                }
            ]
        }

        async def simulate_text_answer():
            await asyncio.sleep(0.05)
            # Simulate user sending text while question is pending
            result = await eng.handle_message("user1", "custom answer", "chat1")
            assert result == ""

        task = asyncio.create_task(simulate_text_answer())
        result = await coordinator.handle_question("chat1", tool_input)
        await task

        assert result.behavior == "allow"
        assert result.updated_input["answers"]["Q?"] == "custom answer"

    @pytest.mark.asyncio
    async def test_agent_crash_cancels_interactions(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        from unittest.mock import MagicMock

        coordinator = InteractionCoordinator(mock_connector, config)
        coordinator.cancel_pending = MagicMock(return_value=[])
        failing_agent = FakeAgent(fail=True)
        eng = Engine(
            connector=mock_connector,
            agent=failing_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "hello", "chat1")

        coordinator.cancel_pending.assert_called_once_with("chat1")

    @pytest.mark.asyncio
    async def test_exit_plan_mode_clean_proceed_resets_session(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        session.claude_session_id = "existing-session-123"
        hook = fake_agent.last_can_use_tool

        async def click_clean():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "clean_edit")

        task = asyncio.create_task(click_clean())
        result = await hook("ExitPlanMode", {}, None)
        await task

        assert result.behavior == "allow"
        assert session.claude_session_id is None
        auto = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Write" in auto
        assert "Edit" in auto


class TestCleanProceedAutoImplementation:
    @pytest.mark.asyncio
    async def test_clean_proceed_triggers_fresh_execution(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        prompts_seen: list[str] = []
        session_ids_at_start: list[str | None] = []

        class PlanAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                prompts_seen.append(prompt)
                session_ids_at_start.append(session.claude_session_id)
                if not prompt.startswith("Implement"):

                    async def click_clean():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    task = asyncio.create_task(click_clean())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await task
                return AgentResponse(
                    content=f"Done: {prompt}",
                    session_id="sid-123",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=PlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "Make a plan", "chat1")

        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "auto"
        # Agent was called twice: first with original prompt, then with plan content
        assert len(prompts_seen) == 2
        assert prompts_seen[0] == "Make a plan"
        assert prompts_seen[1].startswith("Implement")
        # Second call started with a clean session (no prior claude_session_id)
        assert session_ids_at_start[1] is None
        cleared_msgs = [
            m
            for m in mock_connector.sent_messages
            if "Context cleared" in m.get("text", "")
        ]
        assert len(cleared_msgs) >= 1

    @pytest.mark.asyncio
    async def test_clean_proceed_state_not_set_for_normal_proceed(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        async def click_proceed():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "edit")

        task = asyncio.create_task(click_proceed())
        result = await hook("ExitPlanMode", {}, None)
        await task

        assert result.behavior == "allow"

    @pytest.mark.asyncio
    async def test_clean_proceed_deactivates_streaming(
        self, config, policy_engine, audit_logger
    ):
        """After clean_edit, further text chunks are suppressed."""
        from tests.conftest import MockConnector

        connector = MockConnector(support_streaming=True)
        coordinator = InteractionCoordinator(connector, config)
        chunks_after_exit: list[str] = []

        class StreamAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                on_chunk = kwargs.get("on_text_chunk")
                if not prompt.startswith("Implement"):
                    if on_chunk:
                        await on_chunk("before exit ")

                    async def click_clean():
                        await asyncio.sleep(0.05)
                        req = connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    task = asyncio.create_task(click_clean())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await task
                    # These chunks should be suppressed after deactivation
                    if on_chunk:
                        await on_chunk("after exit ")
                        chunks_after_exit.append("after exit ")
                return AgentResponse(
                    content=f"Done: {prompt}",
                    session_id="sid-123",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=connector,
            agent=StreamAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "Make a plan", "chat1")

        # The chunk was called but responder should have ignored it
        assert len(chunks_after_exit) == 1
        # Only the "before exit " chunk should have created a streaming message
        streaming_msgs = [
            m for m in connector.sent_messages if "after exit" in m.get("text", "")
        ]
        assert streaming_msgs == []

    @pytest.mark.asyncio
    async def test_clean_proceed_cancels_agent(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """After clean_edit, the agent cancel is scheduled."""
        coordinator = InteractionCoordinator(mock_connector, config)
        cancel_calls: list[str] = []

        class CancelAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                if not prompt.startswith("Implement"):

                    async def click_clean():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    task = asyncio.create_task(click_clean())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await task
                return AgentResponse(
                    content=f"Done: {prompt}",
                    session_id="sid-123",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                cancel_calls.append(session_id)

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=CancelAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "Make a plan", "chat1")
        # Let the scheduled cancel task run
        await asyncio.sleep(0.2)

        assert len(cancel_calls) >= 1


class TestEngineApprovalTextRouting:
    @pytest.mark.asyncio
    async def test_text_message_rejects_pending_approval(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = ApprovalCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=coordinator,
        )

        # Manually create a pending approval to simulate in-flight approval
        from tether.core.safety.approvals import PendingApproval

        pending = PendingApproval(
            approval_id="test-id",
            chat_id="chat1",
            tool_name="Bash",
            tool_input={"command": "pip install foo"},
            message_id="42",
        )
        coordinator.pending["test-id"] = pending

        result = await eng.handle_message("user1", "use uv add instead", "chat1")
        assert result == ""
        assert pending.decision is False
        assert pending.rejection_reason == "use uv add instead"
        assert {"chat_id": "chat1", "message_id": "42"} in (
            mock_connector.deleted_messages
        )

    @pytest.mark.asyncio
    async def test_approval_routing_before_interaction_routing(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        approval_coord = ApprovalCoordinator(mock_connector, config)
        interaction_coord = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=approval_coord,
            interaction_coordinator=interaction_coord,
        )

        # Both have pending items for same chat
        from tether.core.safety.approvals import PendingApproval

        pending = PendingApproval(
            approval_id="appr-1",
            chat_id="chat1",
            tool_name="Bash",
            tool_input={"command": "npm install"},
        )
        approval_coord.pending["appr-1"] = pending

        from tether.core.interactions import PendingInteraction

        interaction = PendingInteraction(
            interaction_id="inter-1",
            chat_id="chat1",
            kind="question",
        )
        interaction_coord.pending["inter-1"] = interaction
        interaction_coord._chat_index["chat1"] = "inter-1"

        # Approval routing should win
        result = await eng.handle_message("user1", "reject this", "chat1")
        assert result == ""
        assert pending.decision is False
        # Interaction should NOT have been resolved
        assert interaction.answer is None

    @pytest.mark.asyncio
    async def test_no_pending_approval_routes_normally(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        coordinator = ApprovalCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=coordinator,
        )

        result = await eng.handle_message("user1", "hello", "chat1")
        assert result == "Echo: hello"


class TestPlanContentInEngine:
    @pytest.mark.asyncio
    async def test_plan_content_from_responder_passed_to_coordinator(
        self, config, policy_engine, audit_logger
    ):
        from tests.conftest import MockConnector

        streaming_connector = MockConnector(support_streaming=True)
        coordinator = InteractionCoordinator(streaming_connector, config)
        config.streaming_enabled = True

        plan_file_text = (
            "## Implementation Plan\n\n"
            "Step 1: Set up the database schema and migrations\n"
            "Step 2: Implement the API endpoints for CRUD operations\n"
        )

        class StreamingPlanAgent(BaseAgent):
            async def execute(
                self,
                prompt,
                session,
                *,
                can_use_tool=None,
                on_text_chunk=None,
                **kwargs,
            ):
                if on_text_chunk:
                    await on_text_chunk("I'll start by exploring the codebase...\n")

                if prompt != "Implement the plan.":
                    # Write the plan to a .plan file
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": "/tmp/project/.claude/plans/my.plan",
                            "content": plan_file_text,
                        },
                        None,
                    )

                    async def click_proceed():
                        await asyncio.sleep(0.05)
                        req = streaming_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    task = asyncio.create_task(click_proceed())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await task

                return AgentResponse(
                    content="Plan reviewed.",
                    session_id="sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=streaming_connector,
            agent=StreamingPlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan something", "chat1")

        assert len(streaming_connector.plan_review_requests) == 1
        desc = streaming_connector.plan_review_requests[0]["description"]
        # Plan review shows the .plan file content, not the streaming narration
        assert "Implementation Plan" in desc
        assert "database schema" in desc
        assert "I'll start by exploring" not in desc

    @pytest.mark.asyncio
    async def test_plan_file_content_preferred_over_streaming_buffer(
        self, config, policy_engine, audit_logger
    ):
        """When agent writes a .plan file AND streams text, plan review shows file content."""
        from tests.conftest import MockConnector

        streaming_connector = MockConnector(support_streaming=True)
        coordinator = InteractionCoordinator(streaming_connector, config)
        config.streaming_enabled = True

        class PlanFileAgent(BaseAgent):
            async def execute(
                self,
                prompt,
                session,
                *,
                can_use_tool=None,
                on_text_chunk=None,
                **kwargs,
            ):
                if on_text_chunk:
                    await on_text_chunk("Narration that should NOT appear in review\n")

                if not prompt.startswith("Implement"):
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": "/tmp/work/.claude/plans/fix.plan",
                            "content": "# Real Plan\n\n1. Fix the bug\n2. Add tests",
                        },
                        None,
                    )

                    async def click():
                        await asyncio.sleep(0.05)
                        req = streaming_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t

                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=streaming_connector,
            agent=PlanFileAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Fix the bug", "chat1")

        desc = streaming_connector.plan_review_requests[0]["description"]
        assert "Real Plan" in desc
        assert "Fix the bug" in desc
        assert "Narration" not in desc

    @pytest.mark.asyncio
    async def test_streaming_buffer_used_when_no_plan_file(
        self, config, policy_engine, audit_logger, monkeypatch
    ):
        """When no .plan file is written, streaming buffer is used as fallback."""
        from tests.conftest import MockConnector

        streaming_connector = MockConnector(support_streaming=True)
        coordinator = InteractionCoordinator(streaming_connector, config)
        config.streaming_enabled = True
        # Prevent discovery of real plan files on the test machine
        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )

        class NoPlanFileAgent(BaseAgent):
            async def execute(
                self,
                prompt,
                session,
                *,
                can_use_tool=None,
                on_text_chunk=None,
                **kwargs,
            ):
                if on_text_chunk:
                    await on_text_chunk("Here is the streamed plan content\n")

                if not prompt.startswith("Implement"):

                    async def click():
                        await asyncio.sleep(0.05)
                        req = streaming_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t

                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=streaming_connector,
            agent=NoPlanFileAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        desc = streaming_connector.plan_review_requests[0]["description"]
        assert "streamed plan content" in desc

    @pytest.mark.asyncio
    async def test_plan_file_content_used_in_implementation_prompt(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """After clean_proceed, agent receives plan file content, not narration."""
        coordinator = InteractionCoordinator(mock_connector, config)
        prompts_seen: list[str] = []
        plan_text = "# The Real Plan\n\n1. Refactor module\n2. Add validation"

        class WritePlanAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                prompts_seen.append(prompt)
                if not prompt.startswith("Implement"):
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": "/tmp/proj/.claude/plans/refactor.plan",
                            "content": plan_text,
                        },
                        None,
                    )

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t

                return AgentResponse(
                    content="Narration text that should not be the prompt",
                    session_id="sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=WritePlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Refactor it", "chat1")

        assert len(prompts_seen) == 2
        impl_prompt = prompts_seen[1]
        assert impl_prompt.startswith("Implement the following plan:")
        assert "The Real Plan" in impl_prompt
        assert "Refactor module" in impl_prompt
        assert "Narration text" not in impl_prompt


class TestFallbackPlanReview:
    @pytest.mark.asyncio
    async def test_fallback_shown_when_agent_skips_exit_plan_mode(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """When agent responds in plan mode without calling ExitPlanMode,
        fallback plan review buttons should appear."""
        coordinator = InteractionCoordinator(mock_connector, config)

        class PlanSkipAgent(BaseAgent):
            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                # Agent writes a plan file but never calls ExitPlanMode
                if can_use_tool:
                    plan_path = f"{session.working_directory}/.claude/plans/plan.md"
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": plan_path,
                            "content": "Here is my plan:\n1. Do thing\n2. Do other thing",
                        },
                        None,
                    )
                return AgentResponse(
                    content="Here is my plan:\n1. Do thing\n2. Do other thing",
                    session_id="sid-123",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=PlanSkipAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        # Set session to plan mode with prior messages (resumed session)
        session = await eng.session_manager.get_or_create(
            "user1", "chat1", str(config.approved_directories[0])
        )
        session.mode = "plan"
        session.message_count = 2  # simulate prior messages

        # Simulate user clicking "edit" on fallback review
        async def click_proceed():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "edit")

        task = asyncio.create_task(click_proceed())
        await eng.handle_message("user1", "What's the plan?", "chat1")
        await task

        # Fallback plan review was shown with actual plan content
        assert len(mock_connector.plan_review_requests) >= 1
        desc = mock_connector.plan_review_requests[0]["description"]
        assert "Here is my plan" in desc

    @pytest.mark.asyncio
    async def test_fallback_not_shown_when_exit_plan_mode_called(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """When agent calls ExitPlanMode, no fallback review should appear."""
        coordinator = InteractionCoordinator(mock_connector, config)

        class ProperPlanAgent(BaseAgent):
            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                if prompt != "Implement the plan.":
                    # Agent properly calls ExitPlanMode
                    async def click_proceed():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click_proceed())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(
                    content="Done",
                    session_id="sid-123",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=ProperPlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        session = await eng.session_manager.get_or_create(
            "user1", "chat1", str(config.approved_directories[0])
        )
        session.mode = "plan"
        session.message_count = 2  # simulate prior messages

        await eng.handle_message("user1", "Plan it", "chat1")

        # Only one plan review (from ExitPlanMode), not a fallback one
        assert len(mock_connector.plan_review_requests) == 1

    @pytest.mark.asyncio
    async def test_fallback_adjust_sends_feedback(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        coordinator = InteractionCoordinator(mock_connector, config)

        class PlanSkipAgent(BaseAgent):
            def __init__(self):
                self.prompts = []

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.prompts.append(prompt)
                # Agent writes a plan file but never calls ExitPlanMode
                if can_use_tool:
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": ".claude/plans/plan.md",
                            "content": "Plan summary",
                        },
                        None,
                    )
                # On the second call (feedback), switch out of plan mode
                # so the fallback doesn't trigger a third round
                if len(self.prompts) > 1:
                    session.mode = "auto"
                return AgentResponse(
                    content="Plan summary", session_id="sid", cost=0.01
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        agent = PlanSkipAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        session = await eng.session_manager.get_or_create(
            "user1", "chat1", str(config.approved_directories[0])
        )
        session.mode = "plan"
        session.message_count = 2  # simulate prior messages

        async def click_adjust():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "adjust")
            await asyncio.sleep(0.05)
            await coordinator.resolve_text("chat1", "Add error handling")

        task = asyncio.create_task(click_adjust())
        await eng.handle_message("user1", "Plan it", "chat1")
        await task

        # Agent should have been called with the feedback text
        assert "Add error handling" in agent.prompts


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
        from tether.core.events import EventBus
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
        assert "Echo: verify login" in mock_connector.sent_messages[0]["text"]

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

        assert "Directories:" in result
        assert "tether" in result
        assert "api" in result
        assert "✅" in result

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


class TestPlanFileDiskRead:
    @pytest.mark.asyncio
    async def test_plan_file_read_from_disk_on_exit_plan_mode(
        self, config, policy_engine, audit_logger, mock_connector, tmp_path
    ):
        """When agent uses Edit on a plan file, ExitPlanMode reads final content from disk."""
        coordinator = InteractionCoordinator(mock_connector, config)
        plan_file = tmp_path / ".claude" / "plans" / "my.plan"
        plan_file.parent.mkdir(parents=True)
        plan_file.write_text("# Original Plan\n\n1. First step")

        class EditPlanAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                if not prompt.startswith("Implement"):
                    # Write initial content
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": str(plan_file),
                            "content": "# Original Plan\n\n1. First step",
                        },
                        None,
                    )
                    # Edit the plan file (updates on disk)
                    plan_file.write_text(
                        "# Updated Plan\n\n1. First step\n2. Added step"
                    )
                    await can_use_tool(
                        "Edit",
                        {
                            "file_path": str(plan_file),
                            "old_string": "1. First step",
                            "new_string": "1. First step\n2. Added step",
                        },
                        None,
                    )

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t

                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=EditPlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        desc = mock_connector.plan_review_requests[0]["description"]
        # Should show the final on-disk content (with "Added step"), not the Write content
        assert "Updated Plan" in desc
        assert "Added step" in desc


class TestAutoApproveWritesAfterProceed:
    @pytest.mark.asyncio
    async def test_auto_approve_writes_after_plan_proceed(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        """After ExitPlanMode proceed, Write/Edit are auto-approved but Bash is not."""
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        async def click_proceed():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "edit")

        task = asyncio.create_task(click_proceed())
        await hook("ExitPlanMode", {}, None)
        await task

        auto = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Write" in auto
        assert "Edit" in auto
        # Bash should NOT be auto-approved
        assert "Bash" not in auto
        # Blanket auto-approve should NOT be set
        assert "chat1" not in eng._gatekeeper._auto_approved_chats

    @pytest.mark.asyncio
    async def test_auto_approve_writes_after_fallback_proceed(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """After fallback plan review proceed, Write/Edit are auto-approved."""
        coordinator = InteractionCoordinator(mock_connector, config)

        class PlanSkipAgent(BaseAgent):
            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                if prompt.startswith("Implement"):
                    return AgentResponse(
                        content="Implemented", session_id="sid", cost=0.01
                    )
                # Agent writes a plan file but never calls ExitPlanMode
                if can_use_tool:
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": ".claude/plans/plan.md",
                            "content": "Here is the plan:\n1. Do things",
                        },
                        None,
                    )
                return AgentResponse(
                    content="Here is the plan:\n1. Do things",
                    session_id="sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=PlanSkipAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        session = await eng.session_manager.get_or_create(
            "user1", "chat1", str(config.approved_directories[0])
        )
        session.mode = "plan"
        session.message_count = 2

        async def click_proceed():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "edit")

        task = asyncio.create_task(click_proceed())
        await eng.handle_message("user1", "What's the plan?", "chat1")
        await task

        auto = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Write" in auto
        assert "Edit" in auto
        assert "Bash" not in auto


class TestDefaultButtonNoAutoApprove:
    @pytest.mark.asyncio
    async def test_default_button_does_not_auto_approve_writes(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        """After ExitPlanMode with 'default' button, Write/Edit NOT auto-approved."""
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        async def click_default():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "default")

        task = asyncio.create_task(click_default())
        result = await hook("ExitPlanMode", {}, None)
        await task

        assert result.behavior == "allow"
        auto = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Write" not in auto
        assert "Edit" not in auto


class TestEditModeSecurityRegression:
    @pytest.mark.asyncio
    async def test_edit_mode_does_not_blanket_auto_approve(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        """After /edit, Bash commands should NOT be auto-approved."""
        agent = FakeAgent()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_command("user1", "edit", "", "chat1")

        # Bash must not be in the auto-approved set
        auto_tools = eng._gatekeeper._auto_approved_tools.get("chat1", set())
        assert "Bash" not in auto_tools
        assert "chat1" not in eng._gatekeeper._auto_approved_chats


class TestStreamingResponderReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state_and_new_chunk_creates_new_message(
        self, config, policy_engine, audit_logger
    ):
        """After reset(), the next chunk should create a new message (not edit the old one)."""
        from tests.conftest import MockConnector
        from tether.core.engine import _StreamingResponder

        connector = MockConnector(support_streaming=True)
        responder = _StreamingResponder(connector, "chat1", throttle_seconds=0)

        # Send initial chunk — creates first message
        await responder.on_chunk("plan output")
        assert responder._message_id == "1"
        first_id = responder._message_id

        # Reset
        responder.reset()
        assert responder._message_id is None
        assert responder._buffer == ""
        assert responder._has_activity is False
        assert responder._tool_counts == {}

        # Send new chunk — should create a second message, not edit the first
        await responder.on_chunk("implementation output")
        assert responder._message_id == "2"
        assert responder._message_id != first_id


class TestRelativePlanPathDetection:
    @pytest.mark.asyncio
    async def test_relative_plan_path_detected(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        """Plan file with relative path .claude/plans/... is detected."""
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        # Simulate Write with a relative path (no leading /)
        await hook(
            "Write",
            {
                "file_path": ".claude/plans/my-plan.md",
                "content": "# My Plan\n\n1. Step one",
            },
            None,
        )

        assert eng._tool_state.plan_file_path == ".claude/plans/my-plan.md"
        assert eng._tool_state.plan_file_content == "# My Plan\n\n1. Step one"

    @pytest.mark.asyncio
    async def test_absolute_plan_path_still_detected(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        """Plan file with absolute path /home/user/.claude/plans/... is still detected."""
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        await hook(
            "Write",
            {
                "file_path": "/home/user/.claude/plans/fix.md",
                "content": "# Fix Plan",
            },
            None,
        )

        assert eng._tool_state.plan_file_path == "/home/user/.claude/plans/fix.md"

    @pytest.mark.asyncio
    async def test_dot_plan_extension_detected(
        self, config, fake_agent, policy_engine, audit_logger, mock_connector
    ):
        """Files with .plan extension are detected regardless of path."""
        coordinator = InteractionCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=fake_agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )
        await eng.handle_message("user1", "hello", "chat1")
        hook = fake_agent.last_can_use_tool

        await hook(
            "Edit",
            {
                "file_path": "/tmp/project/my.plan",
                "old_string": "a",
                "new_string": "b",
            },
            None,
        )

        assert eng._tool_state.plan_file_path == "/tmp/project/my.plan"
        # Edit doesn't cache content — only Write does
        assert eng._tool_state.plan_file_content is None


class TestPlanContentSourceTracking:
    @pytest.mark.asyncio
    async def test_disk_file_source_used_when_file_exists(
        self, config, policy_engine, audit_logger, mock_connector, tmp_path
    ):
        """When plan file exists on disk, ExitPlanMode reads it (source=disk_file)."""
        coordinator = InteractionCoordinator(mock_connector, config)
        plan_file = tmp_path / ".claude" / "plans" / "test.md"
        plan_file.parent.mkdir(parents=True)
        plan_file.write_text("# Disk Plan\n\nStep 1: Do things")

        class DiskPlanAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                if not prompt.startswith("Implement"):
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": str(plan_file),
                            "content": "# Stale cached content",
                        },
                        None,
                    )
                    # Overwrite with updated content (simulating Edit)
                    plan_file.write_text("# Disk Plan\n\nStep 1: Do things")

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t

                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=DiskPlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        desc = mock_connector.plan_review_requests[0]["description"]
        # Disk file content is preferred over cached Write content
        assert "Disk Plan" in desc
        assert "Stale cached" not in desc

    @pytest.mark.asyncio
    async def test_cached_write_used_when_disk_read_fails(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """When plan file doesn't exist on disk, cached Write content is used."""
        coordinator = InteractionCoordinator(mock_connector, config)

        class CachedPlanAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                if not prompt.startswith("Implement"):
                    # Write to a path that won't exist on disk
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": "/nonexistent/path/.claude/plans/test.md",
                            "content": "# Cached Plan\n\nThis came from cache",
                        },
                        None,
                    )

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t

                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=CachedPlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        desc = mock_connector.plan_review_requests[0]["description"]
        assert "Cached Plan" in desc


class TestPlanFileDiscoveryFromDisk:
    """Bug 1 regression: when SDK bypasses can_use_tool for plan file writes,
    _discover_plan_file finds the plan from ~/.claude/plans/ on disk."""

    @pytest.mark.asyncio
    async def test_discover_plan_file_finds_recent_md(self, tmp_path, monkeypatch):
        plans_dir = tmp_path / ".claude" / "plans"
        plans_dir.mkdir(parents=True)
        plan_file = plans_dir / "test-plan.md"
        plan_file.write_text("# Discovered Plan")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = Engine._discover_plan_file()
        assert result == str(plan_file)

    @pytest.mark.asyncio
    async def test_discover_plan_file_ignores_old_files(self, tmp_path, monkeypatch):
        import os

        plans_dir = tmp_path / ".claude" / "plans"
        plans_dir.mkdir(parents=True)
        plan_file = plans_dir / "old-plan.md"
        plan_file.write_text("# Old Plan")
        # Set mtime to 20 minutes ago (beyond 600s threshold)
        old_time = time.time() - 1200
        os.utime(plan_file, (old_time, old_time))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = Engine._discover_plan_file()
        assert result is None

    @pytest.mark.asyncio
    async def test_discover_plan_file_returns_none_when_no_dir(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = Engine._discover_plan_file()
        assert result is None

    @pytest.mark.asyncio
    async def test_exit_plan_mode_uses_discovered_file(
        self, config, policy_engine, audit_logger, mock_connector, tmp_path, monkeypatch
    ):
        """When can_use_tool is never called for the plan Write (SDK bypass),
        ExitPlanMode discovers the plan file from disk."""
        plans_dir = tmp_path / ".claude" / "plans"
        plans_dir.mkdir(parents=True)
        plan_file = plans_dir / "discovered-plan.md"
        plan_file.write_text("# Discovered Plan\n\n1. Step one\n2. Step two")
        monkeypatch.setattr(
            Engine,
            "_discover_plan_file",
            staticmethod(lambda wd=None: str(plan_file)),
        )

        coordinator = InteractionCoordinator(mock_connector, config)

        class BypassAgent(BaseAgent):
            """Agent that calls ExitPlanMode without prior Write (simulates SDK bypass)."""

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                if not prompt.startswith("Implement"):

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=BypassAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        desc = mock_connector.plan_review_requests[0]["description"]
        assert "Discovered Plan" in desc
        assert "Step one" in desc

    @pytest.mark.asyncio
    async def test_resolve_plan_content_uses_discovered_file(
        self, config, policy_engine, audit_logger, mock_connector, tmp_path, monkeypatch
    ):
        """_resolve_plan_content discovers plan file when state.plan_file_path is None."""
        plans_dir = tmp_path / ".claude" / "plans"
        plans_dir.mkdir(parents=True)
        plan_file = plans_dir / "resolve-plan.md"
        plan_file.write_text(
            "# Resolved Plan\n\nStep 1: Refactor the module structure\n"
            "Step 2: Add comprehensive validation logic\n"
            "Step 3: Write integration tests for the new flow"
        )
        monkeypatch.setattr(
            Engine,
            "_discover_plan_file",
            staticmethod(lambda wd=None: str(plan_file)),
        )

        coordinator = InteractionCoordinator(mock_connector, config)
        prompts_seen: list[str] = []

        class BypassCleanAgent(BaseAgent):
            """Agent that calls ExitPlanMode with clean_proceed (no prior Write)."""

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                prompts_seen.append(prompt)
                if not prompt.startswith("Implement"):

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(
                    content="Narration only", session_id="sid", cost=0.01
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=BypassCleanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        # Implementation prompt should use discovered plan content, not narration
        assert len(prompts_seen) == 2
        impl_prompt = prompts_seen[1]
        assert "Resolved Plan" in impl_prompt
        assert "Narration only" not in impl_prompt


class TestLocalPlanFileDiscovery:
    """Plan discovery should also check project-local .claude/plans/ directory."""

    @pytest.mark.asyncio
    async def test_discover_plan_file_finds_local_plan(self, tmp_path, monkeypatch):
        """Plan file in project-local .claude/plans/ is discovered."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "fake_home")
        local_plans = tmp_path / "project" / ".claude" / "plans"
        local_plans.mkdir(parents=True)
        plan_file = local_plans / "local-plan.md"
        plan_file.write_text("# Local Plan")

        result = Engine._discover_plan_file(str(tmp_path / "project"))
        assert result == str(plan_file)

    @pytest.mark.asyncio
    async def test_discover_prefers_newest_across_both_dirs(
        self, tmp_path, monkeypatch
    ):
        """When both home and local plans exist, the newest one wins."""
        import os

        home_plans = tmp_path / "home" / ".claude" / "plans"
        home_plans.mkdir(parents=True)
        home_plan = home_plans / "home-plan.md"
        home_plan.write_text("# Home Plan")
        old_time = time.time() - 300
        os.utime(home_plan, (old_time, old_time))

        local_plans = tmp_path / "project" / ".claude" / "plans"
        local_plans.mkdir(parents=True)
        local_plan = local_plans / "local-plan.md"
        local_plan.write_text("# Local Plan (newer)")

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

        result = Engine._discover_plan_file(str(tmp_path / "project"))
        assert result == str(local_plan)

    @pytest.mark.asyncio
    async def test_discover_prefers_newest_home_over_old_local(
        self, tmp_path, monkeypatch
    ):
        """When home plan is newer than local plan, home plan wins."""
        import os

        local_plans = tmp_path / "project" / ".claude" / "plans"
        local_plans.mkdir(parents=True)
        local_plan = local_plans / "local-plan.md"
        local_plan.write_text("# Local Plan (older)")
        old_time = time.time() - 300
        os.utime(local_plan, (old_time, old_time))

        home_plans = tmp_path / "home" / ".claude" / "plans"
        home_plans.mkdir(parents=True)
        home_plan = home_plans / "home-plan.md"
        home_plan.write_text("# Home Plan (newer)")

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

        result = Engine._discover_plan_file(str(tmp_path / "project"))
        assert result == str(home_plan)

    @pytest.mark.asyncio
    async def test_discover_without_working_directory_only_checks_home(
        self, tmp_path, monkeypatch
    ):
        """Without working_directory, only home dir is scanned (backward compat)."""
        home_plans = tmp_path / ".claude" / "plans"
        home_plans.mkdir(parents=True)
        plan_file = home_plans / "home-plan.md"
        plan_file.write_text("# Home Plan")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = Engine._discover_plan_file()
        assert result == str(plan_file)


class TestSessionPersistenceOnDirSwitch:
    """Bug 2 regression: session state persisted after /dir and _exit_plan_mode."""

    @pytest.mark.asyncio
    async def test_dir_switch_persists_to_store(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        from unittest.mock import AsyncMock

        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        store = AsyncMock()
        store.save = AsyncMock()
        sm = SessionManager(store=store)

        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "hello", "chat1")
        store.save.reset_mock()

        await eng.handle_command("user1", "dir", "api", "chat1")

        store.save.assert_awaited_once()
        saved_session = store.save.call_args[0][0]
        assert saved_session.working_directory == str(d2.resolve())
        assert saved_session.claude_session_id is None

    @pytest.mark.asyncio
    async def test_exit_plan_mode_persists_to_store(
        self, policy_engine, audit_logger, mock_connector, tmp_path, monkeypatch
    ):
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        config = TetherConfig(
            approved_directories=[tmp_path],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        store = MagicMock()
        store.load = AsyncMock(return_value=None)

        # Capture session state snapshots at each save() call
        save_snapshots: list[dict] = []

        async def capture_save(session):
            save_snapshots.append(
                {
                    "mode": session.mode,
                    "claude_session_id": session.claude_session_id,
                    "working_directory": session.working_directory,
                }
            )

        store.save = AsyncMock(side_effect=capture_save)
        sm = SessionManager(store=store)
        coordinator = InteractionCoordinator(mock_connector, config)

        class PlanAgent(BaseAgent):
            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                if not prompt.startswith("Implement"):

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=PlanAgent(),
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")

        # save() should have been called multiple times
        assert len(save_snapshots) >= 2
        # One of the saves should be from _exit_plan_mode with cleared session
        exit_saves = [s for s in save_snapshots if s["claude_session_id"] is None]
        assert len(exit_saves) >= 1
        assert exit_saves[0]["mode"] == "auto"

    @pytest.mark.asyncio
    async def test_dir_switch_persists_to_sqlite(
        self, audit_logger, policy_engine, mock_connector, tmp_path
    ):
        """End-to-end: /dir switch persists working_directory in SQLite."""
        from tether.storage.sqlite import SqliteSessionStore

        d1 = tmp_path / "tether"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        store = SqliteSessionStore(tmp_path / "sessions.db")
        await store.setup()
        try:
            sm = SessionManager(store=store)
            eng = Engine(
                connector=mock_connector,
                agent=FakeAgent(),
                config=config,
                session_manager=sm,
                policy_engine=policy_engine,
                audit=audit_logger,
                store=store,
            )

            await eng.handle_message("user1", "hello", "chat1")
            await eng.handle_command("user1", "dir", "api", "chat1")

            # Verify: load from store should have updated working_directory
            loaded = await store.load("user1", "chat1")
            assert loaded is not None
            assert loaded.working_directory == str(d2.resolve())
            assert loaded.claude_session_id is None
        finally:
            await store.teardown()


class TestDirectoryPersistenceThroughPlanMode:
    """Directory should survive all plan mode transitions (edit, clean_edit, fallback)."""

    @pytest.mark.asyncio
    async def test_dir_persists_through_clean_edit_proceed(
        self, audit_logger, policy_engine, mock_connector, tmp_path, monkeypatch
    ):
        """ExitPlanMode → 'clean_edit': dir survives _exit_plan_mode + recursive handle_message."""
        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        d1 = tmp_path / "project"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        coordinator = InteractionCoordinator(mock_connector, config)

        class DirTrackingAgent(BaseAgent):
            def __init__(self):
                self.working_dirs: list[str] = []
                self.prompts: list[str] = []
                self.session_ids: list[str | None] = []

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.working_dirs.append(session.working_directory)
                self.prompts.append(prompt)
                self.session_ids.append(session.claude_session_id)
                if session.mode == "plan":

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[-1]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        agent = DirTrackingAgent()
        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")
        await eng.handle_command("user1", "plan", "", "chat1")
        await eng.handle_message("user1", "make plan", "chat1")

        session = sm.get("user1", "chat1")
        d2_resolved = str(d2.resolve())
        assert session.working_directory == d2_resolved
        # 3 agent calls: hello, plan, implement (recursive from _exit_plan_mode)
        assert len(agent.working_dirs) == 3
        assert all(d == d2_resolved for d in agent.working_dirs[1:])
        assert agent.prompts[2].startswith("Implement")
        # clean_edit nulls session_id before implementation call
        assert agent.session_ids[2] is None

    @pytest.mark.asyncio
    async def test_dir_persists_through_edit_proceed(
        self, audit_logger, policy_engine, mock_connector, tmp_path, monkeypatch
    ):
        """ExitPlanMode → 'edit': dir preserved (no recursive handle_message)."""
        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        d1 = tmp_path / "project"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        coordinator = InteractionCoordinator(mock_connector, config)

        class DirTrackingAgent(BaseAgent):
            def __init__(self):
                self.working_dirs: list[str] = []

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.working_dirs.append(session.working_directory)
                if session.mode == "plan":

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[-1]
                        await coordinator.resolve_option(req["interaction_id"], "edit")

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        agent = DirTrackingAgent()
        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")
        await eng.handle_command("user1", "plan", "", "chat1")
        await eng.handle_message("user1", "make plan", "chat1")

        session = sm.get("user1", "chat1")
        d2_resolved = str(d2.resolve())
        assert session.working_directory == d2_resolved
        # "edit" doesn't trigger _exit_plan_mode; 2 calls: hello + plan
        assert len(agent.working_dirs) == 2
        assert agent.working_dirs[1] == d2_resolved

    @pytest.mark.asyncio
    async def test_dir_persists_through_fallback_review_edit(
        self, audit_logger, policy_engine, mock_connector, tmp_path, monkeypatch
    ):
        """Fallback plan review → 'edit': dir survives _exit_plan_mode."""
        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        d1 = tmp_path / "project"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        coordinator = InteractionCoordinator(mock_connector, config)

        class FallbackAgent(BaseAgent):
            def __init__(self):
                self.working_dirs: list[str] = []
                self.prompts: list[str] = []

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.working_dirs.append(session.working_directory)
                self.prompts.append(prompt)
                if session.mode == "plan":
                    plan_path = str(d2 / ".claude" / "plans" / "plan.md")
                    await can_use_tool(
                        "Write",
                        {"file_path": plan_path, "content": "# The Plan\nStep 1"},
                        None,
                    )
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        agent = FallbackAgent()
        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")
        await eng.handle_command("user1", "plan", "", "chat1")

        # Bump message_count so fallback guard (> 1) passes after update_from_result
        session = sm.get("user1", "chat1")
        session.message_count = 1

        async def click_review():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[-1]
            await coordinator.resolve_option(req["interaction_id"], "edit")

        task = asyncio.create_task(click_review())
        await eng.handle_message("user1", "refine plan", "chat1")
        await task

        session = sm.get("user1", "chat1")
        d2_resolved = str(d2.resolve())
        assert session.working_directory == d2_resolved
        # 3 calls: hello, plan (fallback triggers), implement
        assert len(agent.working_dirs) == 3
        assert all(d == d2_resolved for d in agent.working_dirs[1:])
        assert agent.prompts[2].startswith("Implement")

    @pytest.mark.asyncio
    async def test_dir_persists_through_fallback_review_default(
        self, audit_logger, policy_engine, mock_connector, tmp_path, monkeypatch
    ):
        """Fallback plan review → 'default': dir survives, mode becomes 'default'."""
        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        d1 = tmp_path / "project"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        coordinator = InteractionCoordinator(mock_connector, config)

        class FallbackAgent(BaseAgent):
            def __init__(self):
                self.working_dirs: list[str] = []
                self.prompts: list[str] = []

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.working_dirs.append(session.working_directory)
                self.prompts.append(prompt)
                if session.mode == "plan":
                    plan_path = str(d2 / ".claude" / "plans" / "plan.md")
                    await can_use_tool(
                        "Write",
                        {"file_path": plan_path, "content": "# The Plan\nStep 1"},
                        None,
                    )
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        agent = FallbackAgent()
        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")
        await eng.handle_command("user1", "plan", "", "chat1")

        session = sm.get("user1", "chat1")
        session.message_count = 1

        async def click_review():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[-1]
            await coordinator.resolve_option(req["interaction_id"], "default")

        task = asyncio.create_task(click_review())
        await eng.handle_message("user1", "refine plan", "chat1")
        await task

        session = sm.get("user1", "chat1")
        d2_resolved = str(d2.resolve())
        assert session.working_directory == d2_resolved
        assert session.mode == "default"
        assert len(agent.working_dirs) == 3
        assert all(d == d2_resolved for d in agent.working_dirs[1:])

    @pytest.mark.asyncio
    async def test_dir_persists_through_multiple_plan_cycles(
        self, audit_logger, policy_engine, mock_connector, tmp_path, monkeypatch
    ):
        """Two full plan→implement cycles preserve directory throughout."""
        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        d1 = tmp_path / "project"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        coordinator = InteractionCoordinator(mock_connector, config)

        class CycleAgent(BaseAgent):
            def __init__(self):
                self.working_dirs: list[str] = []
                self.prompts: list[str] = []

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.working_dirs.append(session.working_directory)
                self.prompts.append(prompt)
                if session.mode == "plan":

                    async def click():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[-1]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    t = asyncio.create_task(click())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await t
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        agent = CycleAgent()
        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")

        # Cycle 1
        await eng.handle_command("user1", "plan", "", "chat1")
        await eng.handle_message("user1", "plan A", "chat1")

        # Cycle 2
        await eng.handle_command("user1", "plan", "", "chat1")
        await eng.handle_message("user1", "plan B", "chat1")

        session = sm.get("user1", "chat1")
        d2_resolved = str(d2.resolve())
        assert session.working_directory == d2_resolved
        # 5 calls: hello, plan-A, implement-A, plan-B, implement-B
        assert len(agent.working_dirs) == 5
        assert all(d == d2_resolved for d in agent.working_dirs[1:])

    @pytest.mark.asyncio
    async def test_dir_persists_through_plan_mode_sqlite(
        self, audit_logger, policy_engine, mock_connector, tmp_path, monkeypatch
    ):
        """ExitPlanMode → 'clean_edit' with SQLite: dir persists in store."""
        from tether.storage.sqlite import SqliteSessionStore

        monkeypatch.setattr(
            Engine, "_discover_plan_file", staticmethod(lambda wd=None: None)
        )
        d1 = tmp_path / "project"
        d2 = tmp_path / "api"
        d1.mkdir()
        d2.mkdir()
        config = TetherConfig(
            approved_directories=[d1, d2],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        store = SqliteSessionStore(tmp_path / "sessions.db")
        await store.setup()
        try:
            coordinator = InteractionCoordinator(mock_connector, config)

            class DirTrackingAgent(BaseAgent):
                def __init__(self):
                    self.working_dirs: list[str] = []

                async def execute(
                    self, prompt, session, *, can_use_tool=None, **kwargs
                ):
                    self.working_dirs.append(session.working_directory)
                    if session.mode == "plan":

                        async def click():
                            await asyncio.sleep(0.05)
                            req = mock_connector.plan_review_requests[-1]
                            await coordinator.resolve_option(
                                req["interaction_id"], "clean_edit"
                            )

                        t = asyncio.create_task(click())
                        await can_use_tool("ExitPlanMode", {}, None)
                        await t
                    return AgentResponse(content="Done", session_id="sid", cost=0.01)

                async def cancel(self, session_id):
                    pass

                async def shutdown(self):
                    pass

            agent = DirTrackingAgent()
            sm = SessionManager(store=store)
            eng = Engine(
                connector=mock_connector,
                agent=agent,
                config=config,
                session_manager=sm,
                policy_engine=policy_engine,
                audit=audit_logger,
                interaction_coordinator=coordinator,
                store=store,
            )

            await eng.handle_message("user1", "hello", "chat1")
            await eng.handle_command("user1", "dir", "api", "chat1")
            await eng.handle_command("user1", "plan", "", "chat1")
            await eng.handle_message("user1", "make plan", "chat1")

            d2_resolved = str(d2.resolve())
            assert all(d == d2_resolved for d in agent.working_dirs[1:])

            loaded = await store.load("user1", "chat1")
            assert loaded is not None
            assert loaded.working_directory == d2_resolved
        finally:
            await store.teardown()


class TestTurnLimitNotification:
    """Verify user notification when the agent hits the max_turns limit."""

    @pytest.mark.asyncio
    async def test_turn_limit_notification_sent(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        class LimitAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                return AgentResponse(
                    content="partial work",
                    session_id="sid",
                    cost=0.01,
                    num_turns=config.max_turns,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=LimitAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "do stuff", "chat1")

        turn_msgs = [
            m for m in mock_connector.sent_messages if "turn limit" in m["text"].lower()
        ]
        assert len(turn_msgs) == 1
        assert str(config.max_turns) in turn_msgs[0]["text"]

    @pytest.mark.asyncio
    async def test_turn_limit_no_notification_when_under_limit(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        class UnderLimitAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                return AgentResponse(
                    content="done",
                    session_id="sid",
                    cost=0.01,
                    num_turns=config.max_turns - 2,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=UnderLimitAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "do stuff", "chat1")

        turn_msgs = [
            m for m in mock_connector.sent_messages if "turn limit" in m["text"].lower()
        ]
        assert len(turn_msgs) == 0

    @pytest.mark.asyncio
    async def test_turn_limit_no_notification_without_connector(
        self, config, audit_logger, policy_engine
    ):
        class LimitAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                return AgentResponse(
                    content="partial",
                    session_id="sid",
                    cost=0.01,
                    num_turns=config.max_turns,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=None,
            agent=LimitAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_message("user1", "do stuff", "chat1")
        assert result == "partial"

    @pytest.mark.asyncio
    async def test_turn_limit_then_clear_resets_for_fresh_execution(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        call_count = 0

        class TrackingAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                nonlocal call_count
                call_count += 1
                return AgentResponse(
                    content=f"run-{call_count}",
                    session_id=f"sid-{call_count}",
                    cost=0.01,
                    num_turns=config.max_turns if call_count == 1 else 1,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=TrackingAgent(),
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "big task", "chat1")
        turn_msgs = [
            m for m in mock_connector.sent_messages if "turn limit" in m["text"].lower()
        ]
        assert len(turn_msgs) == 1

        session = sm.get("user1", "chat1")
        assert session.claude_session_id == "sid-1"

        await eng.handle_command("user1", "clear", "", "chat1")
        session = sm.get("user1", "chat1")
        assert session.claude_session_id is None

        await eng.handle_message("user1", "continue", "chat1")
        session = sm.get("user1", "chat1")
        assert session.claude_session_id == "sid-2"


class TestMessageQueuing:
    """Verify per-chat message queuing during agent execution."""

    @staticmethod
    def _make_slow_agent(gate: asyncio.Event):
        """Agent that blocks until gate is set, capturing all prompts."""

        class SlowFakeAgent(BaseAgent):
            def __init__(self):
                self.prompts: list[str] = []

            async def execute(self, prompt, session, **kwargs):
                self.prompts.append(prompt)
                await gate.wait()
                return AgentResponse(
                    content=f"Done: {prompt}",
                    session_id="slow-sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        return SlowFakeAgent()

    @pytest.mark.asyncio
    async def test_message_queued_during_execution(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)  # let first enter _execute_turn

        result2 = await eng.handle_message("u1", "second", "c1")
        assert result2 == ""

        gate.set()
        result1 = await task

        assert "Done:" in result1
        assert "first" in agent.prompts
        assert "second" in agent.prompts

    @pytest.mark.asyncio
    async def test_queued_message_sends_interrupt_prompt(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "second", "c1")

        assert len(mock_connector.interrupt_prompts) == 1
        assert mock_connector.interrupt_prompts[0]["message_preview"] == "second"

        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_queued_messages_combined(self, config, audit_logger, mock_connector):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "msg A", "c1")
        await eng.handle_message("u1", "msg B", "c1")
        await eng.handle_message("u1", "msg C", "c1")

        gate.set()
        await task

        assert len(agent.prompts) == 2
        combined = agent.prompts[1]
        assert "msg A" in combined
        assert "msg B" in combined
        assert "msg C" in combined
        assert "\n\n" in combined

    @pytest.mark.asyncio
    async def test_queued_messages_logged_individually(
        self, config, audit_logger, mock_connector, tmp_path
    ):
        from tether.storage.sqlite import SqliteSessionStore

        store = SqliteSessionStore(tmp_path / "test.db")
        await store.setup()

        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            store=store,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "queued-A", "c1")
        await eng.handle_message("u1", "queued-B", "c1")

        gate.set()
        await task

        messages = await store.get_messages("u1", "c1")
        user_msgs = [m for m in messages if m["role"] == "user"]
        user_texts = [m["content"] for m in user_msgs]
        assert "queued-A" in user_texts
        assert "queued-B" in user_texts

        await store.teardown()

    @pytest.mark.asyncio
    async def test_approval_bypasses_queue(
        self, config, audit_logger, mock_connector, policy_engine, tmp_dir
    ):
        from tether.core.safety.approvals import PendingApproval

        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        coordinator = ApprovalCoordinator(mock_connector, config)
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            approval_coordinator=coordinator,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        pending = PendingApproval(
            approval_id="test-aid",
            chat_id="c1",
            tool_name="Write",
            tool_input={},
        )
        coordinator.pending["test-aid"] = pending

        result = await eng.handle_message("u1", "reject reason", "c1")
        assert result == ""
        assert "c1" not in eng._pending_messages or not eng._pending_messages["c1"]

        coordinator.pending.pop("test-aid", None)
        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_interaction_bypasses_queue(
        self, config, audit_logger, mock_connector, event_bus
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        ic = InteractionCoordinator(mock_connector, config, event_bus)
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            interaction_coordinator=ic,
            event_bus=event_bus,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        from tether.core.interactions import PendingInteraction

        pending = PendingInteraction(
            interaction_id="test-iid", chat_id="c1", kind="question"
        )
        ic.pending["test-iid"] = pending
        ic._chat_index["c1"] = "test-iid"

        result = await eng.handle_message("u1", "option A", "c1")
        assert result == ""

        ic.pending.pop("test-iid", None)
        ic._chat_index.pop("c1", None)
        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_agent_error_clears_queue(self, config, audit_logger, mock_connector):
        failing_agent = FakeAgent(fail=True)
        eng = Engine(
            connector=mock_connector,
            agent=failing_agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        eng._pending_messages["c1"] = [("u1", "queued msg")]

        result = await eng.handle_message("u1", "trigger", "c1")
        assert "Error:" in result
        assert "c1" not in eng._pending_messages

    @pytest.mark.asyncio
    async def test_clear_command_clears_queue(
        self, config, audit_logger, mock_connector
    ):
        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        eng._pending_messages["c1"] = [("u1", "leftover")]

        await eng.handle_command("u1", "clear", "", "c1")
        assert "c1" not in eng._pending_messages

    @pytest.mark.asyncio
    async def test_message_queued_event_emitted(
        self, config, audit_logger, mock_connector
    ):
        from tether.core.events import MESSAGE_QUEUED

        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(MESSAGE_QUEUED, capture)

        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "second", "c1")

        assert len(events) == 1
        assert events[0].data["text"] == "second"
        assert events[0].data["chat_id"] == "c1"

        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_queue_isolated_between_chats(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task_a = asyncio.create_task(eng.handle_message("u1", "chat-A", "cA"))
        await asyncio.sleep(0)

        task_b = asyncio.create_task(eng.handle_message("u2", "chat-B", "cB"))
        await asyncio.sleep(0)

        assert "cA" in eng._executing_chats
        assert "cB" in eng._executing_chats

        gate.set()
        await task_a
        await task_b

        assert len(agent.prompts) == 2
        assert "chat-A" in agent.prompts
        assert "chat-B" in agent.prompts


class TestMessageInterrupt:
    """Verify interrupt prompt during agent execution."""

    @staticmethod
    def _make_slow_agent(gate: asyncio.Event):
        """Agent that blocks until gate is set. Cancel sets the gate."""

        class SlowFakeAgent(BaseAgent):
            def __init__(self):
                self.prompts: list[str] = []
                self.cancelled: list[str] = []

            async def execute(self, prompt, session, **kwargs):
                self.prompts.append(prompt)
                await gate.wait()
                return AgentResponse(
                    content=f"Done: {prompt}",
                    session_id="slow-sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                self.cancelled.append(session_id)
                gate.set()

            async def shutdown(self):
                pass

        return SlowFakeAgent()

    @pytest.mark.asyncio
    async def test_interrupt_prompt_shown_on_queued_message(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "second msg", "c1")

        assert len(mock_connector.interrupt_prompts) == 1
        prompt = mock_connector.interrupt_prompts[0]
        assert prompt["chat_id"] == "c1"
        assert prompt["message_preview"] == "second msg"

        # No static acknowledgment sent
        ack_msgs = [
            m
            for m in mock_connector.sent_messages
            if "will process after current task" in m.get("text", "")
        ]
        assert len(ack_msgs) == 0

        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_interrupt_send_now_cancels_agent(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "urgent fix", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]

        # Cancel sets gate → agent returns normally → interrupt detected post-return
        await mock_connector.simulate_interrupt(interrupt_id, send_now=True)
        await task

        assert len(agent.cancelled) == 1
        # Queued message processed after interrupted first task
        assert "urgent fix" in agent.prompts
        # "Task interrupted" sent for the first execution
        interrupted_msgs = [
            m
            for m in mock_connector.sent_messages
            if "Task interrupted" in m.get("text", "")
        ]
        assert len(interrupted_msgs) == 1

    @pytest.mark.asyncio
    async def test_interrupt_wait_queues_normally(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "wait msg", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]

        await mock_connector.simulate_interrupt(interrupt_id, send_now=False)

        # Interrupt state cleared but message stays queued
        assert "c1" not in eng._pending_interrupts
        assert "c1" not in eng._interrupted_chats

        gate.set()
        result = await task

        assert "Done:" in result
        assert "wait msg" in agent.prompts

    @pytest.mark.asyncio
    async def test_interrupt_prompt_shown_once_per_pending(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "second", "c1")
        await eng.handle_message("u1", "third", "c1")

        # Only one prompt despite two queued messages
        assert len(mock_connector.interrupt_prompts) == 1

        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_interrupt_prompt_after_wait_shows_again(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "msg A", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]

        await mock_connector.simulate_interrupt(interrupt_id, send_now=False)

        await eng.handle_message("u1", "msg B", "c1")

        # Second prompt after wait
        assert len(mock_connector.interrupt_prompts) == 2

        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_interrupt_cleanup_on_natural_completion(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "queued", "c1")
        msg_id = mock_connector.interrupt_prompts[0]["message_id"]

        # Don't click any button — let execution finish naturally
        gate.set()
        await task

        # Prompt should be edited to "completed"
        completed_edits = [
            m
            for m in mock_connector.edited_messages
            if m["message_id"] == msg_id and "Task completed" in m["text"]
        ]
        assert len(completed_edits) == 1

    @pytest.mark.asyncio
    async def test_interrupt_stale_button_returns_false(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "queued", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]

        gate.set()
        await task

        # Click after execution completed — interrupt_id already cleaned up
        result = await mock_connector.simulate_interrupt(interrupt_id, send_now=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_interrupt_event_emitted(self, config, audit_logger, mock_connector):
        from tether.core.events import EXECUTION_INTERRUPTED

        events = []

        async def capture(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(EXECUTION_INTERRUPTED, capture)

        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
            event_bus=bus,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "urgent", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]

        # Cancel sets gate → execute returns normally → interrupt detected
        await mock_connector.simulate_interrupt(interrupt_id, send_now=True)
        await task

        assert len(events) == 1
        assert events[0].data["chat_id"] == "c1"

    @pytest.mark.asyncio
    async def test_interrupt_suppresses_partial_response(
        self, config, audit_logger, mock_connector
    ):
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "interrupt me", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]

        await mock_connector.simulate_interrupt(interrupt_id, send_now=True)
        await task

        # The first task's "Done: first" should NOT appear in sent messages
        first_response_msgs = [
            m
            for m in mock_connector.sent_messages
            if "Done: first" in m.get("text", "")
        ]
        assert len(first_response_msgs) == 0

    @pytest.mark.asyncio
    async def test_interrupt_cleanup_schedules_deletion(
        self, config, audit_logger, mock_connector
    ):
        """Natural completion edits prompt to 'Task completed.' and schedules cleanup."""
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "queued", "c1")
        msg_id = mock_connector.interrupt_prompts[0]["message_id"]

        gate.set()
        await task

        cleanups = [
            c for c in mock_connector.scheduled_cleanups if c["message_id"] == msg_id
        ]
        assert len(cleanups) == 1
        assert cleanups[0]["chat_id"] == "c1"
        assert cleanups[0]["delay"] == 5.0

    @pytest.mark.asyncio
    async def test_interrupt_send_now_schedules_interrupted_msg_cleanup(
        self, config, audit_logger
    ):
        """After interrupt, the 'Task interrupted.' message is scheduled for cleanup."""
        from tests.conftest import MockConnector

        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)
        connector = MockConnector(support_streaming=True)

        eng = Engine(
            connector=connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "urgent fix", "c1")
        interrupt_id = connector.interrupt_prompts[0]["interrupt_id"]

        await connector.simulate_interrupt(interrupt_id, send_now=True)
        await task

        interrupted_cleanups = [
            c
            for c in connector.scheduled_cleanups
            if any(
                m["text"] == "\u26a1 Task interrupted."
                and m.get("message_id") == c["message_id"]
                for m in connector.sent_messages
            )
        ]
        assert len(interrupted_cleanups) == 1
        assert interrupted_cleanups[0]["delay"] == 5.0

    @pytest.mark.asyncio
    async def test_fallback_ack_schedules_cleanup(self, config, audit_logger):
        """Fallback ack (when send_interrupt_prompt returns None) schedules cleanup."""
        from tests.conftest import MockConnector

        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)
        connector = MockConnector(support_streaming=True)

        # Override send_interrupt_prompt to return None (simulate unsupported)
        async def _no_interrupt_prompt(chat_id, interrupt_id, preview):
            return None

        connector.send_interrupt_prompt = _no_interrupt_prompt

        eng = Engine(
            connector=connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "queued msg", "c1")

        gate.set()
        await task

        fallback_cleanups = [
            c
            for c in connector.scheduled_cleanups
            if any(
                "will process after current task" in m.get("text", "")
                and m.get("message_id") == c["message_id"]
                for m in connector.sent_messages
            )
        ]
        assert len(fallback_cleanups) == 1
        assert fallback_cleanups[0]["delay"] == 5.0


class TestInterruptIsolation:
    """Cross-chat interrupt isolation and message ordering after interrupt."""

    @staticmethod
    def _make_slow_agent(gate: asyncio.Event):
        """Agent that blocks until gate is set. Cancel sets the gate."""

        class SlowFakeAgent(BaseAgent):
            def __init__(self):
                self.prompts: list[str] = []
                self.cancelled: list[str] = []

            async def execute(self, prompt, session, **kwargs):
                self.prompts.append(prompt)
                await gate.wait()
                return AgentResponse(
                    content=f"Done: {prompt}",
                    session_id="slow-sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                self.cancelled.append(session_id)
                gate.set()

            async def shutdown(self):
                pass

        return SlowFakeAgent()

    @pytest.mark.asyncio
    async def test_cross_chat_interrupt_isolation(
        self, config, audit_logger, mock_connector
    ):
        """Interrupt in chat A must not produce interrupt side-effects in chat B."""
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task_a = asyncio.create_task(eng.handle_message("u1", "task A", "chatA"))
        await asyncio.sleep(0)

        # Chat B starts after A is executing — gets its own execution slot
        # We need a separate gate for B since A's gate will be set by cancel
        gate_b = asyncio.Event()

        # Override agent to use separate gate for second prompt
        original_execute = agent.execute

        async def dual_execute(prompt, session, **kwargs):
            if "task B" in prompt:
                agent.prompts.append(prompt)
                await gate_b.wait()
                return AgentResponse(
                    content=f"Done: {prompt}", session_id="sid-b", cost=0.01
                )
            return await original_execute(prompt, session, **kwargs)

        agent.execute = dual_execute

        # Queue interrupt on chat A
        await eng.handle_message("u1", "interrupt A", "chatA")
        assert len(mock_connector.interrupt_prompts) == 1
        assert mock_connector.interrupt_prompts[0]["chat_id"] == "chatA"

        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]
        await mock_connector.simulate_interrupt(interrupt_id, send_now=True)
        await task_a

        # Now start chat B after A is done
        task_b = asyncio.create_task(eng.handle_message("u2", "task B", "chatB"))
        await asyncio.sleep(0)
        gate_b.set()
        await task_b

        # Chat B should have no interrupted messages
        b_interrupted = [
            m
            for m in mock_connector.sent_messages
            if m.get("chat_id") == "chatB"
            and "interrupted" in m.get("text", "").lower()
        ]
        assert len(b_interrupted) == 0

        # Chat B should NOT appear in any interrupt prompts
        b_prompts = [
            p for p in mock_connector.interrupt_prompts if p["chat_id"] == "chatB"
        ]
        assert len(b_prompts) == 0

    @pytest.mark.asyncio
    async def test_message_ordering_after_interrupt(
        self, config, audit_logger, mock_connector
    ):
        """After Send Now, the queued message executes next in correct order."""
        gate = asyncio.Event()
        agent = self._make_slow_agent(gate)

        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        task = asyncio.create_task(eng.handle_message("u1", "first", "c1"))
        await asyncio.sleep(0)

        await eng.handle_message("u1", "second-urgent", "c1")
        interrupt_id = mock_connector.interrupt_prompts[0]["interrupt_id"]
        await mock_connector.simulate_interrupt(interrupt_id, send_now=True)
        await task

        # Agent received "first" then "second-urgent" — correct execution order
        assert agent.prompts[0] == "first"
        assert agent.prompts[1] == "second-urgent"


class TestPlanModeRegression:
    """Verify /plan clears session context and blocks non-plan edits."""

    @pytest.mark.asyncio
    async def test_plan_command_preserves_claude_session_id(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        agent = FakeAgent()
        sm = SessionManager()
        eng = Engine(
            connector=mock_connector,
            agent=agent,
            config=config,
            session_manager=sm,
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        # Build up a session with a claude_session_id
        await eng.handle_message("user1", "hello", "chat1")
        session = sm.get("user1", "chat1")
        assert session.claude_session_id == "test-session-123"

        # Switch to plan mode
        await eng.handle_command("user1", "plan", "", "chat1")
        session = sm.get("user1", "chat1")
        assert session.claude_session_id == "test-session-123"
        assert session.mode == "plan"

    @pytest.mark.asyncio
    async def test_plan_command_persists_session(
        self, tmp_path, audit_logger, policy_engine, mock_connector
    ):
        from tether.storage.sqlite import SqliteSessionStore

        config = TetherConfig(
            approved_directories=[tmp_path],
            audit_log_path=tmp_path / "audit.jsonl",
        )
        store = SqliteSessionStore(tmp_path / "sessions.db")
        await store.setup()
        try:
            agent = FakeAgent()
            sm = SessionManager(store=store)
            eng = Engine(
                connector=mock_connector,
                agent=agent,
                config=config,
                session_manager=sm,
                policy_engine=policy_engine,
                audit=audit_logger,
                store=store,
            )

            await eng.handle_message("user1", "hello", "chat1")
            await eng.handle_command("user1", "plan", "", "chat1")

            loaded = await store.load("user1", "chat1")
            assert loaded is not None
            assert loaded.claude_session_id is not None
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_write_to_source_file_denied_in_plan_mode(
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

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        session.mode = "plan"

        hook = agent.last_can_use_tool
        result = await hook(
            "Write", {"file_path": "/tmp/project/src/main.py", "content": "x"}, None
        )
        assert result["behavior"] == "deny"
        assert "plan mode" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_edit_to_source_file_denied_in_plan_mode(
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

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        session.mode = "plan"

        hook = agent.last_can_use_tool
        result = await hook(
            "Edit",
            {
                "file_path": "/tmp/project/src/main.py",
                "old_string": "a",
                "new_string": "b",
            },
            None,
        )
        assert result["behavior"] == "deny"

    @pytest.mark.asyncio
    async def test_write_to_plan_file_allowed_in_plan_mode(
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

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        session.mode = "plan"

        hook = agent.last_can_use_tool
        # .claude/plans/ path should be allowed
        result = await hook(
            "Write",
            {"file_path": "/home/user/.claude/plans/plan.md", "content": "the plan"},
            None,
        )
        # Should NOT be denied — goes through to gatekeeper
        assert result != {"behavior": "deny"}

    @pytest.mark.asyncio
    async def test_dot_plan_file_allowed_in_plan_mode(
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

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        session.mode = "plan"

        hook = agent.last_can_use_tool
        result = await hook(
            "Write",
            {"file_path": "/tmp/project/feature.plan", "content": "plan"},
            None,
        )
        assert result != {"behavior": "deny"}

    @pytest.mark.asyncio
    async def test_write_allowed_outside_plan_mode(
        self, config, audit_logger, policy_engine, tmp_dir
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

        await eng.handle_message("user1", "hello", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "default"

        hook = agent.last_can_use_tool
        result = await hook(
            "Write",
            {"file_path": str(tmp_dir / "src" / "main.py"), "content": "x"},
            None,
        )
        # Should NOT be the plan-mode deny — goes through to gatekeeper instead
        assert not (isinstance(result, dict) and result.get("behavior") == "deny")


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


class TestCombineQueuedMessages:
    """Verify _combine_queued_messages static method behavior."""

    def test_single_message_returns_text_directly(self):
        result = Engine._combine_queued_messages([("user1", "hello world")])
        assert result == "hello world"

    def test_multiple_messages_joined_with_double_newline(self):
        messages = [
            ("user1", "first message"),
            ("user2", "second message"),
            ("user1", "third message"),
        ]
        result = Engine._combine_queued_messages(messages)
        assert result == "first message\n\nsecond message\n\nthird message"

    def test_two_messages_joined(self):
        messages = [("u1", "alpha"), ("u1", "beta")]
        result = Engine._combine_queued_messages(messages)
        assert result == "alpha\n\nbeta"


class TestResolvePlanContentFallbacks:
    """Verify _resolve_plan_content priority: disk → cached write → fallback."""

    def test_disk_file_preferred(self, config, audit_logger, tmp_path):
        from tether.core.engine import _ToolCallbackState

        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Disk Plan Content")

        state = _ToolCallbackState()
        state.plan_file_path = str(plan_file)
        state.plan_file_content = "# Cached Content (should not be used)"

        result = eng._resolve_plan_content(state, "fallback text")
        assert result == "# Disk Plan Content"

    def test_cached_write_fallback_when_no_disk_file(self, config, audit_logger):
        from tether.core.engine import _ToolCallbackState

        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        state = _ToolCallbackState()
        state.plan_file_path = "/nonexistent/path/plan.md"
        state.plan_file_content = "# Cached Plan"

        result = eng._resolve_plan_content(state, "fallback text")
        assert result == "# Cached Plan"

    def test_response_fallback_when_neither_exists(self, config, audit_logger):
        from unittest.mock import patch

        from tether.core.engine import _ToolCallbackState

        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        state = _ToolCallbackState()
        # No plan_file_path, no cached content

        with patch.object(eng, "_discover_plan_file", return_value=None):
            result = eng._resolve_plan_content(state, "the agent response content")
        assert result == "the agent response content"

    def test_disk_error_falls_back_to_cached_write(
        self, config, audit_logger, tmp_path
    ):
        from tether.core.engine import _ToolCallbackState

        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        # Create a plan_file_path that exists as a directory (will cause read error)
        bad_path = tmp_path / "plan_as_dir.md"
        bad_path.mkdir()

        state = _ToolCallbackState()
        state.plan_file_path = str(bad_path)
        state.plan_file_content = "# Cached Fallback"

        result = eng._resolve_plan_content(state, "response fallback")
        assert result == "# Cached Fallback"

    def test_disk_error_no_cache_falls_back_to_response(
        self, config, audit_logger, tmp_path
    ):
        from tether.core.engine import _ToolCallbackState

        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            audit=audit_logger,
        )

        bad_path = tmp_path / "plan_as_dir.md"
        bad_path.mkdir()

        state = _ToolCallbackState()
        state.plan_file_path = str(bad_path)
        # No cached content

        result = eng._resolve_plan_content(state, "last resort fallback")
        assert result == "last resort fallback"


class TestDefaultButtonSessionMode:
    """Verify _exit_plan_mode sets session.mode correctly for different target_modes."""

    @pytest.mark.asyncio
    async def test_fallback_default_sets_session_mode_to_default(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """When fallback review fires and user clicks 'default', session.mode = 'default'."""
        coordinator = InteractionCoordinator(mock_connector, config)

        class PlanSkipAgent(BaseAgent):
            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                if can_use_tool and not prompt.startswith("Implement"):
                    plan_path = f"{session.working_directory}/.claude/plans/plan.md"
                    await can_use_tool(
                        "Write",
                        {"file_path": plan_path, "content": "# Plan\n1. Do thing"},
                        None,
                    )
                return AgentResponse(
                    content="# Plan\n1. Do thing",
                    session_id="sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=PlanSkipAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        session = await eng.session_manager.get_or_create(
            "user1", "chat1", str(config.approved_directories[0])
        )
        session.mode = "plan"
        session.message_count = 2

        async def click_default():
            await asyncio.sleep(0.05)
            req = mock_connector.plan_review_requests[0]
            await coordinator.resolve_option(req["interaction_id"], "default")

        task = asyncio.create_task(click_default())
        await eng.handle_message("user1", "Plan it", "chat1")
        await task

        assert session.mode == "default"

    @pytest.mark.asyncio
    async def test_clean_edit_sets_session_mode_to_auto(
        self, config, policy_engine, audit_logger, mock_connector
    ):
        """When user clicks 'clean_edit', _exit_plan_mode sets session.mode = 'auto'."""
        coordinator = InteractionCoordinator(mock_connector, config)

        class PlanAgent(BaseAgent):
            def __init__(self):
                self.last_can_use_tool = None

            async def execute(self, prompt, session, *, can_use_tool=None, **kwargs):
                self.last_can_use_tool = can_use_tool
                if not prompt.startswith("Implement"):

                    async def click_clean():
                        await asyncio.sleep(0.05)
                        req = mock_connector.plan_review_requests[0]
                        await coordinator.resolve_option(
                            req["interaction_id"], "clean_edit"
                        )

                    task = asyncio.create_task(click_clean())
                    await can_use_tool("ExitPlanMode", {}, None)
                    await task
                return AgentResponse(content="Done", session_id="sid", cost=0.01)

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=PlanAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
            interaction_coordinator=coordinator,
        )

        await eng.handle_message("user1", "Plan it", "chat1")
        session = eng.session_manager.get("user1", "chat1")
        assert session.mode == "auto"


class TestTurnLimitWarningContent:
    """Verify turn limit warning includes actionable guidance."""

    @pytest.mark.asyncio
    async def test_warning_includes_continue_option(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        class LimitAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                return AgentResponse(
                    content="partial",
                    session_id="sid",
                    cost=0.01,
                    num_turns=config.max_turns,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=LimitAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "do stuff", "chat1")

        turn_msgs = [
            m for m in mock_connector.sent_messages if "turn limit" in m["text"].lower()
        ]
        assert len(turn_msgs) == 1
        text = turn_msgs[0]["text"]
        assert "Send a message to continue" in text
        assert "/clear" in text
        assert "TETHER_MAX_TURNS" in text

    @pytest.mark.asyncio
    async def test_warning_exceeds_max_turns(
        self, config, audit_logger, policy_engine, mock_connector
    ):
        """Warning also fires when num_turns exceeds max_turns (not just equals)."""

        class OverLimitAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                return AgentResponse(
                    content="exceeded",
                    session_id="sid",
                    cost=0.01,
                    num_turns=config.max_turns + 3,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=mock_connector,
            agent=OverLimitAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        await eng.handle_message("user1", "do stuff", "chat1")

        turn_msgs = [
            m for m in mock_connector.sent_messages if "turn limit" in m["text"].lower()
        ]
        assert len(turn_msgs) == 1


class TestEngineResilience:
    """Tests for engine-level retry, message preservation, timeout, and backoff."""

    @pytest.mark.asyncio
    async def test_engine_retries_transient_error(
        self, config, audit_logger, policy_engine
    ):
        call_count = 0

        class RetryAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return AgentResponse(
                        content="temporarily unavailable",
                        session_id="sid",
                        cost=0.0,
                        is_error=True,
                    )
                return AgentResponse(
                    content="success",
                    session_id="sid",
                    cost=0.01,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=None,
            agent=RetryAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await eng.handle_message("u1", "hello", "c1")

        assert "success" in result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_engine_no_retry_permanent_error(
        self, config, audit_logger, policy_engine
    ):
        call_count = 0

        class PermanentErrorAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                nonlocal call_count
                call_count += 1
                return AgentResponse(
                    content="authentication_error: invalid key",
                    session_id="sid",
                    cost=0.0,
                    is_error=True,
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=None,
            agent=PermanentErrorAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_message("u1", "hello", "c1")
        assert "authentication_error" in result
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_pending_messages_preserved_on_transient_error(
        self, config, audit_logger, policy_engine
    ):
        class TransientFailAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                raise AgentError(
                    "The AI service is temporarily unavailable. Please try again in a moment."
                )

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=None,
            agent=TransientFailAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        eng._pending_messages["c1"] = [("u1", "queued msg")]

        result = await eng.handle_message("u1", "trigger", "c1")
        assert "Error:" in result
        assert eng._pending_messages.get("c1") == [("u1", "queued msg")]

    @pytest.mark.asyncio
    async def test_pending_messages_dropped_on_permanent_error(
        self, config, audit_logger, policy_engine
    ):
        class PermanentFailAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                raise AgentError("Agent error: something broke permanently")

            async def cancel(self, session_id):
                pass

            async def shutdown(self):
                pass

        eng = Engine(
            connector=None,
            agent=PermanentFailAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        eng._pending_messages["c1"] = [("u1", "queued msg")]

        result = await eng.handle_message("u1", "trigger", "c1")
        assert "Error:" in result
        assert "c1" not in eng._pending_messages

    @pytest.mark.asyncio
    async def test_agent_timeout_cancels_and_raises(
        self, config, audit_logger, policy_engine
    ):
        cancel_called = False

        class HangingAgent(BaseAgent):
            async def execute(self, prompt, session, **kwargs):
                await asyncio.sleep(9999)

            async def cancel(self, session_id):
                nonlocal cancel_called
                cancel_called = True

            async def shutdown(self):
                pass

        config_short = TetherConfig(
            approved_directories=config.approved_directories,
            max_turns=5,
            agent_timeout_seconds=1,
            audit_log_path=config.audit_log_path,
        )

        eng = Engine(
            connector=None,
            agent=HangingAgent(),
            config=config_short,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        result = await eng.handle_message("u1", "hello", "c1")
        assert "timed out" in result.lower()
        assert cancel_called

    @pytest.mark.asyncio
    async def test_sustained_degradation_backoff(
        self, config, audit_logger, policy_engine
    ):
        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(),
            policy_engine=policy_engine,
            audit=audit_logger,
        )

        assert eng._failure_backoff("c1") == 0

        now = time.monotonic()
        eng._recent_failures["c1"] = [now - 10, now - 5, now - 1]
        backoff = eng._failure_backoff("c1")
        assert backoff == 30  # 10 * 3

        eng._recent_failures["c1"] = [now] * 7
        backoff = eng._failure_backoff("c1")
        assert backoff == 60  # capped at 60

    def test_is_retryable_response_true(self):
        resp = AgentResponse(content="temporarily unavailable", is_error=True)
        assert Engine._is_retryable_response(resp) is True

    def test_is_retryable_response_false_not_error(self):
        resp = AgentResponse(content="temporarily unavailable", is_error=False)
        assert Engine._is_retryable_response(resp) is False

    def test_is_retryable_response_false_permanent(self):
        resp = AgentResponse(content="authentication_error: invalid key", is_error=True)
        assert Engine._is_retryable_response(resp) is False
