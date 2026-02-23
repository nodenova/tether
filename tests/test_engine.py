"""Tests for the central engine — safety hook wiring."""

import asyncio

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
        self, config, policy_engine, audit_logger
    ):
        """When no .plan file is written, streaming buffer is used as fallback."""
        from tests.conftest import MockConnector

        streaming_connector = MockConnector(support_streaming=True)
        coordinator = InteractionCoordinator(streaming_connector, config)
        config.streaming_enabled = True

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
                    await can_use_tool(
                        "Write",
                        {
                            "file_path": ".claude/plans/plan.md",
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
        assert session.is_active is False
        assert "chat1" not in eng._gatekeeper._auto_approved_chats

        # Next get_or_create returns a new session
        new_session = await eng.session_manager.get_or_create(
            "user1", "chat1", str(config.approved_directories[0])
        )
        assert new_session.session_id != original_id
        assert new_session.is_active is True


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
        assert "(active)" in result

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
