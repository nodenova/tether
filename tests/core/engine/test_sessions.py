"""Engine tests — session management, persistence, directory switches."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.core.engine.conftest import FakeAgent
from tether.agents.base import AgentResponse, BaseAgent
from tether.core.config import TetherConfig
from tether.core.engine import Engine
from tether.core.interactions import InteractionCoordinator
from tether.core.session import SessionManager
from tether.middleware.base import MessageContext, MiddlewareChain


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


class TestTetherDirCreatedOnSessionInit:
    """Verify .tether/ is created on first message and on commands."""

    @pytest.mark.asyncio
    async def test_tether_dir_created_on_first_message(
        self, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "proj1"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
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

        assert not (d1 / ".tether").exists()
        await eng.handle_message("user1", "hello", "chat1")
        assert (d1 / ".tether").is_dir()
        assert (d1 / ".tether" / ".gitignore").is_file()

    @pytest.mark.asyncio
    async def test_tether_dir_created_on_command(
        self, policy_engine, mock_connector, tmp_path
    ):
        d1 = tmp_path / "proj1"
        d1.mkdir()
        config = TetherConfig(
            approved_directories=[d1],
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

        assert not (d1 / ".tether").exists()
        await eng.handle_command("user1", "status", "", "chat1")
        assert (d1 / ".tether").is_dir()
        assert (d1 / ".tether" / ".gitignore").is_file()


class TestSessionPersistenceOnDirSwitch:
    """Bug 2 regression: session state persisted after /dir and _exit_plan_mode."""

    @pytest.mark.asyncio
    async def test_dir_switch_persists_to_store(
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
        store = AsyncMock()
        store.load = AsyncMock(return_value=None)
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
        from unittest.mock import MagicMock

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


class TestDirectoryPersistenceAcrossRestart:
    """Verify /dir selection survives engine restart via two-tier storage."""

    @pytest.mark.asyncio
    async def test_dir_survives_restart(
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
        session_db = tmp_path / "sessions.db"
        msg_db_1 = d1 / ".tether" / "tether.db"
        msg_db_1.parent.mkdir(parents=True, exist_ok=True)

        # Engine 1: switch to api
        session_store_1 = SqliteSessionStore(session_db)
        message_store_1 = SqliteSessionStore(msg_db_1)
        await session_store_1.setup()
        await message_store_1.setup()
        eng1 = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(store=session_store_1),
            policy_engine=policy_engine,
            audit=audit_logger,
            store=session_store_1,
            message_store=message_store_1,
        )
        await eng1.handle_message("user1", "hello", "chat1")
        await eng1.handle_command("user1", "dir", "api", "chat1")
        session = eng1.session_manager.get("user1", "chat1")
        assert session.working_directory == str(d2.resolve())
        await session_store_1.teardown()
        await message_store_1.teardown()

        # Engine 2: fresh stores on same session DB — simulates restart
        session_store_2 = SqliteSessionStore(session_db)
        msg_db_default = d1 / ".tether" / "tether.db"
        message_store_2 = SqliteSessionStore(msg_db_default)
        await session_store_2.setup()
        await message_store_2.setup()
        eng2 = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(store=session_store_2),
            policy_engine=policy_engine,
            audit=audit_logger,
            store=session_store_2,
            message_store=message_store_2,
        )
        await eng2.handle_message("user1", "continue work", "chat1")
        session2 = eng2.session_manager.get("user1", "chat1")
        assert session2.working_directory == str(d2.resolve())
        await session_store_2.teardown()
        await message_store_2.teardown()

    @pytest.mark.asyncio
    async def test_session_restore_realigns_message_store(
        self, audit_logger, policy_engine, tmp_path
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
        session_db = tmp_path / "sessions.db"

        # Pre-seed: save session with api directory
        session_store = SqliteSessionStore(session_db)
        await session_store.setup()
        from tether.core.session import Session

        await session_store.save(
            Session(
                session_id="pre-seed",
                user_id="user1",
                chat_id="chat1",
                working_directory=str(d2.resolve()),
            )
        )

        msg_db = d1 / ".tether" / "tether.db"
        msg_db.parent.mkdir(parents=True, exist_ok=True)
        message_store = SqliteSessionStore(msg_db)
        await message_store.setup()

        eng = Engine(
            connector=None,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(store=session_store),
            policy_engine=policy_engine,
            audit=audit_logger,
            store=session_store,
            message_store=message_store,
            storage_path_pinned=False,
            storage_path_template=config.storage_path,
        )
        await eng.handle_message("user1", "hello", "chat1")

        # Message store should have been realigned to api's tether.db
        expected_path = str(d2.resolve() / ".tether" / "tether.db")
        assert message_store._db_path == expected_path
        await session_store.teardown()
        await message_store.teardown()

    @pytest.mark.asyncio
    async def test_dir_switch_saves_to_session_store_not_message_store(
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
        session_db = tmp_path / "sessions.db"
        msg_db = d1 / ".tether" / "tether.db"
        msg_db.parent.mkdir(parents=True, exist_ok=True)

        session_store = SqliteSessionStore(session_db)
        message_store = SqliteSessionStore(msg_db)
        await session_store.setup()
        await message_store.setup()

        eng = Engine(
            connector=mock_connector,
            agent=FakeAgent(),
            config=config,
            session_manager=SessionManager(store=session_store),
            policy_engine=policy_engine,
            audit=audit_logger,
            store=session_store,
            message_store=message_store,
        )
        await eng.handle_message("user1", "hello", "chat1")
        await eng.handle_command("user1", "dir", "api", "chat1")

        # Session should be in the session store (fixed DB)
        loaded = await session_store.load("user1", "chat1")
        assert loaded is not None
        assert loaded.working_directory == str(d2.resolve())
        await session_store.teardown()
        await message_store.teardown()
