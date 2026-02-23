"""Central orchestrator — connector-agnostic message handling with safety."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from tether.core.config import build_directory_names
from tether.core.events import (
    COMMAND_TEST,
    ENGINE_STARTED,
    ENGINE_STOPPED,
    MESSAGE_IN,
    MESSAGE_OUT,
    Event,
    EventBus,
)
from tether.core.safety.audit import AuditLogger
from tether.core.safety.gatekeeper import ToolGatekeeper
from tether.core.safety.policy import PolicyEngine
from tether.core.safety.sandbox import SandboxEnforcer
from tether.exceptions import AgentError
from tether.middleware.base import MessageContext
from tether.storage.sqlite import SqliteSessionStore

if TYPE_CHECKING:
    from tether.agents.base import BaseAgent, ToolActivity
    from tether.connectors.base import BaseConnector
    from tether.core.config import TetherConfig
    from tether.core.interactions import InteractionCoordinator
    from tether.core.safety.approvals import ApprovalCoordinator
    from tether.core.session import Session, SessionManager
    from tether.middleware.base import MiddlewareChain
    from tether.plugins.registry import PluginRegistry
    from tether.storage.base import SessionStore

logger = structlog.get_logger()

_STREAMING_CURSOR = "\u258d"
_MAX_STREAMING_DISPLAY = 4000


class _ToolCallbackState:
    __slots__ = (
        "clean_proceed",
        "plan_file_content",
        "plan_file_path",
        "plan_review_shown",
        "target_mode",
    )

    def __init__(self) -> None:
        self.clean_proceed = False
        self.plan_review_shown = False
        self.plan_file_content: str | None = None
        self.plan_file_path: str | None = None
        self.target_mode: str = "edit"


class _StreamingResponder:
    """Accumulates text chunks and progressively edits a Telegram message."""

    def __init__(
        self,
        connector: BaseConnector,
        chat_id: str,
        *,
        throttle_seconds: float = 1.5,
    ) -> None:
        self._connector = connector
        self._chat_id = chat_id
        self._throttle = throttle_seconds
        self._buffer = ""
        self._message_id: str | None = None
        self._last_edit: float = 0.0
        self._active = True
        self._has_activity: bool = False
        self._tool_counts: dict[str, int] = {}

    @property
    def buffer(self) -> str:
        return self._buffer

    def _build_display(self) -> str:
        text = self._buffer[:_MAX_STREAMING_DISPLAY]
        return text + _STREAMING_CURSOR

    def _build_tools_summary(self) -> str:
        if not self._tool_counts:
            return ""
        parts = []
        for name, count in self._tool_counts.items():
            parts.append(f"{name} x{count}" if count > 1 else name)
        return "\U0001f9f0 " + ", ".join(parts)

    async def on_chunk(self, text: str) -> None:
        if not self._active:
            return

        if self._has_activity:
            await self._connector.clear_activity(self._chat_id)
            self._has_activity = False

        self._buffer += text

        if self._message_id is None:
            display = self._build_display()
            msg_id = await self._connector.send_message_with_id(self._chat_id, display)
            if msg_id is None:
                self._active = False
                return
            self._message_id = msg_id
            self._last_edit = time.monotonic()
            return

        now = time.monotonic()
        if now - self._last_edit >= self._throttle:
            display = self._build_display()
            await self._connector.edit_message(self._chat_id, self._message_id, display)
            self._last_edit = now

    async def on_activity(self, activity: ToolActivity | None) -> None:
        if not self._active:
            return

        if activity is None:
            return

        self._tool_counts[activity.tool_name] = (
            self._tool_counts.get(activity.tool_name, 0) + 1
        )
        await self._connector.send_activity(
            self._chat_id, activity.tool_name, activity.description
        )
        self._has_activity = True

    def reset(self) -> None:
        self._message_id = None
        self._buffer = ""
        self._has_activity = False
        self._tool_counts = {}
        self._last_edit = 0.0

    async def deactivate(self) -> None:
        """Suppress all further streaming and clear any visible activity."""
        self._active = False
        await self._connector.clear_activity(self._chat_id)

    async def finalize(self, final_text: str) -> bool:
        if not self._active or self._message_id is None:
            return False

        summary = self._build_tools_summary()
        if summary:
            final_text = final_text + "\n\n" + summary

        if len(final_text) <= _MAX_STREAMING_DISPLAY:
            await self._connector.edit_message(
                self._chat_id, self._message_id, final_text
            )
        else:
            first_chunk = final_text[:_MAX_STREAMING_DISPLAY]
            await self._connector.edit_message(
                self._chat_id, self._message_id, first_chunk
            )
            remainder = final_text[_MAX_STREAMING_DISPLAY:]
            await self._connector.send_message(self._chat_id, remainder)

        return True


class Engine:
    def __init__(
        self,
        connector: BaseConnector | None,
        agent: BaseAgent,
        config: TetherConfig,
        session_manager: SessionManager,
        *,
        policy_engine: PolicyEngine | None = None,
        sandbox: SandboxEnforcer | None = None,
        audit: AuditLogger | None = None,
        approval_coordinator: ApprovalCoordinator | None = None,
        interaction_coordinator: InteractionCoordinator | None = None,
        event_bus: EventBus | None = None,
        plugin_registry: PluginRegistry | None = None,
        middleware_chain: MiddlewareChain | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.connector = connector
        self.agent = agent
        self.config = config
        self.session_manager = session_manager
        self.policy_engine = policy_engine
        self.sandbox = sandbox or SandboxEnforcer(
            [*config.approved_directories, Path.home() / ".claude" / "plans"]
        )
        self._dir_names = build_directory_names(config.approved_directories)
        self._default_directory = str(config.approved_directories[0])
        self.audit = audit or AuditLogger(config.audit_log_path)
        self.approval_coordinator = approval_coordinator
        self.interaction_coordinator = interaction_coordinator
        self.event_bus = event_bus or EventBus()
        self.plugin_registry = plugin_registry
        self.middleware_chain = middleware_chain
        self._store = store
        self._message_store: SqliteSessionStore | None = (
            store if isinstance(store, SqliteSessionStore) else None
        )

        self._gatekeeper = ToolGatekeeper(
            sandbox=self.sandbox,
            audit=self.audit,
            event_bus=self.event_bus,
            policy_engine=self.policy_engine,
            approval_coordinator=self.approval_coordinator,
            approval_timeout=config.approval_timeout_seconds,
        )

        if connector:
            if self.middleware_chain and self.middleware_chain.has_middleware():
                connector.set_message_handler(self._handle_with_middleware)
            else:
                connector.set_message_handler(self.handle_message)
            if approval_coordinator:
                connector.set_approval_resolver(approval_coordinator.resolve_approval)
            if interaction_coordinator:
                connector.set_interaction_resolver(
                    interaction_coordinator.resolve_option
                )
            connector.set_auto_approve_handler(
                self._gatekeeper.enable_tool_auto_approve
            )
            connector.set_command_handler(self.handle_command)

    async def startup(self) -> None:
        if self._store:
            await self._store.setup()
        if self.plugin_registry:
            from tether.plugins.base import PluginContext

            ctx = PluginContext(event_bus=self.event_bus, config=self.config)
            await self.plugin_registry.init_all(ctx)
            await self.plugin_registry.start_all()
        await self.event_bus.emit(Event(name=ENGINE_STARTED))

    async def shutdown(self) -> None:
        await self.event_bus.emit(Event(name=ENGINE_STOPPED))
        if self.plugin_registry:
            await self.plugin_registry.stop_all()
        if self._store:
            await self._store.teardown()
        await self.agent.shutdown()

    async def handle_message(self, user_id: str, text: str, chat_id: str) -> str:
        if self.approval_coordinator and self.approval_coordinator.has_pending(chat_id):
            resolved = await self.approval_coordinator.reject_with_reason(chat_id, text)
            if resolved:
                logger.debug(
                    "message_routed_to_approval_rejection",
                    chat_id=chat_id,
                    text_length=len(text),
                )
                return ""

        if self.interaction_coordinator and self.interaction_coordinator.has_pending(
            chat_id
        ):
            resolved = await self.interaction_coordinator.resolve_text(chat_id, text)
            if resolved:
                logger.debug(
                    "message_routed_to_interaction",
                    chat_id=chat_id,
                    text_length=len(text),
                )
                return ""

        start = time.monotonic()
        logger.info(
            "request_started",
            user_id=user_id,
            chat_id=chat_id,
            text_length=len(text),
        )

        await self.event_bus.emit(
            Event(
                name=MESSAGE_IN,
                data={"user_id": user_id, "text": text, "chat_id": chat_id},
            )
        )

        await self._log_message(
            user_id=user_id,
            chat_id=chat_id,
            role="user",
            content=text,
        )

        session = await self.session_manager.get_or_create(
            user_id, chat_id, self._default_directory
        )

        responder = None
        on_text_chunk = None
        on_tool_activity = None
        if self.connector and self.config.streaming_enabled:
            responder = _StreamingResponder(
                self.connector,
                chat_id,
                throttle_seconds=self.config.streaming_throttle_seconds,
            )
            on_text_chunk = responder.on_chunk
            on_tool_activity = responder.on_activity

        can_use_tool, tool_state = self._build_can_use_tool(session, chat_id, responder)
        self._tool_state = tool_state

        try:
            response = await self.agent.execute(
                prompt=text,
                session=session,
                can_use_tool=can_use_tool,
                on_text_chunk=on_text_chunk,
                on_tool_activity=on_tool_activity,
            )
            await self.session_manager.update_from_result(
                session,
                claude_session_id=response.session_id,
                cost=response.cost,
            )

            duration_ms = round((time.monotonic() - start) * 1000)
            await self._log_message(
                user_id=user_id,
                chat_id=chat_id,
                role="assistant",
                content=response.content,
                cost=response.cost,
                duration_ms=duration_ms,
                session_id=response.session_id,
            )

            streamed = False
            if responder:
                try:
                    streamed = await responder.finalize(response.content)
                except Exception:
                    logger.exception("streaming_finalize_failed")

            if not streamed and self.connector:
                await self.connector.send_message(chat_id, response.content)

            await self.event_bus.emit(
                Event(
                    name=MESSAGE_OUT,
                    data={"chat_id": chat_id, "content": response.content},
                )
            )

            logger.info(
                "request_completed",
                chat_id=chat_id,
                duration_ms=duration_ms,
                response_length=len(response.content),
                cost_usd=response.cost,
                num_turns=response.num_turns,
            )

            if tool_state.clean_proceed:
                plan = self._resolve_plan_content(tool_state, response.content)
                return await self._exit_plan_mode(
                    session,
                    chat_id,
                    user_id,
                    plan,
                    trigger="clean_proceed",
                    clear_context=True,
                    target_mode=tool_state.target_mode,
                )

            if (
                session.mode == "plan"
                and session.message_count > 1
                and not tool_state.plan_review_shown
                and tool_state.plan_file_path is not None
                and self.interaction_coordinator
                and self.connector
            ):
                fallback_content = self._resolve_plan_content(
                    tool_state,
                    response.content,
                )
                logger.info(
                    "fallback_plan_review_triggered",
                    content_length=len(fallback_content),
                    chat_id=chat_id,
                )
                from tether.core.interactions import PlanReviewDecision

                review = await self.interaction_coordinator.handle_plan_review(
                    chat_id,
                    {},
                    plan_content=fallback_content.strip() or None,
                )
                if isinstance(review, PlanReviewDecision):
                    return await self._exit_plan_mode(
                        session,
                        chat_id,
                        user_id,
                        fallback_content,
                        trigger=(
                            "fallback_clean_proceed"
                            if review.clear_context
                            else "fallback_allow"
                        ),
                        clear_context=review.clear_context,
                        target_mode=review.target_mode,
                    )
                if review.behavior == "deny":
                    return await self.handle_message(user_id, review.message, chat_id)

            return response.content

        except AgentError as e:
            duration_ms = round((time.monotonic() - start) * 1000)
            logger.error(
                "request_failed",
                error=str(e),
                user_id=user_id,
                chat_id=chat_id,
                duration_ms=duration_ms,
            )
            if self.approval_coordinator:
                self.approval_coordinator.cancel_pending(chat_id)
            if self.interaction_coordinator:
                self.interaction_coordinator.cancel_pending(chat_id)
            error_msg = f"Error: {e}"
            if self.connector:
                await self.connector.send_message(chat_id, error_msg)
            return error_msg

    async def _log_message(
        self,
        *,
        user_id: str,
        chat_id: str,
        role: str,
        content: str,
        cost: float | None = None,
        duration_ms: int | None = None,
        session_id: str | None = None,
    ) -> None:
        if not self._message_store:
            return
        try:
            await self._message_store.save_message(
                user_id=user_id,
                chat_id=chat_id,
                role=role,
                content=content,
                cost=cost,
                duration_ms=duration_ms,
                session_id=session_id,
            )
        except Exception:
            logger.exception("message_log_failed")

    async def _handle_with_middleware(
        self, user_id: str, text: str, chat_id: str
    ) -> str:
        ctx = MessageContext(user_id=user_id, chat_id=chat_id, text=text)
        return await self.middleware_chain.run(ctx, self.handle_message_ctx)  # type: ignore[union-attr]

    async def handle_message_ctx(self, ctx: MessageContext) -> str:
        """Adapter for middleware chain — delegates to handle_message."""
        return await self.handle_message(ctx.user_id, ctx.text, ctx.chat_id)

    async def handle_command(
        self,
        user_id: str,
        command: str,
        args: str,
        chat_id: str,
    ) -> str:
        logger.info(
            "command_received", user_id=user_id, chat_id=chat_id, command=command
        )

        session = await self.session_manager.get_or_create(
            user_id, chat_id, self._default_directory
        )

        if command == "dir":
            return self._handle_dir_command(session, args, chat_id)

        if command == "plan":
            old_mode = session.mode
            session.mode = "plan"
            self._gatekeeper.disable_auto_approve(chat_id)
            logger.info(
                "mode_switched",
                user_id=user_id,
                chat_id=chat_id,
                from_mode=old_mode,
                to_mode="plan",
            )
            return "Switched to plan mode. I'll create a plan before implementing."

        if command == "test":
            event = Event(
                name=COMMAND_TEST,
                data={
                    "session": session,
                    "chat_id": chat_id,
                    "args": args,
                    "gatekeeper": self._gatekeeper,
                    "prompt": "",
                },
            )
            await self.event_bus.emit(event)
            prompt = event.data.get("prompt", "")
            if prompt:
                await self.handle_message(user_id, prompt, chat_id)
            return ""

        if command == "edit":
            old_mode = session.mode
            session.mode = "auto"
            self._gatekeeper.enable_tool_auto_approve(chat_id, "Write")
            self._gatekeeper.enable_tool_auto_approve(chat_id, "Edit")
            self._gatekeeper.enable_tool_auto_approve(chat_id, "NotebookEdit")
            logger.info(
                "mode_switched",
                user_id=user_id,
                chat_id=chat_id,
                from_mode=old_mode,
                to_mode="auto",
            )
            return (
                "Accept edits on. I'll implement directly and auto-approve file edits."
            )

        if command == "default":
            old_mode = session.mode
            session.mode = "default"
            session.mode_instruction = None
            self._gatekeeper.disable_auto_approve(chat_id)
            logger.info(
                "mode_switched",
                user_id=user_id,
                chat_id=chat_id,
                from_mode=old_mode,
                to_mode="default",
            )
            return "Default mode. All file writes require per-call approval."

        if command == "clear":
            await self.session_manager.deactivate(user_id, chat_id)
            self._gatekeeper.disable_auto_approve(chat_id)
            logger.info("session_cleared", user_id=user_id, chat_id=chat_id)
            return "Session cleared. Next message starts a fresh conversation."

        if command == "status":
            mode = "accept edits" if session.mode == "auto" else session.mode
            cost = f"${session.total_cost:.4f}"
            blanket, per_tool = self._gatekeeper.get_auto_approve_status(chat_id)
            if blanket:
                auto_str = "on (all tools)"
            elif per_tool:
                auto_str = ", ".join(sorted(per_tool))
            else:
                auto_str = "off"
            active_name = self._active_dir_name(session)
            return (
                f"Mode: {mode}\n"
                f"Directory: {active_name}\n"
                f"Messages: {session.message_count}\n"
                f"Total cost: {cost}\n"
                f"Auto-approve: {auto_str}"
            )

        logger.warning(
            "unknown_command", user_id=user_id, chat_id=chat_id, command=command
        )
        return f"Unknown command: /{command}"

    def _active_dir_name(self, session: Session) -> str:
        wd = Path(session.working_directory)
        for name, path in self._dir_names.items():
            if path == wd:
                return name
        return wd.name

    def _handle_dir_command(self, session: Session, args: str, chat_id: str) -> str:
        if not args:
            lines = []
            for name, path in self._dir_names.items():
                marker = " (active)" if str(path) == session.working_directory else ""
                lines.append(f"  {name} → {path}{marker}")
            return "Directories:\n" + "\n".join(lines)

        target = args.strip()
        if target not in self._dir_names:
            available = ", ".join(self._dir_names)
            return f"Unknown directory: {target}\nAvailable: {available}"

        target_path = self._dir_names[target]
        if str(target_path) == session.working_directory:
            return f"Already in {target}."

        session.working_directory = str(target_path)
        session.claude_session_id = None
        self._gatekeeper.disable_auto_approve(chat_id)
        logger.info(
            "directory_switched",
            chat_id=chat_id,
            directory=str(target_path),
            name=target,
        )
        return f"Switched to {target} ({target_path})"

    def _resolve_plan_content(self, state: _ToolCallbackState, fallback: str) -> str:
        plan_path = state.plan_file_path
        if plan_path:
            try:
                content = Path(plan_path).read_text()
                logger.info(
                    "plan_content_resolved",
                    source="disk_file",
                    content_length=len(content),
                    plan_file_path=plan_path,
                )
                return content
            except Exception:
                logger.warning("plan_file_read_failed", path=plan_path)
        cached = state.plan_file_content
        if cached:
            logger.info(
                "plan_content_resolved",
                source="cached_write",
                content_length=len(cached),
                plan_file_path=plan_path,
            )
            return cached
        logger.info(
            "plan_content_resolved",
            source="fallback_response",
            content_length=len(fallback),
            plan_file_path=plan_path,
        )
        return fallback

    async def _exit_plan_mode(
        self,
        session: Session,
        chat_id: str,
        user_id: str,
        plan_content: str,
        trigger: str,
        *,
        clear_context: bool = False,
        target_mode: str = "edit",
    ) -> str:
        logger.info("plan_mode_exit", chat_id=chat_id, trigger=trigger)
        if clear_context:
            session.claude_session_id = None
        session.mode = "auto" if target_mode == "edit" else "default"
        if target_mode == "edit":
            self._gatekeeper.enable_tool_auto_approve(chat_id, "Write")
            self._gatekeeper.enable_tool_auto_approve(chat_id, "Edit")
        if clear_context and self.connector:
            await self.connector.send_message(
                chat_id, "Context cleared. Starting implementation..."
            )
        return await self.handle_message(
            user_id,
            self._build_implementation_prompt(plan_content),
            chat_id,
        )

    def _build_implementation_prompt(self, plan_content: str) -> str:
        content = plan_content.strip()
        if content and len(content) > 50:
            return f"Implement the following plan:\n\n{content}"
        return "Implement the plan."

    def _build_can_use_tool(
        self,
        session: Session,
        chat_id: str,
        responder: _StreamingResponder | None = None,
    ) -> tuple[Any, _ToolCallbackState]:
        state = _ToolCallbackState()

        async def can_use_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            _context: Any,
        ) -> Any:
            if self.interaction_coordinator and tool_name == "AskUserQuestion":
                return await self.interaction_coordinator.handle_question(
                    chat_id, tool_input
                )

            if tool_name in ("Write", "Edit"):
                file_path = tool_input.get("file_path", "")
                if file_path.endswith(".plan") or ".claude/plans/" in file_path:
                    state.plan_file_path = file_path
                    if tool_name == "Write":
                        state.plan_file_content = tool_input.get("content")

            if self.interaction_coordinator and tool_name == "ExitPlanMode":
                state.plan_review_shown = True
                plan_content = None
                content_source = "none"
                plan_path = state.plan_file_path
                if plan_path:
                    try:
                        plan_content = Path(plan_path).read_text()
                        content_source = "disk_file"
                    except Exception:
                        logger.warning("plan_file_read_failed", path=plan_path)
                if not plan_content:
                    plan_content = state.plan_file_content
                    if plan_content:
                        content_source = "cached_write"
                if not plan_content and responder:
                    buf = responder.buffer.strip()
                    if buf:
                        plan_content = buf
                        content_source = "streaming_buffer"
                logger.info(
                    "exit_plan_mode_content_resolved",
                    source=content_source,
                    content_length=len(plan_content) if plan_content else 0,
                    plan_file_path=plan_path,
                    has_cached_content=state.plan_file_content is not None,
                    has_streaming_buffer=bool(responder and responder.buffer.strip()),
                )
                from tether.core.interactions import PlanReviewDecision

                result = await self.interaction_coordinator.handle_plan_review(
                    chat_id, tool_input, plan_content=plan_content
                )
                if isinstance(result, PlanReviewDecision):
                    if result.clear_context:
                        session.claude_session_id = None
                        state.clean_proceed = True
                    state.target_mode = result.target_mode
                    if result.target_mode == "edit":
                        self._gatekeeper.enable_tool_auto_approve(chat_id, "Write")
                        self._gatekeeper.enable_tool_auto_approve(chat_id, "Edit")
                    if responder:
                        responder.reset()
                        await responder.deactivate()
                    if result.clear_context:
                        _bg: set[asyncio.Task[None]] = set()

                        async def _cancel_agent() -> None:
                            await asyncio.sleep(0.1)
                            await self.agent.cancel(session.session_id)

                        t = asyncio.create_task(_cancel_agent())
                        _bg.add(t)
                        t.add_done_callback(_bg.discard)
                    return result.permission
                return result

            return await self._gatekeeper.check(
                tool_name, tool_input, session.session_id, chat_id
            )

        return can_use_tool, state
