"""Claude Code agent — wraps the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message

from tether.agents.base import AgentResponse, BaseAgent, ToolActivity
from tether.exceptions import AgentError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from claude_agent_sdk import Message

    from tether.core.config import TetherConfig
    from tether.core.session import Session

logger = structlog.get_logger()


def _truncate(text: str, max_len: int = 60) -> str:
    """Collapse newlines and truncate with ellipsis."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1] + "\u2026"


_RETRYABLE_PATTERNS = ("api_error", "overloaded", "rate_limit", "529", "500")


def _is_retryable_error(content: str) -> bool:
    lowered = content.lower()
    return any(p in lowered for p in _RETRYABLE_PATTERNS)


def _describe_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Return a brief human-readable description of a tool call."""
    if name == "Bash":
        return _truncate(tool_input.get("command", ""))
    if name in ("Read", "Write", "Edit"):
        return str(tool_input.get("file_path", ""))
    if name == "Glob":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"{pattern} in {path}" if path else pattern
    if name == "Grep":
        pattern = tool_input.get("pattern", "")
        return f"/{pattern}/"
    if name == "WebFetch":
        return str(tool_input.get("url", ""))
    if name == "WebSearch":
        return str(tool_input.get("query", ""))
    if name in ("TodoWrite", "TaskCreate"):
        return _truncate(tool_input.get("subject", ""))
    if name == "TaskUpdate":
        task_id = tool_input.get("taskId", "")
        status = tool_input.get("status", "")
        if task_id and status:
            return f"#{task_id} → {status}"
        return f"#{task_id}" if task_id else ""
    if name == "TaskGet":
        return f"#{tool_input.get('taskId', '')}"
    if name == "TaskList":
        return "all tasks"
    # Unknown tool — show first string value
    for v in tool_input.values():
        if isinstance(v, str) and v:
            return _truncate(v)
    return ""


class _SafeSDKClient(ClaudeSDKClient):
    """SDK client that skips unknown message types instead of crashing."""

    async def receive_messages(self) -> AsyncIterator[Message]:
        if self._query is None:
            return
        async for data in self._query.receive_messages():
            try:
                yield parse_message(data)
            except MessageParseError:
                logger.debug(
                    "skipping_unknown_sdk_message",
                    message_type=data.get("type"),
                )


class ClaudeCodeAgent(BaseAgent):
    def __init__(self, config: TetherConfig) -> None:
        self._config = config
        self._active_clients: dict[str, ClaudeSDKClient] = {}

    async def execute(
        self,
        prompt: str,
        session: Session,
        *,
        can_use_tool: Callable[..., Any] | None = None,
        on_text_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_tool_activity: Callable[[ToolActivity | None], Coroutine[Any, Any, None]]
        | None = None,
    ) -> AgentResponse:
        options = self._build_options(session, can_use_tool)

        logger.info(
            "agent_execute_started",
            session_id=session.session_id,
            prompt_length=len(prompt),
            mode=session.mode,
            has_resume=session.claude_session_id is not None,
        )

        try:
            response = await self._run_with_resume(
                prompt,
                session,
                options,
                on_text_chunk=on_text_chunk,
                on_tool_activity=on_tool_activity,
            )
            if not response:
                return AgentResponse(content="No response from agent.", is_error=True)
            if response.is_error and _is_retryable_error(response.content):
                return AgentResponse(
                    content="The AI service is temporarily unavailable. Please try again in a moment.",
                    is_error=True,
                    session_id=response.session_id,
                    cost=response.cost,
                    duration_ms=response.duration_ms,
                    num_turns=response.num_turns,
                    tools_used=response.tools_used,
                )
            return response
        except Exception as e:
            logger.error(
                "agent_execute_failed", error=str(e), session=session.session_id
            )
            raise AgentError(f"Agent execution failed: {e}") from e

    _PLAN_MODE_INSTRUCTION = (
        "You are in plan mode. Before implementing, create a detailed plan first. "
        "Use EnterPlanMode to start planning, ask questions with AskUserQuestion "
        "when you need clarification. IMPORTANT: Before calling ExitPlanMode, you "
        "MUST write your complete plan to a file in .claude/plans/ using the Write "
        "tool (e.g., .claude/plans/plan.md). Then call ExitPlanMode so the user can "
        "review the plan. Always call ExitPlanMode before implementation begins — "
        "even if a plan already exists from a previous turn."
    )

    def _build_options(
        self,
        session: Session,
        can_use_tool: Callable[..., Any] | None,
    ) -> ClaudeAgentOptions:
        opts = ClaudeAgentOptions(
            cwd=session.working_directory,
            max_turns=self._config.max_turns,
            can_use_tool=can_use_tool,
            permission_mode=None,
            setting_sources=["project"],
        )
        system_prompt = self._config.system_prompt or ""
        if session.mode == "plan":
            system_prompt = (
                self._PLAN_MODE_INSTRUCTION + "\n\n" + system_prompt
                if system_prompt
                else self._PLAN_MODE_INSTRUCTION
            )
        if system_prompt:
            opts.system_prompt = system_prompt
        if self._config.allowed_tools:
            opts.allowed_tools = self._config.allowed_tools
        if self._config.disallowed_tools:
            opts.disallowed_tools = self._config.disallowed_tools
        if session.claude_session_id:
            opts.resume = session.claude_session_id
        return opts

    async def _run_with_resume(
        self,
        prompt: str,
        session: Session,
        options: ClaudeAgentOptions,
        *,
        on_text_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_tool_activity: Callable[[ToolActivity | None], Coroutine[Any, Any, None]]
        | None = None,
    ) -> AgentResponse:
        last_error: AgentResponse | None = None
        for _attempt in range(3):
            start = time.monotonic()
            tools_used: list[str] = []
            text_parts: list[str] = []

            async with _SafeSDKClient(options) as client:
                self._active_clients[session.session_id] = client

                try:
                    await client.query(prompt)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    text_parts.append(block.text)
                                    if on_text_chunk:
                                        try:
                                            await on_text_chunk(block.text)
                                        except Exception:
                                            logger.debug("on_text_chunk_error")
                                elif isinstance(block, ToolUseBlock):
                                    tools_used.append(block.name)
                                    if on_tool_activity:
                                        try:
                                            activity = ToolActivity(
                                                tool_name=block.name,
                                                description=_describe_tool(
                                                    block.name, block.input or {}
                                                ),
                                            )
                                            await on_tool_activity(activity)
                                        except Exception:
                                            logger.debug("on_tool_activity_error")
                                elif isinstance(block, ToolResultBlock):
                                    if on_tool_activity:
                                        try:
                                            await on_tool_activity(None)
                                        except Exception:
                                            logger.debug("on_tool_activity_error")

                        elif isinstance(message, ResultMessage):
                            duration = int((time.monotonic() - start) * 1000)

                            if message.num_turns == 0 and options.resume:
                                logger.info(
                                    "resume_zero_turns_retry",
                                    session=session.session_id,
                                )
                                options.resume = None
                                session.claude_session_id = None
                                break

                            content = message.result or "\n".join(text_parts)

                            if message.is_error and _is_retryable_error(content):
                                logger.warning(
                                    "retryable_api_error",
                                    session_id=session.session_id,
                                    error_preview=content[:200],
                                )
                                last_error = AgentResponse(
                                    content=content,
                                    session_id=message.session_id,
                                    cost=message.total_cost_usd or 0.0,
                                    duration_ms=duration,
                                    num_turns=message.num_turns,
                                    tools_used=tools_used,
                                    is_error=True,
                                )
                                await asyncio.sleep(2)
                                break
                            logger.info(
                                "agent_execute_completed",
                                session_id=session.session_id,
                                duration_ms=duration,
                                num_turns=message.num_turns,
                                cost_usd=message.total_cost_usd or 0.0,
                                tools_used_count=len(tools_used),
                                content_length=len(content),
                                is_error=message.is_error,
                            )
                            return AgentResponse(
                                content=content,
                                session_id=message.session_id,
                                cost=message.total_cost_usd or 0.0,
                                duration_ms=duration,
                                num_turns=message.num_turns,
                                tools_used=tools_used,
                                is_error=message.is_error,
                            )
                finally:
                    self._active_clients.pop(session.session_id, None)

        if last_error:
            return last_error
        logger.warning("agent_execute_no_response", session_id=session.session_id)
        return AgentResponse(content="No response received.", is_error=True)

    async def cancel(self, session_id: str) -> None:
        client = self._active_clients.get(session_id)
        if client:
            await client.interrupt()

    async def shutdown(self) -> None:
        for client in list(self._active_clients.values()):
            with contextlib.suppress(Exception):
                await client.disconnect()
        self._active_clients.clear()
