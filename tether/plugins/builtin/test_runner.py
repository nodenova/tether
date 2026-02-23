"""Test runner plugin — sets up test mode via event-driven command handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from tether.core.events import COMMAND_TEST
from tether.plugins.base import PluginMeta, TetherPlugin
from tether.plugins.builtin.browser_tools import BROWSER_MUTATION_TOOLS

if TYPE_CHECKING:
    from tether.core.events import Event
    from tether.plugins.base import PluginContext

logger = structlog.get_logger()

TEST_MODE_INSTRUCTION = (
    "You are in test mode. Your goal is to comprehensively verify that the "
    "code works correctly.\n\n"
    "Workflow:\n"
    "1. ANALYZE — Run git diff and git status to see what changed. Read the "
    "project structure to understand what test frameworks are available.\n"
    "2. UNIT TESTS — Run existing test suites first (pytest, jest, vitest, "
    "go test, cargo test, etc.). Report results.\n"
    "3. INTEGRATION — If integration tests exist, run them.\n"
    "4. E2E — If this is a web application with a UI:\n"
    "   a. Start the dev server if not already running\n"
    "   b. Navigate to the app with browser tools\n"
    "   c. Test key user flows by interacting with the UI\n"
    "   d. Use browser_snapshot to verify page state (prefer over screenshots "
    "for efficiency)\n"
    "   e. Take screenshots only for visual verification that matters\n"
    "5. REPORT — Provide a clear summary at the end:\n"
    "   - What was tested (unit, integration, E2E)\n"
    "   - Pass/fail counts per category\n"
    "   - Any failures with specifics (file, test name, error)\n"
    "   - Suggested fixes if obvious\n\n"
    "If a specific test focus was provided, prioritize that area but still run "
    "the full test suite for regression coverage.\n"
    "Always run the fastest tests first (unit before integration before E2E)."
)


class TestRunnerPlugin(TetherPlugin):
    meta = PluginMeta(
        name="test_runner",
        version="0.1.0",
        description="Event-driven test mode setup for /test command",
    )

    async def initialize(self, context: PluginContext) -> None:
        context.event_bus.subscribe(COMMAND_TEST, self._on_test_command)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def _on_test_command(self, event: Event) -> None:
        session = event.data["session"]
        session.mode = "test"
        session.mode_instruction = TEST_MODE_INSTRUCTION

        gatekeeper = event.data["gatekeeper"]
        chat_id = event.data["chat_id"]
        for tool in BROWSER_MUTATION_TOOLS:
            gatekeeper.enable_tool_auto_approve(chat_id, tool)

        args = event.data.get("args", "")
        event.data["prompt"] = (
            args.strip() or "Run comprehensive tests for the current codebase."
        )

        logger.info(
            "test_mode_activated",
            chat_id=chat_id,
            has_custom_args=bool(args.strip()),
        )
