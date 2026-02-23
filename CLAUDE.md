# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tether is a remote AI-assisted development system with safety constraints. It lets developers drive Claude Code agent sessions from any device (e.g., phone via Telegram) while enforcing YAML-driven safety policies that gate dangerous AI actions behind human approval.

## Commands

```bash
# Install dependencies
uv sync

# Run the CLI
uv run -m tether

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_policy.py -v

# Run a specific test
uv run pytest tests/test_policy.py::test_function_name -v

# Run tests with coverage
uv run pytest --cov=tether tests/

# Lint
uv run ruff check .

# Lint and auto-fix (removes unused imports, sorts imports, etc.)
uv run ruff check --fix .

# Format
uv run ruff format .

# Lint fix + format (equivalent to VS Code save)
uv run ruff check --fix . && uv run ruff format .
```

## Architecture

The system follows a three-layer safety pipeline: **Sandbox → Policy → Approval**.

**Engine** (`core/engine.py`) is the central orchestrator. It receives user messages from connectors, passes them through the middleware chain, routes messages to the Claude Code agent, and sends responses back through connectors. Supports `/dir` command to list and switch between approved directories (updates `session.working_directory`, resets agent context).

**Safety pipeline** (all in `core/safety/`):
0. **Gatekeeper** (`gatekeeper.py`) — `ToolGatekeeper` orchestrates the full sandbox → policy → approval chain per tool call, emitting events at each stage. Extracted from Engine to keep it focused on message routing.
1. **Sandbox** (`sandbox.py`) — enforces directory boundaries, prevents path traversal
2. **Policy** (`policy.py`) — stateless YAML rule matching that classifies tool calls as ALLOW, DENY, or REQUIRE_APPROVAL based on tool name, command patterns, and path patterns
3. **Approvals** (`approvals.py`) — async human-in-the-loop approval via connectors with configurable timeout (defaults to deny)
4. **Analyzer** (`analyzer.py`) — detects risky bash patterns and credential file access, used by the policy engine
5. **Audit** (`audit.py`) — append-only JSONL log of all tool attempts and decisions

**Middleware** (`middleware/`): `MiddlewareChain` processes messages before they reach the Engine. Each middleware can pass through or short-circuit.
- `AuthMiddleware` — user whitelist via `TETHER_ALLOWED_USER_IDS`
- `RateLimitMiddleware` — token-bucket rate limiting per user via `TETHER_RATE_LIMIT_RPM`

**EventBus** (`core/events.py`): Pub/sub system for decoupling subsystems. Plugins and internal components subscribe to named events. Key events: `tool.gated`, `tool.allowed`, `tool.denied`, `message.in`, `message.out`, `approval.requested`, `approval.resolved`, `safety.violation`, `engine.started`, `engine.stopped`.

**Plugin system** (`plugins/`):
- `TetherPlugin` ABC with lifecycle hooks: `initialize → start → stop`
- `PluginRegistry` for explicit registration (no auto-discovery)
- Plugins receive a `PluginContext` (event bus + config) and subscribe to `EventBus` events in `initialize()`
- Built-in: `AuditPlugin` logs sandbox violations from `tool.denied` events
- Built-in: `BrowserToolsPlugin` provides structured logging for the 25 Playwright MCP browser tools (classifies as readonly vs mutation, logs gated/allowed/denied events)

**Agent abstraction** (`agents/`): `BaseAgent` protocol with `ClaudeCodeAgent` implementation wrapping `claude-agent-sdk`. Supports session resume for multi-turn continuity.

**Connector protocol** (`connectors/base.py`): Abstract interface for I/O transports (Telegram, Slack, etc.). Handles message delivery, typing indicators, approval requests, and file sending.

**Policies** (`policies/`): Three built-in YAML policies — `default.yaml` (balanced), `strict.yaml` (maximum restrictions, shorter timeout), `permissive.yaml` (maximum freedom for trusted environments). All deny credential file access and destructive patterns.

**Configuration** (`core/config.py`): `TetherConfig` uses pydantic-settings, loaded from environment variables prefixed with `TETHER_`. Required: `TETHER_APPROVED_DIRECTORIES` (comma-separated paths). `build_directory_names()` derives short names from basenames for the `/dir` command.

**Storage** (`storage/`): `SessionStore` ABC with two backends — `MemorySessionStore` (in-process dict) and `SqliteSessionStore` (persistent via aiosqlite). Sessions are keyed by user+chat pair.

## Browser Testing (Playwright MCP)

Tether integrates with Playwright MCP for browser automation. The `.mcp.json` at project root configures Claude Code to spawn the MCP server (pinned `@playwright/mcp@0.0.41`, `--headless`). Tether's Python process does not touch Playwright — Claude Code's SDK manages the MCP server lifecycle.

- **Prerequisites:** Node.js 18+, one-time `npx playwright install chromium`
- **25 browser tools** (7 readonly, 18 mutation) flow through the existing safety pipeline — policy rules are defined in all three YAML presets (`default.yaml`, `strict.yaml`, `permissive.yaml`)
- **`BrowserToolsPlugin`** (`plugins/builtin/browser_tools.py`) provides structured logging; exports `BROWSER_READONLY_TOOLS`, `BROWSER_MUTATION_TOOLS`, `ALL_BROWSER_TOOLS`, `is_browser_tool()`
- **Playwright test agents:** `npx playwright init-agents --loop=claude` initializes Planner, Generator, and Healer agents
- **`/healer` slash command** at `.claude/commands/healer.md` runs the healer agent workflow to find and fix broken Playwright tests

## Code Conventions

- Python 3.13+ required
- **Always use `uv run` for all Python commands** — never use `python3`, `python`, or `python3 -m`. Examples: `uv run pytest`, `uv run ruff`, `uv run mypy`, `uv run tether`
- Async-first: all agent/connector operations use asyncio
- Ruff for linting and formatting (88-char line length, rules: E, F, I, N, W, UP, B, SIM, RUF, S, C4, PT, RET, ARG)
- Pydantic models for data validation, pydantic-settings for configuration
- structlog for structured logging — keyword args only, no string interpolation in log messages
- Protocol classes (`BaseAgent`, `BaseConnector`) define extensibility points
- Custom exception hierarchy in `exceptions.py`: `ConfigError`, `AgentError`, `SafetyError`, `ApprovalTimeoutError`, `SessionError`, `StorageError`, `PluginError`
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`; coverage minimum: 89% (`fail_under = 89`)
- No `__init__.py` or other boilerplate junk files — use implicit namespace packages
- No redundant or obvious comments — only comment non-obvious logic
- Only use `from __future__ import annotations` when necessary (e.g., forward references needed at runtime by Pydantic models)
- `TYPE_CHECKING` blocks to break circular imports — runtime imports only what's needed
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Composition over inheritance; prefer small collaborating objects
- Flat over nested: early returns, extract before exceeding 2 indentation levels
- `_` prefix for internal APIs, no `__` name mangling
- YAGNI: don't build for speculative future requirements
- Rule of Three: don't abstract until third duplication
- Test behavior, not implementation details
- After building a feature, write a short concise summary under `CHANGELOG.md` if applicable
