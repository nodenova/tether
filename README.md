# Tether

**Drive Claude Code from your phone via Telegram, with safety guardrails.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage 89%+](https://img.shields.io/badge/coverage-89%25%2B-brightgreen.svg)](#development)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#)

---

Tether lets you send natural-language coding instructions from Telegram on your phone to a Claude Code agent running on your dev machine. A three-layer safety pipeline — sandbox, policy rules, and human approval — keeps the AI from doing anything dangerous without your explicit sign-off.

## What's New in 0.2.0

- **Git from your phone** — run `/git status`, `/git commit`, `/git push` and more directly from Telegram with inline action buttons
- **Smart git workflows** — auto-generated commit messages, fuzzy branch search, interactive file staging
- **Full audit trail** — every git operation is logged with session context

See [CHANGELOG.md](CHANGELOG.md) for full details.

## Features

- **Remote AI coding from any device** — send instructions via Telegram, Claude writes the code on your machine
- **Three-layer safety pipeline** — sandbox boundaries, YAML policy rules, and human approval on every risky action
- **YAML-driven policy rules** — 3 built-in presets (default, strict, permissive) or write your own
- **Inline approve/reject buttons** — tap to approve or block actions right from your phone
- **Multi-turn sessions** — Claude remembers the full conversation, so you can iterate naturally
- **Audit logging** — every tool attempt and decision is logged to an append-only JSONL file
- **Streaming responses** — see Claude's output live as it types, with real-time tool activity indicators
- **Message history** — with SQLite storage, every message is persisted with cost and duration metadata
- **Pluggable architecture** — connectors, middleware, and plugins are all extensible

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Safety](#safety)
- [Session Persistence](#session-persistence)
- [Browser Testing](#browser-testing)
- [Streaming](#streaming)
- [CLI Mode](#cli-mode)
- [Architecture](#architecture)
- [Development](#development)
- [Status](#status)

## Quick Start

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** — installed and authenticated. The `claude` command must work in your terminal. Tether delegates to it via `claude-agent-sdk`.
- **Telegram account** — to create a bot and chat with it from your phone
- **Node.js 18+** *(optional)* — required only for [browser testing](#browser-testing) via Playwright MCP

### 1. Clone and install

```bash
git clone git@github.com:nodenova/tether.git && cd tether
uv sync
```

### 2. Create a Telegram bot

1. Open Telegram on your phone
2. Search for **@BotFather** and start a chat
3. Send `/newbot`
4. Pick a display name (e.g., "My Tether Bot")
5. Pick a username ending in `bot` (e.g., `my_tether_bot`)
6. BotFather replies with a **token** like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11` — copy it

### 3. Find your Telegram user ID

Message **@userinfobot** on Telegram. It replies instantly with your numeric user ID (e.g., `981234567`). You'll use this to restrict the bot to only you.

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` and set these three values:

```bash
# The project directory you want the AI to work on (must exist)
TETHER_APPROVED_DIRECTORY=/path/to/your/project

# Bot token from step 2
TETHER_TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11

# Your Telegram user ID from step 3 (restricts access to only you)
TETHER_ALLOWED_USER_IDS=981234567
```

### 5. Run

```bash
uv run -m tether
```

### 6. Start coding from your phone

Open Telegram, find your bot, and send a message like:

> "Add a health check endpoint to the FastAPI app"

Claude will start working. When it needs to do something gated by policy (like writing a file), you'll see an approval request with **Approve** / **Reject** buttons right in the chat.

## How It Works

You type a message in Telegram on your phone. Tether receives it, runs it through the safety pipeline, and forwards it to a Claude Code agent session scoped to your project directory. Claude reads files, writes code, runs tests — whatever you asked. Each tool call is checked against sandbox boundaries, YAML policy rules, and (when required) sent back to you as an inline Approve/Reject button. The response flows back to your Telegram chat.

Sessions are multi-turn: Claude remembers the full conversation context across messages, so you can iterate naturally ("now add tests for that", "rename it to X").

## Configuration

All settings are environment variables prefixed with `TETHER_`. Set them in `.env` or export them directly.

The three essential variables are covered in [Quick Start](#4-configure). Here's the full reference:

<details>
<summary><strong>All configuration options</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `TETHER_APPROVED_DIRECTORY` | **required** | Directory the AI agent can work in. Must exist. |
| `TETHER_TELEGRAM_BOT_TOKEN` | — | Bot token from @BotFather. Without this, Tether runs in local CLI mode instead. |
| `TETHER_ALLOWED_USER_IDS` | *(no restriction)* | Comma-separated Telegram user IDs that can use the bot. Empty = anyone can use it. |
| `TETHER_MAX_TURNS` | `25` | Max conversation turns per request. |
| `TETHER_SYSTEM_PROMPT` | — | Custom system prompt for the agent. |
| `TETHER_POLICY_FILES` | built-in `default.yaml` | Comma-separated paths to YAML policy files. |
| `TETHER_APPROVAL_TIMEOUT_SECONDS` | `300` | Seconds to wait for your approval tap before auto-denying. |
| `TETHER_RATE_LIMIT_RPM` | `0` *(off)* | Max requests per minute per user. |
| `TETHER_RATE_LIMIT_BURST` | `5` | Burst capacity for the rate limiter. |
| `TETHER_STORAGE_BACKEND` | `memory` | `memory` (sessions lost on restart) or `sqlite` (persistent). |
| `TETHER_STORAGE_PATH` | `tether.db` | SQLite database path. Only used when backend is `sqlite`. |
| `TETHER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `TETHER_AUDIT_LOG_PATH` | `audit.jsonl` | Path for the append-only audit log of all tool decisions. |
| `TETHER_ALLOWED_TOOLS` | *(all)* | Allowlist of Claude tool names. Empty = all tools allowed. |
| `TETHER_DISALLOWED_TOOLS` | *(none)* | Denylist of Claude tool names. |
| `TETHER_STREAMING_ENABLED` | `true` | Progressive streaming updates in Telegram. |
| `TETHER_STREAMING_THROTTLE_SECONDS` | `1.5` | Min seconds between message edits during streaming. |
| `TETHER_LOG_DIR` | *(disabled)* | Directory for rotating JSON file logging (`{dir}/tether.log`). |
| `TETHER_LOG_MAX_BYTES` | `10485760` | Max log file size before rotation. Used with `LOG_DIR`. |
| `TETHER_LOG_BACKUP_COUNT` | `5` | Rotated log backups to keep. Used with `LOG_DIR`. |

</details>

## Safety

Every tool call Claude makes passes through a three-layer pipeline before it can execute:

**1. Sandbox** — The agent can only touch files inside `TETHER_APPROVED_DIRECTORY`. Any path traversal attempt is blocked immediately and logged as a security violation.

**2. Policy rules** — YAML rules classify each tool call as `allow`, `deny`, or `require_approval` based on the tool name, command patterns, and file path patterns. Rules are evaluated in order; first match wins.

**3. Human approval** — For `require_approval` actions, Tether sends you an inline message on Telegram with **Approve** and **Reject** buttons. If you don't respond within the timeout (default: 5 minutes), the action is denied automatically. Safe by default.

Everything is logged to `audit.jsonl` — every tool attempt, every decision.

### Built-in policies

Tether ships with three policies in `policies/`:

**`default.yaml`** (recommended) — Good starting point for phone-based vibe coding.
- Auto-allows: file reads, search, grep, git status/log/diff, readonly browser tools (snapshots, screenshots)
- Requires approval: file writes, edits, git push/rebase/merge, network commands, browser mutations (click, navigate, type)
- Hard-blocks: credential file access, `rm -rf`, `sudo`, force push, pipe-to-shell, SQL DROP/TRUNCATE

**`strict.yaml`** — Maximum safety, but chatty (lots of approval taps).
- Auto-allows: only file reads (`Read`, `Glob`, `Grep`, `LS`)
- Requires approval: everything else, including writes, bash, web tools, and all browser tools
- 2-minute approval timeout (vs 5 minutes for default)

**`permissive.yaml`** — For trusted environments where you want minimal interruptions.
- Auto-allows: reads, writes, package managers (`npm`, `pip`, `uv`, `cargo`), test runners (`pytest`, `jest`), `git add/commit/stash`, all browser tools
- Requires approval: git push, network commands, anything not explicitly listed
- 10-minute approval timeout

To switch policies:

```bash
TETHER_POLICY_FILES=policies/strict.yaml
```

You can also combine multiple policy files (rules are merged, evaluated in order):

```bash
TETHER_POLICY_FILES=policies/default.yaml,policies/my-overrides.yaml
```

## Session Persistence

By default, sessions are stored in memory and lost when Tether restarts. To persist sessions across restarts (so Claude remembers your conversation context), switch to SQLite:

```bash
TETHER_STORAGE_BACKEND=sqlite
TETHER_STORAGE_PATH=tether.db
```

With SQLite storage, Tether also logs every message (user and assistant) with cost, duration, and session metadata — giving you a queryable conversation history.

## Browser Testing

Tether integrates with [Playwright MCP](https://github.com/playwright-community/mcp) to give Claude browser automation capabilities — navigating pages, clicking elements, taking snapshots, and generating Playwright tests — all gated by the safety pipeline.

**Prerequisites:** Node.js 18+ and a one-time browser install:

```bash
npx playwright install chromium
```

**It's already configured.** The `.mcp.json` at the project root tells Claude Code to spawn the Playwright MCP server (pinned to `@playwright/mcp@0.0.41`, headless). The 25 browser tools are classified in all three policy presets — readonly tools (snapshots, screenshots) are auto-allowed in default policy, while mutation tools (click, navigate, type) require your approval.

**Typical workflow:**

1. Start your dev server (`npm run dev`, `uv run uvicorn`, etc.)
2. Launch Claude Code in the Tether project directory
3. Ask Claude to test your UI — "Navigate to localhost:3000 and verify the login form"
4. Claude uses browser tools, gated by your policy, and reports findings

Tether also includes Playwright test agents (Planner, Generator, Healer) and a `/healer` slash command for automated test repair.

See [docs/browser-testing.md](docs/browser-testing.md) for the full guide.

## Streaming

When connected via Telegram, responses stream in real-time — the message updates progressively as Claude types. While tools are running, you see a live indicator (e.g., `🔧 Bash: pytest tests/`). The final message includes a tool usage summary (e.g., `🧰 Bash x3, Read, Glob`). Disable with `TETHER_STREAMING_ENABLED=false`.

## Logging

Tether uses [structlog](https://www.structlog.org/) for structured logging. Every important code path — session lifecycle, agent execution, safety decisions, commands, and plugin activity — emits structured log events.

### Changing the log level

Set `TETHER_LOG_LEVEL` in your `.env` or environment:

```bash
# Show all logs including detailed tracing
TETHER_LOG_LEVEL=DEBUG

# Default — operational events only
TETHER_LOG_LEVEL=INFO

# Quieter — only warnings and errors
TETHER_LOG_LEVEL=WARNING
```

### Enabling file logging

By default, logs go only to the console. To also write JSON logs to a rotating file (useful for production debugging):

```bash
TETHER_LOG_DIR=logs
```

This creates `logs/tether.log` with automatic rotation. You can tune rotation with `TETHER_LOG_MAX_BYTES` (default: 10 MB) and `TETHER_LOG_BACKUP_COUNT` (default: 5 backups).

### Key log events

At `INFO` level you'll see the request lifecycle:

```
engine_building → engine_built → cli_starting → session_created →
request_started → agent_execute_started → agent_execute_completed →
request_completed → cli_shutting_down
```

At `DEBUG` level, additional events trace safety decisions (`policy_evaluated`, `sandbox_path_denied`), session cache behavior (`session_cache_hit`, `session_updated`), and interaction routing (`message_routed_to_interaction`, `resolve_text_answer`).

## CLI Mode

If you don't set `TETHER_TELEGRAM_BOT_TOKEN`, Tether runs as a local REPL in your terminal — useful for testing your configuration before going mobile. Note: actions that require approval are auto-denied in CLI mode since there's no approval UI.

```bash
# No TETHER_TELEGRAM_BOT_TOKEN set
uv run tether
# > type your prompts here
```

## Architecture

Tether's core is the **Engine**, which receives messages from connectors, runs them through a middleware chain (auth, rate limiting), routes them to the Claude Code agent, and sends responses back. Every tool call the agent makes is intercepted by the **Gatekeeper**, which orchestrates the three-layer safety pipeline: sandbox enforcement, YAML policy matching, and async human approval. An **EventBus** decouples subsystems — plugins subscribe to events like `tool.allowed`, `tool.denied`, and `approval.requested` to extend behavior without touching core code. Connectors (Telegram, with more planned) and storage backends (memory, SQLite) are swappable via protocol classes.

## Development

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_policy.py -v

# Run tests with coverage
uv run pytest --cov=tether tests/

# Lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format
uv run ruff format .
```

