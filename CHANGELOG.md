# Changelog

## [0.4.0] - 2026-02-27
- **added**: `/workspace` (alias `/ws`) — group related repos under named workspaces for multi-repo context. YAML config in `.tether/workspaces.yaml`, inline keyboard buttons, workspace-aware system prompt injection, and MCP server merging across workspace directories

## [0.3.0] - 2026-02-26

### Added
- `/git merge <branch>` — AI-assisted conflict resolution with auto-resolve/abort buttons and 4-phase merge workflow
- `/test` command — 9-phase agent-driven test workflow with structured args (`--url`, `--framework`, `--dir`, `--no-e2e`, `--no-unit`, `--no-backend`), project config (`.tether/test.yaml`), write-ahead crash recovery, and context persistence across sessions
- `/plan <text>` and `/edit <text>` — switch mode and start agent in one step
- `/dir` inline keyboard buttons for one-tap directory switching
- Message interrupt — inline buttons to interrupt or wait during agent execution instead of silent queuing
- `dev-tools.yaml` policy overlay — auto-allows common dev commands (package managers, linters, test runners)
- Auto-delete transient messages (interrupt prompts, ack messages, completion notices)

### Fixed
- Git callback buttons now auto-delete after action completes instead of persisting as stale UI
- Plan approval messages (content + buttons) now fully cleaned up after user decision, with brief ack for proceed actions
- **fixed**: Comprehensive edge case tests for git callback and plan approval cleanup (8 tests: exception resilience, non-Message guards, missing handlers, empty/missing plan IDs, dedup, ack failures, expired interactions)
- Agent resilience — exponential backoff on retries, auto-retry for transient API errors, 30-minute execution timeout, human-readable error messages
- Session continuity — `claude_session_id` persisted on agent timeout so next message resumes
- Pending messages no longer dropped on transient errors
- Playwright MCP tools now available when agent works in repos without their own `.mcp.json`
- **fixed**: Documentation audit — 30+ discrepancies corrected across README, config, plugins, engine, architecture, events, storage, and index docs (wrong defaults, missing features, stale counts, version mismatch)

## [0.2.1] - 2026-02-23
- **added**: Network resilience for Telegram connector — exponential-backoff retries on `NetworkError`/`TimedOut` for startup and send operations
- **fixed**: Streaming freezes on long responses — overflow now finalizes current message and chains into a new one instead of silently truncating at 4000 chars
- **fixed**: Sub-agent permission inheritance — map session modes to SDK `PermissionMode` so Task-spawned sub-agents can write/edit files in auto mode
- **added**: 3-word approval keys — `uv run pytest` and `uv run python` now get distinct auto-approve keys instead of both being `Bash::uv run`
- **added**: CD prefix stripping — `cd /project && git status` now matches `^git` policy patterns and produces `Bash::git status` approval keys instead of `Bash::cd`
- **added**: Hierarchical auto-approve matching — stored `Bash::uv run` covers `Bash::uv run pytest` with word-boundary safety


## [0.2.0] - 2026-02-23

### Added
- **Git integration** — full `/git` command suite accessible from Telegram with inline action buttons
  - `status`, `branch`, `checkout`, `diff`, `log`, `add`, `commit`, `push`, `pull`
  - Auto-generated commit messages from staged changes
  - Fuzzy branch matching with fallback to remote tracking
  - Interactive workflows: stage files, confirm pushes, enter commit messages via chat
- **Git service layer** (`tether/git/service.py`) — async wrapper around git CLI with 30s timeout, input validation against shell injection
- **Git data models** (`tether/git/models.py`) — frozen Pydantic models for status, branches, log entries, and results
- **Git display formatters** (`tether/git/formatter.py`) — Telegram-friendly formatting with emoji indicators and 4096-char truncation
- **Git callback routing** — inline button support in Telegram connector for git operations
- **Audit logging for git operations** — every git command logged with session context and working directory

### Changed
- **Engine** wired to route `/git` commands and git button callbacks to the new handler
- **App bootstrap** instantiates `GitService` and `GitCommandHandler` during startup
- **Connector base protocol** extended with `set_git_handler()` for registering git callbacks
- **Telegram connector** routes `git:` prefix callbacks to the git handler

## [0.1.0] - 2026-02-22

- Initial release
