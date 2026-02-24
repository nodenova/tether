# Changelog

## [0.2.1] - 2026-02-23
- **added**: Network resilience for Telegram connector — exponential-backoff retries on `NetworkError`/`TimedOut` for startup and send operations
- **fixed**: Streaming freezes on long responses — overflow now finalizes current message and chains into a new one instead of silently truncating at 4000 chars
- **fixed**: Sub-agent permission inheritance — map session modes to SDK `PermissionMode` so Task-spawned sub-agents can write/edit files in auto mode
- **added**: 3-word approval keys — `uv run pytest` and `uv run python` now get distinct auto-approve keys instead of both being `Bash::uv run`
- **added**: CD prefix stripping — `cd /project && git status` now matches `^git` policy patterns and produces `Bash::git status` approval keys instead of `Bash::cd`
- **added**: Hierarchical auto-approve matching — stored `Bash::uv run` covers `Bash::uv run pytest` with word-boundary safety


## [0.2.0] — 2026-02-23

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

## [0.1.0] — 2026-02-22

- Initial release
