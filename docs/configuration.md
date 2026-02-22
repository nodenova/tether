# Configuration Reference

`TetherConfig` (`core/config.py`) uses pydantic-settings to load configuration from environment variables prefixed with `TETHER_`. Values can also be set in a `.env` file in the project root.

## Loading Flow

```mermaid
flowchart LR
    env[".env file"]
    vars["Environment variables"]
    pydantic["pydantic-settings"]
    validators["Field validators"]
    config["TetherConfig"]
    build["build_engine()"]

    env --> pydantic
    vars --> pydantic
    pydantic --> validators --> config --> build
```

Environment variables override `.env` file values. Validators run after loading to resolve paths, parse user IDs, and validate policy files.

## Required Variables

| Variable | Type | Description |
|---|---|---|
| `TETHER_APPROVED_DIRECTORIES` | `list[Path]` | Directories the agent is allowed to operate in (comma-separated). All must exist. Expanded and resolved to absolute paths. First directory is the default working directory. |

## Full Reference

### Agent Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_MAX_TURNS` | `int` | `25` | Maximum agent turns per message |
| `TETHER_SYSTEM_PROMPT` | `str \| None` | `None` | Additional system prompt appended to the agent |
| `TETHER_ALLOWED_TOOLS` | `list[str]` | `[]` | Whitelist of tools the agent can use (empty = all) |
| `TETHER_DISALLOWED_TOOLS` | `list[str]` | `[]` | Blacklist of tools the agent cannot use |

### Safety Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_POLICY_FILES` | `list[Path]` | `[]` | Comma-separated paths to YAML policy files. Loaded in order; later files can override settings. |
| `TETHER_APPROVAL_TIMEOUT_SECONDS` | `int` | `300` | Seconds to wait for human approval before defaulting to deny |

### Authentication & Rate Limiting

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_ALLOWED_USER_IDS` | `set[str]` | `set()` | Comma-separated user IDs allowed to interact. Empty set = allow all. |
| `TETHER_RATE_LIMIT_RPM` | `int` | `0` | Requests per minute per user. `0` = no rate limiting. |
| `TETHER_RATE_LIMIT_BURST` | `int` | `5` | Maximum burst size for rate limiter token bucket |

### Connector

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_TELEGRAM_BOT_TOKEN` | `str \| None` | `None` | Telegram bot API token. If set, runs with Telegram connector. If unset, runs CLI REPL. |

### Storage

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_STORAGE_BACKEND` | `str` | `"memory"` | Storage backend: `"memory"` or `"sqlite"` |
| `TETHER_STORAGE_PATH` | `Path` | `tether.db` | Path to SQLite database file (only used when backend is `"sqlite"`) |

### Agent Mode

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_DEFAULT_MODE` | `str` | `"plan"` | Default session mode: `"plan"` (review before execute) or `"auto"` (implement directly) |

### Streaming

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_STREAMING_ENABLED` | `bool` | `True` | Enable real-time message streaming to connector |
| `TETHER_STREAMING_THROTTLE_SECONDS` | `float` | `1.5` | Minimum interval between streaming updates |

### Logging

| Variable | Type | Default | Description |
|---|---|---|---|
| `TETHER_LOG_LEVEL` | `str` | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `TETHER_AUDIT_LOG_PATH` | `Path` | `audit.jsonl` | Path to the audit JSONL log file |
| `TETHER_LOG_DIR` | `Path \| None` | `None` | Directory for rotating JSON log files. If unset, file logging is disabled. |
| `TETHER_LOG_MAX_BYTES` | `int` | `10485760` | Maximum size per log file (default 10 MB) |
| `TETHER_LOG_BACKUP_COUNT` | `int` | `5` | Number of rotated log file backups to keep |

## Validators

### `resolve_approved_directories`

Expands `~`, resolves each path to absolute, and verifies all directories exist. Raises `ConfigError` if any directory does not exist or the list is empty.

### `parse_approved_directories`

Accepts a comma-separated string, a single `Path`, or a `list[Path]`. Parses into `list[Path]`. Example: `"/path/a,/path/b"` becomes `[Path("/path/a"), Path("/path/b")]`.

### `parse_allowed_user_ids`

Accepts a comma-separated string, a single integer, or a set of strings. Parses into `set[str]`. Example: `"123,456"` becomes `{"123", "456"}`.

### `parse_policy_files`

Accepts a comma-separated string of file paths. Parses into `list[Path]`. Example: `"policies/default.yaml,policies/strict.yaml"`.

## Example `.env` Files

### Minimal

```env
TETHER_APPROVED_DIRECTORIES=/path/to/your/project
```

### Full

```env
# Required
TETHER_APPROVED_DIRECTORIES=/path/to/your/project

# Agent
TETHER_MAX_TURNS=25
TETHER_SYSTEM_PROMPT="Focus on writing tests first."
TETHER_DEFAULT_MODE=plan

# Safety
TETHER_POLICY_FILES=policies/default.yaml
TETHER_APPROVAL_TIMEOUT_SECONDS=300

# Auth
TETHER_ALLOWED_USER_IDS=123456789,987654321
TETHER_RATE_LIMIT_RPM=30
TETHER_RATE_LIMIT_BURST=5

# Connector
TETHER_TELEGRAM_BOT_TOKEN=your-bot-token-here

# Storage
TETHER_STORAGE_BACKEND=sqlite
TETHER_STORAGE_PATH=tether.db

# Streaming
TETHER_STREAMING_ENABLED=true
TETHER_STREAMING_THROTTLE_SECONDS=1.5

# Logging
TETHER_LOG_LEVEL=INFO
TETHER_AUDIT_LOG_PATH=audit.jsonl
TETHER_LOG_DIR=logs
TETHER_LOG_MAX_BYTES=10485760
TETHER_LOG_BACKUP_COUNT=5
```
