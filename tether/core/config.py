"""Unified configuration via pydantic-settings."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def build_directory_names(directories: list[Path]) -> dict[str, Path]:
    """Map short names to paths using basenames, disambiguating conflicts with parent."""
    if not directories:
        return {}

    basenames: dict[str, list[Path]] = {}
    for d in directories:
        basenames.setdefault(d.name, []).append(d)

    names: dict[str, Path] = {}
    for basename, paths in basenames.items():
        if len(paths) == 1:
            names[basename] = paths[0]
        else:
            for p in paths:
                qualified = f"{p.parent.name}/{p.name}"
                names[qualified] = p
    return names


class TetherConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TETHER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Required
    approved_directories: list[Path]

    # Agent settings
    max_turns: int = 25
    system_prompt: str | None = None
    allowed_tools: list[str] = []
    disallowed_tools: list[str] = []

    # Safety settings
    policy_files: list[Path] = []
    approval_timeout_seconds: int = 300

    # Auth & rate limiting
    allowed_user_ids: set[str] = set()
    rate_limit_rpm: int = 0
    rate_limit_burst: int = 5

    # Connector
    telegram_bot_token: str | None = None

    # Storage
    storage_backend: str = "memory"
    storage_path: Path = Path("tether.db")

    # Agent mode
    default_mode: str = "default"  # "default", "plan", or "auto"

    # Streaming
    streaming_enabled: bool = True
    streaming_throttle_seconds: float = 1.5

    # Logging
    log_level: str = "INFO"
    audit_log_path: Path = Path("audit.jsonl")
    log_dir: Path | None = None
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5

    @field_validator("approved_directories", mode="before")
    @classmethod
    def parse_approved_directories(cls, v: list[Path] | str | Path) -> list[Path]:
        if isinstance(v, Path):
            return [v]
        if isinstance(v, str):
            return [Path(p.strip()) for p in v.split(",") if p.strip()]
        return v

    @field_validator("approved_directories")
    @classmethod
    def resolve_approved_directories(cls, v: list[Path]) -> list[Path]:
        if not v:
            raise ValueError("approved_directories must not be empty")
        resolved = []
        for p in v:
            r = p.expanduser().resolve()
            if not r.is_dir():
                raise ValueError(f"approved directory does not exist: {r}")
            resolved.append(r)
        return resolved

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, v: set[str] | str | int) -> set[str]:
        if isinstance(v, int):
            return {str(v)}
        if isinstance(v, str):
            return {s.strip() for s in v.split(",") if s.strip()}
        return v

    @field_validator("policy_files", mode="before")
    @classmethod
    def parse_policy_files(cls, v: list[Path] | str) -> list[Path]:
        if isinstance(v, str):
            return [Path(p.strip()) for p in v.split(",") if p.strip()]
        return v
