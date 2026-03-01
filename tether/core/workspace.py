"""Workspace model and YAML loader — groups related repos under a named workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger()


class Workspace(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    directories: list[Path]
    description: str = ""

    @property
    def primary_directory(self) -> Path:
        return self.directories[0]


def load_workspaces(tether_root: Path, approved: list[Path]) -> dict[str, Workspace]:
    """Load workspace definitions from `.tether/workspaces.yaml`.

    Returns empty dict if the file is missing (expected for users without workspaces).
    Skips individual workspaces that reference non-existent or unapproved directories.
    """
    tether_dir = tether_root / ".tether"
    yaml_path = _find_yaml(tether_dir)
    if yaml_path is None:
        return {}

    try:
        raw = yaml.safe_load(yaml_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "workspaces_yaml_read_failed", path=str(yaml_path), error=str(exc)
        )
        return {}

    if not isinstance(raw, dict):
        return {}

    return _parse_workspaces(raw.get("workspaces", {}), approved)


def _find_yaml(tether_dir: Path) -> Path | None:
    for ext in ("yaml", "yml"):
        p = tether_dir / f"workspaces.{ext}"
        if p.is_file():
            return p
    return None


def _parse_workspaces(
    raw: dict[str, Any] | None, approved: list[Path]
) -> dict[str, Workspace]:
    if not isinstance(raw, dict):
        return {}

    approved_set = {p.resolve() for p in approved}
    workspaces: dict[str, Workspace] = {}

    for name, entry in raw.items():
        if not isinstance(entry, dict):
            logger.warning("workspace_invalid_entry", workspace=name)
            continue

        raw_dirs = entry.get("directories", [])
        if not isinstance(raw_dirs, list) or not raw_dirs:
            logger.warning("workspace_no_directories", workspace=name)
            continue

        dirs: list[Path] = []
        for d in raw_dirs:
            resolved = Path(d).expanduser().resolve()
            if not resolved.is_dir():
                logger.warning(
                    "workspace_dir_not_found", workspace=name, directory=str(resolved)
                )
                continue
            if resolved not in approved_set:
                logger.warning(
                    "workspace_dir_not_approved",
                    workspace=name,
                    directory=str(resolved),
                )
                continue
            dirs.append(resolved)

        if not dirs:
            logger.warning("workspace_no_valid_directories", workspace=name)
            continue

        workspaces[name] = Workspace(
            name=name,
            directories=dirs,
            description=entry.get("description", ""),
        )

    if workspaces:
        logger.info("workspaces_loaded", count=len(workspaces), names=list(workspaces))

    return workspaces
