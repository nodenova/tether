"""Project test config loader — reads .tether/test.yaml from target project."""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger()


class ProjectTestConfig(BaseModel):
    """Project-level test defaults loaded from .tether/test.yaml."""

    model_config = ConfigDict(frozen=True)
    __test__ = False

    url: str | None = None
    server: str | None = None
    framework: str | None = None
    directory: str | None = None
    credentials: dict[str, str] = {}
    preconditions: list[str] = []
    focus_areas: list[str] = []
    environment: dict[str, str] = {}


def load_project_test_config(working_dir: str) -> ProjectTestConfig | None:
    """Read .tether/test.yaml or .tether/test.yml from the working directory.

    Returns ``None`` if the file is missing or invalid.
    """
    root = Path(working_dir)
    for name in ("test.yaml", "test.yml"):
        path = root / ".tether" / name
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text()) or {}
                config = ProjectTestConfig(**data)
                logger.info(
                    "project_test_config_loaded",
                    path=str(path),
                    url=config.url,
                    framework=config.framework,
                )
                return config
            except Exception:
                logger.warning(
                    "project_test_config_invalid",
                    path=str(path),
                    exc_info=True,
                )
                return None
    return None
