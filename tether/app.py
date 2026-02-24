"""Bootstrap: wires all components together."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from tether.agents.claude_code import ClaudeCodeAgent
from tether.core.config import TetherConfig
from tether.core.engine import Engine
from tether.core.events import EventBus
from tether.core.interactions import InteractionCoordinator
from tether.core.safety.approvals import ApprovalCoordinator
from tether.core.safety.audit import AuditLogger
from tether.core.safety.policy import PolicyEngine
from tether.core.safety.sandbox import SandboxEnforcer
from tether.core.session import SessionManager
from tether.git.handler import GitCommandHandler
from tether.git.service import GitService
from tether.middleware.auth import AuthMiddleware
from tether.middleware.base import MiddlewareChain
from tether.middleware.rate_limit import RateLimitMiddleware
from tether.plugins.builtin.audit_plugin import AuditPlugin
from tether.plugins.builtin.browser_tools import BrowserToolsPlugin
from tether.plugins.builtin.test_runner import TestRunnerPlugin
from tether.plugins.registry import PluginRegistry
from tether.storage.memory import MemorySessionStore
from tether.storage.sqlite import SqliteSessionStore

if TYPE_CHECKING:
    from tether.connectors.base import BaseConnector
    from tether.plugins.base import TetherPlugin
    from tether.storage.base import SessionStore

logger = structlog.get_logger()


def _configure_logging(config: TetherConfig) -> None:
    """Set up structlog with console output and optional rotating JSON file handler."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level)
    root_logger.handlers.clear()

    # Console handler — colored dev-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
        )
    )
    root_logger.addHandler(console_handler)

    # File handler — JSON lines for machine parsing
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_dir / "tether.log",
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )
        root_logger.addHandler(file_handler)

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def build_engine(
    config: TetherConfig | None = None,
    connector: BaseConnector | None = None,
    plugins: list[TetherPlugin] | None = None,
) -> Engine:
    if config is None:
        config = TetherConfig()  # type: ignore[call-arg]  # pydantic-settings loads from env

    _configure_logging(config)

    logger.info(
        "engine_building",
        storage_backend=config.storage_backend,
        has_connector=connector is not None,
        policy_count=len(config.policy_files),
        log_level=config.log_level,
        approved_directories=[str(d) for d in config.approved_directories],
    )

    # Storage
    store: SessionStore
    if config.storage_backend == "sqlite":
        store = SqliteSessionStore(config.storage_path)
    else:
        store = MemorySessionStore()

    session_manager = SessionManager(store=store)
    agent = ClaudeCodeAgent(config)
    event_bus = EventBus()

    # Safety components
    policy_paths = list(config.policy_files)
    if not policy_paths:
        default_policy = Path(__file__).parent / "policies" / "default.yaml"
        if default_policy.exists():
            policy_paths = [default_policy]

    policy_engine = PolicyEngine(policy_paths) if policy_paths else None
    sandbox = SandboxEnforcer(
        [*config.approved_directories, Path.home() / ".claude" / "plans"]
    )
    audit = AuditLogger(config.audit_log_path)

    approval_coordinator = None
    interaction_coordinator = None
    if connector:
        approval_coordinator = ApprovalCoordinator(connector, config)
        interaction_coordinator = InteractionCoordinator(connector, config, event_bus)

    # Plugins
    registry = PluginRegistry()
    registry.register(AuditPlugin(audit))
    registry.register(BrowserToolsPlugin())
    registry.register(TestRunnerPlugin())
    for plugin in plugins or []:
        registry.register(plugin)

    # Middleware
    middleware_chain = MiddlewareChain()
    if config.allowed_user_ids:
        middleware_chain.add(AuthMiddleware(config.allowed_user_ids))
    if config.rate_limit_rpm > 0:
        middleware_chain.add(
            RateLimitMiddleware(config.rate_limit_rpm, config.rate_limit_burst)
        )

    # Git command handler
    git_handler = None
    if connector:
        git_handler = GitCommandHandler(
            service=GitService(),
            connector=connector,
            sandbox=sandbox,
            audit=audit,
            event_bus=event_bus,
        )

    logger.info(
        "engine_built",
        has_auth=bool(config.allowed_user_ids),
        has_rate_limit=config.rate_limit_rpm > 0,
        plugin_count=len(registry.plugins),
        streaming=config.streaming_enabled,
    )

    return Engine(
        connector=connector,
        agent=agent,
        config=config,
        session_manager=session_manager,
        policy_engine=policy_engine,
        sandbox=sandbox,
        audit=audit,
        approval_coordinator=approval_coordinator,
        interaction_coordinator=interaction_coordinator,
        event_bus=event_bus,
        plugin_registry=registry,
        middleware_chain=middleware_chain,
        store=store,
        git_handler=git_handler,
    )
