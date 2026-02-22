"""Audit plugin — demonstrates the plugin pattern by wrapping AuditLogger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tether.core.events import TOOL_DENIED
from tether.plugins.base import PluginMeta, TetherPlugin

if TYPE_CHECKING:
    from tether.core.events import Event
    from tether.core.safety.audit import AuditLogger
    from tether.plugins.base import PluginContext


class AuditPlugin(TetherPlugin):
    meta = PluginMeta(
        name="audit",
        version="0.1.0",
        description="Logs sandbox violations from TOOL_DENIED events",
    )

    def __init__(self, audit_logger: AuditLogger) -> None:
        self._audit = audit_logger

    async def initialize(self, context: PluginContext) -> None:
        context.event_bus.subscribe(TOOL_DENIED, self._on_tool_denied)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def _on_tool_denied(self, event: Event) -> None:
        if event.data.get("violation_type") != "sandbox":
            return
        self._audit.log_security_violation(
            session_id=event.data.get("session_id", "unknown"),
            tool_name=event.data.get("tool_name", "unknown"),
            reason=event.data.get("reason", ""),
            risk_level="critical",
        )
