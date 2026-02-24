"""Shared exception types for Tether."""


class TetherError(Exception):
    """Base exception for all Tether errors."""


class ConfigError(TetherError):
    """Configuration is invalid or missing."""


class AgentError(TetherError):
    """Error from the AI agent backend."""


class SafetyError(TetherError):
    """A safety policy violation occurred."""


class ApprovalTimeoutError(TetherError):
    """User did not respond to an approval request in time."""


class SessionError(TetherError):
    """Session management error."""


class StorageError(TetherError):
    """Persistent storage error."""


class PluginError(TetherError):
    """Plugin lifecycle error."""


class InteractionTimeoutError(TetherError):
    """User did not respond to an interaction prompt in time."""


class ConnectorError(TetherError):
    """Connector failed after exhausting retries."""
