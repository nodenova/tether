"""YAML-driven policy engine — loads rules and classifies tool calls."""

import re
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger()


class PolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    action: PolicyDecision
    tools: list[str] = Field(default_factory=list)
    command_patterns: list[re.Pattern[str]] = Field(default_factory=list)
    path_patterns: list[re.Pattern[str]] = Field(default_factory=list)
    reason: str | None = None
    description: str | None = None
    risk_level: str = "medium"


class Classification(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: str
    tool_name: str
    tool_input: dict[str, Any]
    risk_level: str = "medium"
    description: str = ""
    deny_reason: str | None = None
    matched_rule: PolicyRule | None = None


class PolicyEngine:
    def __init__(self, policy_paths: list[Path] | None = None) -> None:
        self.rules: list[PolicyRule] = []
        self.settings: dict[str, Any] = {
            "default_action": "require_approval",
            "approval_timeout_seconds": 300,
        }
        if policy_paths:
            for path in policy_paths:
                self._load_policy(path)
            logger.info(
                "policy_engine_initialized",
                total_rules=len(self.rules),
                policy_count=len(policy_paths),
                default_action=self.settings.get("default_action"),
            )

    def _load_policy(self, path: Path) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            return

        if "settings" in data:
            self.settings.update(data["settings"])

        rules_data = data.get("rules", [])
        for rule_data in rules_data:
            self.rules.append(self._parse_rule(rule_data))
        logger.debug("policy_loaded", path=str(path), rule_count=len(rules_data))

    def _parse_rule(self, data: dict[str, Any]) -> PolicyRule:
        # Normalize tools: accept both "tool" (single) and "tools" (list)
        tools: list[str] = []
        if "tools" in data:
            tools = (
                data["tools"] if isinstance(data["tools"], list) else [data["tools"]]
            )
        elif "tool" in data:
            tools = [data["tool"]]

        # Compile regex patterns
        command_patterns = [re.compile(p) for p in data.get("command_patterns", [])]
        path_patterns = [re.compile(p) for p in data.get("path_patterns", [])]

        action_str = data["action"]
        action = PolicyDecision(action_str)

        return PolicyRule(
            name=data["name"],
            action=action,
            tools=tools,
            command_patterns=command_patterns,
            path_patterns=path_patterns,
            reason=data.get("reason"),
            description=data.get("description"),
            risk_level=data.get("risk_level", "medium"),
        )

    def classify(self, tool_name: str, tool_input: dict[str, Any]) -> Classification:
        for rule in self.rules:
            if self._rule_matches(rule, tool_name, tool_input):
                return Classification(
                    category=rule.name,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    risk_level=rule.risk_level,
                    description=rule.description or rule.reason or rule.name,
                    deny_reason=rule.reason,
                    matched_rule=rule,
                )

        # No rule matched — use default
        return Classification(
            category="unmatched",
            tool_name=tool_name,
            tool_input=tool_input,
            risk_level="medium",
            description=f"Unmatched tool call: {tool_name}",
        )

    def evaluate(self, classification: Classification) -> PolicyDecision:
        if classification.matched_rule:
            return classification.matched_rule.action

        # Default action from settings
        default = self.settings.get("default_action", "require_approval")
        return PolicyDecision(default)

    def _rule_matches(
        self,
        rule: PolicyRule,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> bool:
        # Tool name must match if tools are specified
        if rule.tools and tool_name not in rule.tools:
            return False

        # If rule has no tools, it won't match anything (rules must specify tools)
        if not rule.tools:
            return False

        # If rule has command_patterns, the tool must be Bash and command must match
        if rule.command_patterns:
            if tool_name != "Bash":
                return False
            command = tool_input.get("command", "")
            if not any(p.search(command) for p in rule.command_patterns):
                return False

        # If rule has path_patterns, check file_path or path in input
        if rule.path_patterns:
            path = tool_input.get("file_path") or tool_input.get("path") or ""
            if not any(p.search(path) for p in rule.path_patterns):
                return False

        return True
