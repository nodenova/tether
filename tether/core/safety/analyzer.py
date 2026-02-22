"""Bash command parser and path classifier."""

import re

from pydantic import BaseModel, Field


class CommandAnalysis(BaseModel):
    original: str
    commands: list[str]
    has_pipe: bool = False
    has_chain: bool = False
    has_sudo: bool = False
    has_subshell: bool = False
    has_redirect: bool = False
    risk_factors: list[str] = Field(default_factory=list)
    risk_level: str = "low"

    @property
    def is_compound(self) -> bool:
        return self.has_pipe or self.has_chain or self.has_subshell


class PathAnalysis(BaseModel):
    path: str
    operation: str
    is_credential: bool = False
    has_traversal: bool = False
    sensitivity: str = "normal"
    reason: str = ""


_CREDENTIAL_PATTERNS = [
    re.compile(r"\.env($|\.)"),
    re.compile(r"\.ssh/"),
    re.compile(r"\.aws/"),
    re.compile(r"\.gnupg/"),
    re.compile(r"\.key$"),
    re.compile(r"\.pem$"),
    re.compile(r"\.p12$"),
    re.compile(r"\.pfx$"),
    re.compile(r"id_rsa"),
    re.compile(r"id_ed25519"),
    re.compile(r"credentials"),
    re.compile(r"secrets?\."),
    re.compile(r"\.keystore$"),
    re.compile(r"token\.json$"),
]


class CommandAnalyzer:
    @staticmethod
    def analyze_bash(command: str) -> CommandAnalysis:
        analysis = CommandAnalysis(original=command, commands=[])
        risk_factors: list[str] = []

        # Detect structural features
        analysis.has_pipe = "|" in command
        analysis.has_chain = "&&" in command or "||" in command or ";" in command
        analysis.has_subshell = "$(" in command or "`" in command
        analysis.has_redirect = any(op in command for op in [">", ">>", "<"])
        analysis.has_sudo = bool(re.search(r"\bsudo\b", command))

        if analysis.has_sudo:
            risk_factors.append("uses sudo")
        if analysis.has_subshell:
            risk_factors.append("contains subshell")
        if analysis.has_pipe and analysis.has_redirect:
            risk_factors.append("pipe with redirect")

        # Split on pipes and chains to get individual commands
        parts = re.split(r"\s*[|;]\s*|\s*&&\s*|\s*\|\|\s*", command)
        for part in parts:
            part = part.strip()
            if part:
                analysis.commands.append(part)

        # Check for dangerous patterns
        if re.search(r"\brm\s.*-.*r.*f|\brm\s+-rf", command):
            risk_factors.append("recursive force delete")
        if re.search(r"\bchmod\s+777\b", command):
            risk_factors.append("world-writable permissions")
        if re.search(r"\b(curl|wget)\b.*\|\s*\b(bash|sh|zsh)\b", command):
            risk_factors.append("remote code execution via pipe")
        if re.search(r"\b(DROP|TRUNCATE)\s+(TABLE|DATABASE)\b", command, re.IGNORECASE):
            risk_factors.append("database destructive operation")

        analysis.risk_factors = risk_factors
        if len(risk_factors) >= 2:
            analysis.risk_level = "critical"
        elif risk_factors:
            analysis.risk_level = "high"
        elif analysis.is_compound:
            analysis.risk_level = "medium"
        else:
            analysis.risk_level = "low"

        return analysis


class PathAnalyzer:
    @staticmethod
    def analyze(path: str, operation: str = "read") -> PathAnalysis:
        analysis = PathAnalysis(path=path, operation=operation)

        if ".." in path:
            analysis.has_traversal = True
            analysis.sensitivity = "high"
            analysis.reason = "Path contains traversal components"

        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(path):
                analysis.is_credential = True
                analysis.sensitivity = "critical"
                analysis.reason = f"Matches credential pattern: {pattern.pattern}"
                break

        if operation in ("write", "edit") and analysis.sensitivity == "normal":
            analysis.sensitivity = "elevated"

        return analysis
