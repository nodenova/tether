"""Data models for git command results."""

from pydantic import BaseModel, ConfigDict


class FileChange(BaseModel):
    """A single file change in git status."""

    model_config = ConfigDict(frozen=True)

    path: str
    status: str  # "modified", "added", "deleted", "renamed", "untracked", "conflicted"


class GitStatus(BaseModel):
    """Parsed output of git status."""

    model_config = ConfigDict(frozen=True)

    branch: str
    tracking: str | None = None
    ahead: int = 0
    behind: int = 0
    staged: list[FileChange] = []
    unstaged: list[FileChange] = []
    untracked: list[str] = []


class GitBranch(BaseModel):
    """A git branch."""

    model_config = ConfigDict(frozen=True)

    name: str
    is_current: bool = False
    is_remote: bool = False


class GitLogEntry(BaseModel):
    """A single commit in git log."""

    model_config = ConfigDict(frozen=True)

    hash: str
    short_hash: str
    author: str
    date: str
    message: str


class GitResult(BaseModel):
    """Generic result from a git mutation operation."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    details: str = ""


class MergeResult(BaseModel):
    """Result from a git merge — distinguishes clean merge from conflicts."""

    model_config = ConfigDict(frozen=True)

    success: bool
    had_conflicts: bool = False
    conflicted_files: list[str] = []
    message: str
    details: str = ""
