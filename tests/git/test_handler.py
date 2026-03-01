"""Tests for GitCommandHandler with mocked service and connector."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from leashd.core.events import EventBus
from leashd.core.safety.audit import AuditLogger
from leashd.core.safety.sandbox import SandboxEnforcer
from leashd.core.session import Session
from leashd.git.formatter import build_auto_message
from leashd.git.handler import (
    GIT_CALLBACK_PREFIX,
    GitCommandHandler,
)
from leashd.git.models import (
    FileChange,
    GitBranch,
    GitLogEntry,
    GitResult,
    GitStatus,
)

# Re-use MockConnector from conftest
from tests.conftest import MockConnector

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_service():
    service = AsyncMock()
    service.is_repo = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_connector():
    return MockConnector()


@pytest.fixture
def sandbox(tmp_path):
    return SandboxEnforcer([tmp_path])


@pytest.fixture
def audit(tmp_path):
    return AuditLogger(tmp_path / "audit.jsonl")


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def session(tmp_path):
    return Session(
        session_id="test-session",
        user_id="user1",
        chat_id="chat1",
        working_directory=str(tmp_path),
    )


@pytest.fixture
def handler(mock_service, mock_connector, sandbox, audit, event_bus):
    return GitCommandHandler(
        service=mock_service,
        connector=mock_connector,
        sandbox=sandbox,
        audit=audit,
        event_bus=event_bus,
    )


# ── Subcommand routing tests ────────────────────────────────────────


class TestSubcommandRouting:
    async def test_empty_args_routes_to_status(self, handler, mock_service, session):
        mock_service.status.return_value = GitStatus(branch="main")
        await handler.handle_command("user1", "", "chat1", session)
        mock_service.status.assert_called_once()

    async def test_status_subcommand(self, handler, mock_service, session):
        mock_service.status.return_value = GitStatus(branch="main")
        await handler.handle_command("user1", "status", "chat1", session)
        mock_service.status.assert_called_once()

    async def test_branch_subcommand_no_args(self, handler, mock_service, session):
        mock_service.branches.return_value = [GitBranch(name="main")]
        await handler.handle_command("user1", "branch", "chat1", session)
        mock_service.branches.assert_called_once()

    async def test_branch_subcommand_with_query(self, handler, mock_service, session):
        mock_service.search_branches.return_value = []
        await handler.handle_command("user1", "branch feat", "chat1", session)
        mock_service.search_branches.assert_called_once()

    async def test_checkout_subcommand(self, handler, mock_service, session):
        mock_service.checkout.return_value = GitResult(success=True, message="Switched")
        await handler.handle_command("user1", "checkout main", "chat1", session)
        mock_service.checkout.assert_called_once()

    async def test_checkout_no_args(self, handler, session):
        result = await handler.handle_command("user1", "checkout", "chat1", session)
        assert "Usage" in result

    async def test_diff_subcommand(self, handler, mock_service, session):
        mock_service.diff.return_value = "diff output"
        await handler.handle_command("user1", "diff", "chat1", session)
        mock_service.diff.assert_called_once()

    async def test_log_subcommand(self, handler, mock_service, session):
        mock_service.log.return_value = []
        await handler.handle_command("user1", "log", "chat1", session)
        mock_service.log.assert_called_once()

    async def test_add_dot(self, handler, mock_service, session):
        mock_service.add_all.return_value = GitResult(success=True, message="Staged")
        await handler.handle_command("user1", "add .", "chat1", session)
        mock_service.add_all.assert_called_once()

    async def test_add_specific_file(self, handler, mock_service, session):
        mock_service.add.return_value = GitResult(success=True, message="Staged")
        await handler.handle_command("user1", "add src/app.py", "chat1", session)
        mock_service.add.assert_called_once()

    async def test_add_no_args_interactive(self, handler, mock_service, session):
        mock_service.status.return_value = GitStatus(
            branch="main",
            unstaged=[FileChange(path="file.py", status="modified")],
        )
        await handler.handle_command("user1", "add", "chat1", session)
        mock_service.status.assert_called_once()

    async def test_commit_with_message(self, handler, mock_service, session):
        mock_service.commit.return_value = GitResult(success=True, message="abc — msg")
        await handler.handle_command("user1", "commit fix: bug", "chat1", session)
        mock_service.commit.assert_called_once()

    async def test_push_subcommand(self, handler, mock_service, session):
        mock_service.status.return_value = GitStatus(
            branch="main", tracking="origin/main"
        )
        await handler.handle_command("user1", "push", "chat1", session)
        mock_service.status.assert_called_once()

    async def test_pull_subcommand(self, handler, mock_service, session):
        mock_service.pull.return_value = GitResult(success=True, message="Pull ok")
        await handler.handle_command("user1", "pull", "chat1", session)
        mock_service.pull.assert_called_once()

    async def test_help_subcommand(self, handler, session):
        result = await handler.handle_command("user1", "help", "chat1", session)
        assert "/git" in result
        assert "/git status" in result

    async def test_unknown_subcommand(self, handler, session):
        result = await handler.handle_command("user1", "stash", "chat1", session)
        assert "Unknown git subcommand" in result
        assert "stash" in result


# ── Sandbox enforcement ──────────────────────────────────────────────


class TestSandboxEnforcement:
    async def test_rejects_outside_approved_directory(
        self, mock_service, mock_connector, audit, event_bus
    ):
        sandbox = SandboxEnforcer([Path("/allowed/dir")])
        handler = GitCommandHandler(
            service=mock_service,
            connector=mock_connector,
            sandbox=sandbox,
            audit=audit,
            event_bus=event_bus,
        )
        session = Session(
            session_id="s1",
            user_id="u1",
            chat_id="c1",
            working_directory="/forbidden/path",
        )
        result = await handler.handle_command("u1", "status", "c1", session)
        assert "\u274c" in result
        assert "outside" in result.lower() or "Path" in result
        mock_service.is_repo.assert_not_called()

    async def test_callback_rejects_outside_sandbox(
        self, mock_service, mock_connector, audit, event_bus
    ):
        sandbox = SandboxEnforcer([Path("/allowed/dir")])
        handler = GitCommandHandler(
            service=mock_service,
            connector=mock_connector,
            sandbox=sandbox,
            audit=audit,
            event_bus=event_bus,
        )
        session = Session(
            session_id="s1",
            user_id="u1",
            chat_id="c1",
            working_directory="/forbidden/path",
        )
        await handler.handle_callback("u1", "c1", "checkout", "main", session)
        assert len(mock_connector.sent_messages) == 1
        assert "\u274c" in mock_connector.sent_messages[0]["text"]


# ── Not a git repo ───────────────────────────────────────────────────


class TestNotARepo:
    async def test_returns_error(self, handler, mock_service, session):
        mock_service.is_repo.return_value = False
        result = await handler.handle_command("user1", "status", "chat1", session)
        assert "Not a git repository" in result


# ── Status with buttons ──────────────────────────────────────────────


class TestStatusCommand:
    async def test_clean_status_no_buttons(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(branch="main")
        result = await handler.handle_command("user1", "status", "chat1", session)
        assert "main" in result
        assert len(mock_connector.sent_messages) == 0  # no buttons sent

    async def test_status_with_unstaged_shows_buttons(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            unstaged=[FileChange(path="app.py", status="modified")],
        )
        result = await handler.handle_command("user1", "", "chat1", session)
        assert result == ""  # sent via connector with buttons
        assert len(mock_connector.sent_messages) == 1
        msg = mock_connector.sent_messages[0]
        assert msg["buttons"] is not None
        button_texts = [b.text for row in msg["buttons"] for b in row]
        assert "Stage All" in button_texts
        assert "Diff" in button_texts

    async def test_status_with_staged_shows_commit_button(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="app.py", status="modified")],
        )
        await handler.handle_command("user1", "", "chat1", session)
        msg = mock_connector.sent_messages[0]
        button_texts = [b.text for row in msg["buttons"] for b in row]
        assert "Commit" in button_texts
        assert "Diff" in button_texts

    async def test_status_with_untracked_shows_stage_all(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            untracked=["new.txt"],
        )
        await handler.handle_command("user1", "", "chat1", session)
        msg = mock_connector.sent_messages[0]
        button_texts = [b.text for row in msg["buttons"] for b in row]
        assert "Stage All" in button_texts


# ── Branch list ──────────────────────────────────────────────────────


class TestBranchCommand:
    async def test_branches_with_buttons(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.branches.return_value = [
            GitBranch(name="main", is_current=True),
            GitBranch(name="develop"),
        ]
        result = await handler.handle_command("user1", "branch", "chat1", session)
        assert result == ""  # sent via connector
        msg = mock_connector.sent_messages[0]
        assert msg["buttons"] is not None
        button_data = [b.callback_data for row in msg["buttons"] for b in row]
        assert f"{GIT_CALLBACK_PREFIX}checkout:main" in button_data
        assert f"{GIT_CALLBACK_PREFIX}checkout:develop" in button_data

    async def test_no_branches_returns_text(self, handler, mock_service, session):
        mock_service.branches.return_value = []
        result = await handler.handle_command("user1", "branch", "chat1", session)
        assert "No branches found" in result

    async def test_max_branch_buttons_capped(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.branches.return_value = [
            GitBranch(name=f"branch-{i}") for i in range(20)
        ]
        await handler.handle_command("user1", "branch", "chat1", session)
        msg = mock_connector.sent_messages[0]
        button_count = sum(len(row) for row in msg["buttons"])
        assert button_count == 10


# ── Branch search ────────────────────────────────────────────────────


class TestBranchSearch:
    async def test_search_with_results(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.search_branches.return_value = [
            GitBranch(name="feature/auth"),
            GitBranch(name="feature/dashboard"),
        ]
        result = await handler.handle_command("user1", "branch feat", "chat1", session)
        assert result == ""
        msg = mock_connector.sent_messages[0]
        assert "feat" in msg["text"]

    async def test_search_no_results(self, handler, mock_service, session):
        mock_service.search_branches.return_value = []
        result = await handler.handle_command("user1", "branch zzz", "chat1", session)
        assert "No branches matching" in result

    async def test_search_strips_remote_prefix_from_buttons(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.search_branches.return_value = [
            GitBranch(name="remotes/origin/feature/auth", is_remote=True),
        ]
        await handler.handle_command("user1", "branch auth", "chat1", session)
        msg = mock_connector.sent_messages[0]
        btn = msg["buttons"][0][0]
        assert btn.text == "feature/auth"
        assert "checkout:feature/auth" in btn.callback_data


# ── Checkout ─────────────────────────────────────────────────────────


class TestCheckoutCommand:
    async def test_checkout_success(self, handler, mock_service, session):
        mock_service.checkout.return_value = GitResult(
            success=True, message="Switched to branch 'main'"
        )
        result = await handler.handle_command(
            "user1", "checkout main", "chat1", session
        )
        assert "Switched" in result

    async def test_checkout_fuzzy_fallback(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.checkout.return_value = GitResult(
            success=False, message="Failed to checkout 'payme'"
        )
        mock_service.search_branches.return_value = [
            GitBranch(name="feature/payments"),
            GitBranch(name="fix/payment-gateway"),
        ]
        result = await handler.handle_command(
            "user1", "checkout payme", "chat1", session
        )
        assert result == ""  # sent via connector with buttons
        msg = mock_connector.sent_messages[0]
        assert "Did you mean" in msg["text"]
        button_texts = [b.text for row in msg["buttons"] for b in row]
        assert "feature/payments" in button_texts
        assert "fix/payment-gateway" in button_texts

    async def test_checkout_no_fuzzy_matches(self, handler, mock_service, session):
        mock_service.checkout.return_value = GitResult(success=False, message="Failed")
        mock_service.search_branches.return_value = []
        result = await handler.handle_command("user1", "checkout zzz", "chat1", session)
        assert "Failed" in result

    async def test_checkout_strips_remote_prefix_in_fuzzy(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.checkout.return_value = GitResult(success=False, message="Fail")
        mock_service.search_branches.return_value = [
            GitBranch(name="remotes/origin/feature/new", is_remote=True),
        ]
        await handler.handle_command("user1", "checkout new", "chat1", session)
        msg = mock_connector.sent_messages[0]
        btn = msg["buttons"][0][0]
        assert btn.text == "feature/new"


# ── Diff ─────────────────────────────────────────────────────────────


class TestDiffCommand:
    async def test_diff_output(self, handler, mock_service, session):
        mock_service.diff.return_value = "--- a/f\n+++ b/f\n-old\n+new"
        result = await handler.handle_command("user1", "diff", "chat1", session)
        assert "+new" in result

    async def test_diff_empty(self, handler, mock_service, session):
        mock_service.diff.return_value = ""
        result = await handler.handle_command("user1", "diff", "chat1", session)
        assert "No changes" in result

    async def test_diff_staged(self, handler, mock_service, session):
        mock_service.diff.return_value = "staged diff"
        await handler.handle_command("user1", "diff --staged", "chat1", session)
        mock_service.diff.assert_called_once()
        call_kwargs = mock_service.diff.call_args
        assert call_kwargs[1]["staged"] is True


# ── Log ──────────────────────────────────────────────────────────────


class TestLogCommand:
    async def test_log_output(self, handler, mock_service, session):
        mock_service.log.return_value = [
            GitLogEntry(
                hash="abc", short_hash="abc", author="Alice", date="now", message="fix"
            )
        ]
        result = await handler.handle_command("user1", "log", "chat1", session)
        assert "abc" in result
        assert "Alice" in result

    async def test_log_empty(self, handler, mock_service, session):
        mock_service.log.return_value = []
        result = await handler.handle_command("user1", "log", "chat1", session)
        assert "No commits" in result


# ── Add ──────────────────────────────────────────────────────────────


class TestAddCommand:
    async def test_add_specific_file(self, handler, mock_service, session):
        mock_service.add.return_value = GitResult(
            success=True, message="Staged 1 file(s)"
        )
        result = await handler.handle_command(
            "user1", "add src/app.py", "chat1", session
        )
        assert "Staged" in result
        mock_service.add.assert_called_once()
        call_args = mock_service.add.call_args[0]
        assert "src/app.py" in call_args[1]

    async def test_add_multiple_files(self, handler, mock_service, session):
        mock_service.add.return_value = GitResult(
            success=True, message="Staged 2 file(s)"
        )
        await handler.handle_command("user1", "add a.py b.py", "chat1", session)
        call_args = mock_service.add.call_args[0]
        assert "a.py" in call_args[1]
        assert "b.py" in call_args[1]

    async def test_add_all(self, handler, mock_service, session):
        mock_service.add_all.return_value = GitResult(
            success=True, message="Staged all"
        )
        result = await handler.handle_command("user1", "add .", "chat1", session)
        assert "Staged" in result
        mock_service.add_all.assert_called_once()

    async def test_add_interactive_with_files(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            unstaged=[
                FileChange(path="a.py", status="modified"),
                FileChange(path="b.py", status="modified"),
            ],
            untracked=["c.txt"],
        )
        result = await handler.handle_command("user1", "add", "chat1", session)
        assert result == ""
        msg = mock_connector.sent_messages[0]
        assert "Unstaged files" in msg["text"]
        button_texts = [b.text for row in msg["buttons"] for b in row]
        assert "a.py" in button_texts
        assert "b.py" in button_texts
        assert "c.txt" in button_texts
        assert "Stage All" in button_texts

    async def test_add_interactive_no_files(self, handler, mock_service, session):
        mock_service.status.return_value = GitStatus(branch="main")
        result = await handler.handle_command("user1", "add", "chat1", session)
        assert "No unstaged files" in result

    async def test_add_interactive_caps_buttons(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            unstaged=[
                FileChange(path=f"file{i}.py", status="modified") for i in range(15)
            ],
        )
        await handler.handle_command("user1", "add", "chat1", session)
        msg = mock_connector.sent_messages[0]
        # 10 file buttons + 1 "Stage All" button = 11 rows
        assert len(msg["buttons"]) == 11


# ── Commit ───────────────────────────────────────────────────────────


class TestCommitCommand:
    async def test_commit_with_message(self, handler, mock_service, session):
        mock_service.commit.return_value = GitResult(
            success=True, message="abc — fix bug"
        )
        result = await handler.handle_command(
            "user1", "commit fix bug", "chat1", session
        )
        assert "fix bug" in result

    async def test_commit_failure(self, handler, mock_service, session):
        mock_service.commit.return_value = GitResult(
            success=False, message="Failed to commit", details="nothing staged"
        )
        result = await handler.handle_command("user1", "commit msg", "chat1", session)
        assert "Failed" in result

    async def test_commit_prompt_no_staged(self, handler, mock_service, session):
        mock_service.status.return_value = GitStatus(branch="main")
        result = await handler.handle_command("user1", "commit", "chat1", session)
        assert "No staged changes" in result

    async def test_commit_prompt_with_staged(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="app.py", status="modified")],
        )
        mock_service.commit.return_value = GitResult(success=True, message="abc — msg")

        # Start commit prompt in background
        task = asyncio.create_task(
            handler.handle_command("user1", "commit", "chat1", session)
        )

        # Wait for pending input to be registered
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)
        assert handler.has_pending_input("chat1")

        # Simulate user reply
        resolved = await handler.resolve_input("chat1", "my commit message")
        assert resolved is True

        result = await task
        assert "msg" in result or "my commit message" in result
        mock_service.commit.assert_called_once()

    async def test_commit_prompt_cleans_up_pending(
        self, handler, mock_service, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="a.py", status="modified")],
        )
        mock_service.commit.return_value = GitResult(success=True, message="done")

        task = asyncio.create_task(
            handler.handle_command("user1", "commit", "chat1", session)
        )
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)

        await handler.resolve_input("chat1", "msg")
        await task

        assert not handler.has_pending_input("chat1")


# ── Push ─────────────────────────────────────────────────────────────


class TestPushCommand:
    async def test_push_confirm_buttons(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main", tracking="origin/main", ahead=2
        )
        result = await handler.handle_command("user1", "push", "chat1", session)
        assert result == ""
        msg = mock_connector.sent_messages[0]
        assert "Push" in msg["text"]
        assert "2 commits ahead" in msg["text"]
        button_texts = [b.text for row in msg["buttons"] for b in row]
        assert "Push" in button_texts
        assert "Cancel" in button_texts

    async def test_push_no_tracking(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(branch="main")
        await handler.handle_command("user1", "push", "chat1", session)
        msg = mock_connector.sent_messages[0]
        assert "origin" in msg["text"]  # fallback to "origin"


# ── Pull ─────────────────────────────────────────────────────────────


class TestPullCommand:
    async def test_pull_success(self, handler, mock_service, session):
        mock_service.pull.return_value = GitResult(
            success=True, message="Pull successful", details="Already up to date."
        )
        result = await handler.handle_command("user1", "pull", "chat1", session)
        assert "Pull successful" in result

    async def test_pull_failure(self, handler, mock_service, session):
        mock_service.pull.return_value = GitResult(
            success=False, message="Pull failed", details="conflict"
        )
        result = await handler.handle_command("user1", "pull", "chat1", session)
        assert "Pull failed" in result


# ── Callback handling ────────────────────────────────────────────────


class TestCallbackHandling:
    async def test_checkout_callback(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.checkout.return_value = GitResult(
            success=True, message="Switched to branch 'develop'"
        )
        await handler.handle_callback("user1", "chat1", "checkout", "develop", session)
        mock_service.checkout.assert_called_once()
        assert len(mock_connector.sent_messages) == 1
        assert "Switched" in mock_connector.sent_messages[0]["text"]

    async def test_add_callback(self, handler, mock_service, mock_connector, session):
        mock_service.add.return_value = GitResult(
            success=True, message="Staged 1 file(s)"
        )
        await handler.handle_callback("user1", "chat1", "add", "src/app.py", session)
        mock_service.add.assert_called_once_with(
            Path(session.working_directory), ["src/app.py"]
        )
        assert "Staged" in mock_connector.sent_messages[0]["text"]

    async def test_add_all_callback(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.add_all.return_value = GitResult(
            success=True, message="Staged all"
        )
        await handler.handle_callback("user1", "chat1", "add_all", "", session)
        mock_service.add_all.assert_called_once()

    async def test_push_confirm_callback(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.push.return_value = GitResult(
            success=True, message="Push successful"
        )
        await handler.handle_callback("user1", "chat1", "push_confirm", "", session)
        mock_service.push.assert_called_once()
        assert "\U0001f680" in mock_connector.sent_messages[0]["text"]

    async def test_push_cancel_callback(
        self, handler, mock_service, mock_connector, session
    ):
        await handler.handle_callback("user1", "chat1", "push_cancel", "", session)
        assert "cancelled" in mock_connector.sent_messages[0]["text"].lower()

    async def test_status_callback(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(branch="main")
        await handler.handle_callback("user1", "chat1", "status", "", session)
        assert len(mock_connector.sent_messages) == 1

    async def test_diff_callback(self, handler, mock_service, mock_connector, session):
        mock_service.diff.return_value = "some diff"
        await handler.handle_callback("user1", "chat1", "diff", "", session)
        assert "some diff" in mock_connector.sent_messages[0]["text"]

    async def test_commit_prompt_callback_no_staged(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(branch="main")
        await handler.handle_callback("user1", "chat1", "commit_prompt", "", session)
        assert "No staged changes" in mock_connector.sent_messages[0]["text"]

    async def test_search_callback(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.search_branches.return_value = [GitBranch(name="feature/auth")]
        await handler.handle_callback("user1", "chat1", "search", "auth", session)
        # _search_branches sends with buttons, then callback sends the result
        assert len(mock_connector.sent_messages) >= 1
        # First message has the search results with buttons
        first = mock_connector.sent_messages[0]
        assert "auth" in first["text"]
        assert first["buttons"] is not None

    async def test_unknown_callback(self, handler, mock_connector, session):
        await handler.handle_callback(
            "user1", "chat1", "unknown_action", "payload", session
        )
        assert len(mock_connector.sent_messages) == 0


# ── Pending input ────────────────────────────────────────────────────


class TestPendingInput:
    def test_no_pending_initially(self, handler):
        assert handler.has_pending_input("chat1") is False

    async def test_resolve_no_pending(self, handler):
        result = await handler.resolve_input("chat1", "text")
        assert result is False

    async def test_resolve_sets_value_and_signals(self, handler):
        from leashd.git.handler import _PendingInput

        pending = _PendingInput(kind="commit")
        handler._pending["chat1"] = pending

        assert handler.has_pending_input("chat1") is True

        resolved = await handler.resolve_input("chat1", "my message")
        assert resolved is True
        assert pending.value == "my message"
        assert pending.event.is_set()


# ── Audit logging ────────────────────────────────────────────────────


class TestAuditLogging:
    async def test_checkout_logs_audit(self, handler, mock_service, session, tmp_path):
        mock_service.checkout.return_value = GitResult(success=True, message="Switched")
        await handler.handle_command("user1", "checkout main", "chat1", session)
        audit_path = tmp_path / "audit.jsonl"
        content = audit_path.read_text()
        assert "git_operation" in content
        assert "checkout" in content

    async def test_commit_logs_audit(self, handler, mock_service, session, tmp_path):
        mock_service.commit.return_value = GitResult(success=True, message="done")
        await handler.handle_command("user1", "commit msg", "chat1", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert "commit" in content

    async def test_add_logs_audit(self, handler, mock_service, session, tmp_path):
        mock_service.add.return_value = GitResult(success=True, message="staged")
        await handler.handle_command("user1", "add file.py", "chat1", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert "add" in content

    async def test_add_all_logs_audit(self, handler, mock_service, session, tmp_path):
        mock_service.add_all.return_value = GitResult(success=True, message="staged")
        await handler.handle_command("user1", "add .", "chat1", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert "add_all" in content

    async def test_push_callback_logs_audit(
        self, handler, mock_service, session, tmp_path
    ):
        mock_service.push.return_value = GitResult(success=True, message="pushed")
        await handler.handle_callback("user1", "chat1", "push_confirm", "", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert "push" in content

    async def test_pull_logs_audit(self, handler, mock_service, session, tmp_path):
        mock_service.pull.return_value = GitResult(success=True, message="pulled")
        await handler.handle_command("user1", "pull", "chat1", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert "pull" in content

    async def test_checkout_callback_logs_audit(
        self, handler, mock_service, session, tmp_path
    ):
        mock_service.checkout.return_value = GitResult(success=True, message="Switched")
        await handler.handle_callback("user1", "chat1", "checkout", "develop", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert "checkout" in content

    async def test_audit_contains_session_id(
        self, handler, mock_service, session, tmp_path
    ):
        mock_service.pull.return_value = GitResult(success=True, message="ok")
        await handler.handle_command("user1", "pull", "chat1", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert session.session_id in content

    async def test_audit_contains_working_directory(
        self, handler, mock_service, session, tmp_path
    ):
        mock_service.pull.return_value = GitResult(success=True, message="ok")
        await handler.handle_command("user1", "pull", "chat1", session)
        content = (tmp_path / "audit.jsonl").read_text()
        assert session.working_directory in content


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_diff_with_path_argument(self, handler, mock_service, session):
        mock_service.diff.return_value = "path diff"
        await handler.handle_command("user1", "diff src/app.py", "chat1", session)
        call_kwargs = mock_service.diff.call_args
        assert call_kwargs[1]["path"] == "src/app.py"

    async def test_status_with_only_staged_no_unstaged(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="a.py", status="added")],
        )
        await handler.handle_command("user1", "", "chat1", session)
        msg = mock_connector.sent_messages[0]
        button_texts = [b.text for row in msg["buttons"] for b in row]
        # Should have Diff and Commit but NOT Stage All
        assert "Stage All" not in button_texts
        assert "Diff" in button_texts
        assert "Commit" in button_texts

    async def test_add_callback_sends_result(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.add.return_value = GitResult(
            success=False, message="Failed to stage files", details="pathspec error"
        )
        await handler.handle_callback("user1", "chat1", "add", "bad.py", session)
        text = mock_connector.sent_messages[0]["text"]
        assert "Failed" in text

    async def test_commit_prompt_callback_with_staged(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="a.py", status="modified")],
        )
        mock_service.commit.return_value = GitResult(success=True, message="done")

        task = asyncio.create_task(
            handler.handle_callback("user1", "chat1", "commit_prompt", "", session)
        )
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)
        await handler.resolve_input("chat1", "commit msg")
        await task

        # Should have sent prompt message and then result is sent
        assert len(mock_connector.sent_messages) >= 2

    async def test_multiple_chats_independent_pending(self, handler):
        from leashd.git.handler import _PendingInput

        handler._pending["chat1"] = _PendingInput(kind="commit")
        handler._pending["chat2"] = _PendingInput(kind="commit")
        assert handler.has_pending_input("chat1")
        assert handler.has_pending_input("chat2")
        assert not handler.has_pending_input("chat3")

    async def test_commit_prompt_message_includes_staged_summary(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[
                FileChange(path="a.py", status="modified"),
                FileChange(path="b.py", status="added"),
            ],
        )
        mock_service.commit.return_value = GitResult(success=True, message="done")

        task = asyncio.create_task(
            handler.handle_command("user1", "commit", "chat1", session)
        )
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)

        # Check the prompt was sent with staged summary
        prompt_msg = mock_connector.sent_messages[0]
        assert "a.py" in prompt_msg["text"]
        assert "b.py" in prompt_msg["text"]
        assert "Reply with your commit message" in prompt_msg["text"]

        await handler.resolve_input("chat1", "msg")
        await task


# ── Auto-generated commit messages ───────────────────────────────


class TestBuildAutoMessage:
    def test_single_modified(self):
        staged = [FileChange(path="src/app.py", status="modified")]
        assert build_auto_message(staged) == "update src/app.py"

    def test_single_added(self):
        staged = [FileChange(path="new_file.py", status="added")]
        assert build_auto_message(staged) == "add new_file.py"

    def test_single_deleted(self):
        staged = [FileChange(path="old_file.py", status="deleted")]
        assert build_auto_message(staged) == "delete old_file.py"

    def test_single_renamed(self):
        staged = [FileChange(path="config.py", status="renamed")]
        assert build_auto_message(staged) == "rename config.py"

    def test_multiple_same_status(self):
        staged = [
            FileChange(path="a.py", status="modified"),
            FileChange(path="b.py", status="modified"),
            FileChange(path="c.py", status="modified"),
        ]
        assert build_auto_message(staged) == "update 3 files"

    def test_multiple_mixed(self):
        staged = [
            FileChange(path="a.py", status="modified"),
            FileChange(path="b.py", status="modified"),
            FileChange(path="c.py", status="added"),
        ]
        result = build_auto_message(staged)
        assert result.startswith("update 3 files (")
        assert "2 modified" in result
        assert "1 added" in result

    def test_empty(self):
        assert build_auto_message([]) == "update files"

    def test_unknown_status_falls_back_to_update(self):
        staged = [FileChange(path="file.py", status="copied")]
        assert build_auto_message(staged) == "update file.py"


class TestCommitPromptAutoSuggestion:
    async def test_prompt_shows_auto_suggestion(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="app.py", status="modified")],
        )
        mock_service.commit.return_value = GitResult(success=True, message="done")

        task = asyncio.create_task(
            handler.handle_command("user1", "commit", "chat1", session)
        )
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)

        prompt_msg = mock_connector.sent_messages[0]
        assert "Suggested: update app.py" in prompt_msg["text"]

        await handler.resolve_input("chat1", "msg")
        await task

    async def test_prompt_shows_use_button(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="app.py", status="modified")],
        )
        mock_service.commit.return_value = GitResult(success=True, message="done")

        task = asyncio.create_task(
            handler.handle_command("user1", "commit", "chat1", session)
        )
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)

        prompt_msg = mock_connector.sent_messages[0]
        assert prompt_msg["buttons"] is not None
        button_data = [b.callback_data for row in prompt_msg["buttons"] for b in row]
        assert f"{GIT_CALLBACK_PREFIX}commit_auto" in button_data
        button_texts = [b.text for row in prompt_msg["buttons"] for b in row]
        assert any("Use: update app.py" in t for t in button_texts)

        await handler.resolve_input("chat1", "msg")
        await task

    async def test_commit_auto_callback_resolves_pending(
        self, handler, mock_service, mock_connector, session
    ):
        mock_service.status.return_value = GitStatus(
            branch="main",
            staged=[FileChange(path="app.py", status="modified")],
        )
        mock_service.commit.return_value = GitResult(success=True, message="done")

        task = asyncio.create_task(
            handler.handle_command("user1", "commit", "chat1", session)
        )
        for _ in range(50):
            if handler.has_pending_input("chat1"):
                break
            await asyncio.sleep(0.01)
        assert handler.has_pending_input("chat1")

        # Simulate tapping the "Use" button
        await handler.handle_callback("user1", "chat1", "commit_auto", "", session)

        result = await task
        # Should have committed with the auto-generated message
        mock_service.commit.assert_called_once()
        commit_msg = mock_service.commit.call_args[0][1]
        assert commit_msg == "update app.py"
        assert "done" in result

    async def test_commit_auto_callback_no_pending(
        self, handler, mock_connector, session
    ):
        # No pending input — tapping button should show error
        await handler.handle_callback("user1", "chat1", "commit_auto", "", session)
        assert len(mock_connector.sent_messages) == 1
        assert "No pending commit" in mock_connector.sent_messages[0]["text"]
