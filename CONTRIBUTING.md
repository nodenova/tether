# Contributing to leashd

Thanks for your interest in contributing! leashd is an open-source project and we welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code.

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) — installed and authenticated (for integration testing)

### Development Setup

```bash
# Clone the repo
git clone git@github.com:nodenova/leashd.git && cd leashd

# Install all dependencies (including dev)
uv sync

# Verify everything works
uv run pytest tests/ -v
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_policy.py -v

# Run tests with coverage
uv run pytest --cov=leashd tests/

# Run tests matching a pattern
uv run pytest -k "test_sandbox" -v
```

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
# Check for lint issues
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

All code must pass `ruff check` and `ruff format --check` before merging. CI enforces this.

## How to Contribute

### Reporting Bugs

Open a [bug report issue](https://github.com/nodenova/leashd/issues/new?template=bug_report.md) with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Python version, leashd version)

### Suggesting Features

Open a [feature request issue](https://github.com/nodenova/leashd/issues/new?template=feature_request.md). Describe the problem you're trying to solve, not just the solution you have in mind.

### Submitting Code

1. Fork the repository
2. Create a feature branch from `main` (`git checkout -b feat/my-feature`)
3. Make your changes
4. Ensure tests pass (`uv run pytest tests/`)
5. Ensure code passes lint (`uv run ruff check .`)
6. Commit with a descriptive message
7. Push to your fork and open a Pull Request

### Commit Messages

We follow conventional-ish commits. Prefix with the type of change:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `chore:` — tooling, CI, dependencies

Example: `feat: add Slack connector`

### Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if behavior changes
- Ensure CI passes before requesting review

## Architecture Overview

If you're new to the codebase, here's a quick orientation:

- `leashd/` — main package
- `tests/` — test suite
- `policies/` — built-in YAML safety policies
- `docs/` — additional documentation
- `.leashd/` — runtime configuration

The core flow is: **Connector** (Telegram) → **Middleware** (auth, rate limiting) → **Engine** → **Agent** (Claude Code) → **Gatekeeper** (safety pipeline) → response.

The **Gatekeeper** orchestrates three safety layers: sandbox → policy rules → human approval.

## Questions?

Open a [discussion](https://github.com/nodenova/leashd/discussions) in the Q&A category. We're happy to help!
