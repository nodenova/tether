.PHONY: check

check:
	uv run ruff check --fix . && uv run ruff format .
	uv run mypy leashd/ --explicit-package-bases || true
	uv run pytest
