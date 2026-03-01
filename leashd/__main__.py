"""Shim for `uv run -m leashd`."""

from leashd.main import run

if __name__ == "__main__":
    run()
