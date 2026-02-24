"""CLI entry point for Tether."""

import asyncio
import signal
import sys

import structlog

from tether.app import build_engine
from tether.core.config import TetherConfig
from tether.exceptions import ConnectorError

logger = structlog.get_logger()


async def _run_cli(config: TetherConfig) -> None:
    engine = build_engine(config)
    await engine.startup()

    logger.info(
        "cli_starting",
        working_directories=[str(d) for d in config.approved_directories],
    )
    print(f"Tether ready — working in {config.approved_directories}")
    print("Enter a prompt (Ctrl+D to exit):\n")

    try:
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                break

            if not prompt.strip():
                continue

            print("\nProcessing...\n")
            response = await engine.handle_message(
                user_id="cli",
                text=prompt,
                chat_id="cli",
            )
            print(f"\n{response}\n")
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("cli_shutting_down")
        await engine.shutdown()
        print("\nShutdown complete.")


async def _run_telegram(config: TetherConfig) -> None:
    from tether.connectors.telegram import TelegramConnector

    connector = TelegramConnector(config.telegram_bot_token)  # type: ignore[arg-type]
    engine = build_engine(config, connector=connector)
    await engine.startup()
    try:
        await connector.start()
    except Exception:
        logger.error("telegram_startup_failed")
        await engine.shutdown()
        raise

    logger.info(
        "telegram_starting",
        working_directories=[str(d) for d in config.approved_directories],
    )
    print(f"Tether ready via Telegram — working in {config.approved_directories}")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    finally:
        logger.info("telegram_shutting_down")
        await connector.stop()
        await engine.shutdown()
        print("\nShutdown complete.")


async def main() -> None:
    try:
        config = TetherConfig()  # type: ignore[call-arg]  # pydantic-settings loads from env
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("Set TETHER_APPROVED_DIRECTORIES or create a .env file.", file=sys.stderr)
        sys.exit(1)

    try:
        if config.telegram_bot_token:
            await _run_telegram(config)
        else:
            await _run_cli(config)
    except ConnectorError as e:
        print(f"Connector failed: {e}", file=sys.stderr)
        sys.exit(1)


def run() -> None:
    asyncio.run(main())
