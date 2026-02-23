"""Telegram connector — translates between Telegram API and BaseConnector."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from tether.connectors.base import BaseConnector, InlineButton

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_MAX_MESSAGE_LENGTH = 4000  # Telegram limit is 4096; leave buffer
_APPROVAL_PREFIX = "approval:"
_INTERACTION_PREFIX = "interact:"
_INTERACTION_CLEANUP_DELAY = (
    4.0  # seconds before deleting resolved interaction messages
)
_GIT_PREFIX = "git:"


class TelegramConnector(BaseConnector):
    def __init__(self, bot_token: str) -> None:
        super().__init__()
        self._token = bot_token
        self._app: Application | None = None  # type: ignore[type-arg]
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._activity_message_id: dict[str, str] = {}
        self._plan_message_ids: dict[str, list[str]] = {}
        self._question_message_ids: dict[str, str] = {}
        self._approval_tool_names: dict[str, str] = {}  # approval_id -> tool_name

    async def start(self) -> None:
        self._app = (
            Application.builder().token(self._token).concurrent_updates(True).build()
        )
        self._app.add_handler(
            CommandHandler(
                ["plan", "edit", "default", "status", "clear", "dir", "git"],
                self._on_command,
            )
        )
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
        self._app.add_error_handler(self._on_error)
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(  # type: ignore[union-attr]
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        logger.info("telegram_connector_started")

    async def stop(self) -> None:
        if self._app is None:
            return
        await self._app.updater.stop()  # type: ignore[union-attr]
        await self._app.stop()
        await self._app.shutdown()
        logger.info("telegram_connector_stopped")

    async def send_message(
        self,
        chat_id: str,
        text: str,
        buttons: list[list[InlineButton]] | None = None,
    ) -> None:
        if self._app is None:
            return
        chunks = _split_text(text)
        markup = _to_telegram_markup(buttons) if buttons else None
        try:
            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                    reply_markup=markup if is_last else None,
                )
            logger.info(
                "telegram_message_sent",
                chat_id=chat_id,
                text_length=len(text),
                chunk_count=len(chunks),
            )
        except Exception:
            logger.exception("telegram_send_message_failed", chat_id=chat_id)

    async def send_message_with_id(self, chat_id: str, text: str) -> str | None:
        if self._app is None:
            return None
        try:
            msg = await self._app.bot.send_message(
                chat_id=int(chat_id),
                text=text[:_MAX_MESSAGE_LENGTH],
            )
            return str(msg.message_id)
        except Exception:
            logger.exception("telegram_send_message_with_id_failed", chat_id=chat_id)
            return None

    async def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        if self._app is None:
            return
        try:
            await self._app.bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(message_id),
                text=text[:_MAX_MESSAGE_LENGTH],
            )
        except Exception:
            logger.debug("telegram_edit_message_failed", chat_id=chat_id)

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        if self._app is None:
            return
        try:
            await self._app.bot.delete_message(
                chat_id=int(chat_id),
                message_id=int(message_id),
            )
        except Exception:
            logger.debug("telegram_delete_message_failed", chat_id=chat_id)

    async def _send_message_with_id_and_buttons(
        self,
        chat_id: str,
        text: str,
        buttons: list[list[InlineButton]],
    ) -> str | None:
        if self._app is None:
            return None
        try:
            markup = _to_telegram_markup(buttons)
            msg = await self._app.bot.send_message(
                chat_id=int(chat_id),
                text=text[:_MAX_MESSAGE_LENGTH],
                reply_markup=markup,
            )
            return str(msg.message_id)
        except Exception:
            logger.exception(
                "telegram_send_message_with_buttons_failed", chat_id=chat_id
            )
            return None

    async def send_activity(
        self,
        chat_id: str,
        tool_name: str,  # noqa: ARG002
        description: str,
    ) -> str | None:
        if self._app is None:
            return None
        text = f"\u23f3 Running: {description}"
        existing = self._activity_message_id.get(chat_id)
        if existing:
            await self.edit_message(chat_id, existing, text)
            return existing
        msg_id = await self.send_message_with_id(chat_id, text)
        if msg_id:
            self._activity_message_id[chat_id] = msg_id
        return msg_id

    async def clear_activity(self, chat_id: str) -> None:
        msg_id = self._activity_message_id.pop(chat_id, None)
        if msg_id:
            await self.delete_message(chat_id, msg_id)

    async def send_plan_messages(
        self,
        chat_id: str,
        plan_text: str,
    ) -> list[str]:
        if self._app is None:
            return []
        ids: list[str] = []
        chunks = _split_text(plan_text)
        for chunk in chunks:
            msg_id = await self.send_message_with_id(chat_id, chunk)
            if msg_id:
                ids.append(msg_id)
        self._plan_message_ids[chat_id] = ids
        return ids

    async def delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
    ) -> None:
        for msg_id in message_ids:
            await self.delete_message(chat_id, msg_id)
        self._plan_message_ids.pop(chat_id, None)

    async def clear_plan_messages(self, chat_id: str) -> None:
        if self._app is None:
            return
        plan_ids = self._plan_message_ids.pop(chat_id, [])
        for msg_id in plan_ids:
            await self.delete_message(chat_id, msg_id)

    async def _delayed_delete(
        self, chat_id: str, message_id: str, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        await self.delete_message(chat_id, message_id)

    def _schedule_cleanup(self, chat_id: str, message_id: str) -> None:
        task = asyncio.create_task(
            self._delayed_delete(chat_id, message_id, _INTERACTION_CLEANUP_DELAY)
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def send_typing_indicator(self, chat_id: str) -> None:
        if self._app is None:
            return
        try:
            await self._app.bot.send_chat_action(
                chat_id=int(chat_id), action=ChatAction.TYPING
            )
        except Exception:
            logger.exception("telegram_typing_indicator_failed", chat_id=chat_id)

    async def request_approval(
        self, chat_id: str, approval_id: str, description: str, tool_name: str = ""
    ) -> str | None:
        if tool_name.startswith("Bash::"):
            cmd = tool_name.split("::", 1)[1]
            approve_all_text = f"Approve all '{cmd}' cmds"
        elif tool_name:
            approve_all_text = f"Approve all {tool_name}"
        else:
            approve_all_text = "Approve all in session"

        self._approval_tool_names[approval_id] = tool_name

        buttons = [
            [
                InlineButton(
                    text="Approve",
                    callback_data=f"{_APPROVAL_PREFIX}yes:{approval_id}",
                ),
                InlineButton(
                    text="Reject",
                    callback_data=f"{_APPROVAL_PREFIX}no:{approval_id}",
                ),
            ],
            [
                InlineButton(
                    text=approve_all_text,
                    callback_data=f"{_APPROVAL_PREFIX}all:{approval_id}",
                ),
            ],
        ]
        msg_id = await self._send_message_with_id_and_buttons(
            chat_id, description, buttons
        )
        logger.info(
            "telegram_approval_requested",
            chat_id=chat_id,
            approval_id=approval_id,
        )
        return msg_id

    async def send_file(self, chat_id: str, file_path: str) -> None:
        if self._app is None:
            return
        try:
            path = Path(file_path)
            with path.open("rb") as f:
                await self._app.bot.send_document(chat_id=int(chat_id), document=f)
            logger.info(
                "telegram_file_sent",
                chat_id=chat_id,
                file_path=file_path,
            )
        except Exception:
            logger.exception(
                "telegram_send_file_failed",
                chat_id=chat_id,
                file_path=file_path,
            )

    async def send_question(
        self,
        chat_id: str,
        interaction_id: str,
        question_text: str,
        header: str,
        options: list[dict[str, str]],
    ) -> None:
        text = f"**{header}**\n{question_text}" if header else question_text
        rows = []
        for opt in options:
            label = opt.get("label", "")
            callback_data = f"{_INTERACTION_PREFIX}{interaction_id}:{label}"
            # Telegram callback_data max is 64 bytes
            if len(callback_data.encode()) > 64:
                callback_data = callback_data[:64]
            rows.append([InlineButton(text=label, callback_data=callback_data)])
        hint = "\nOr reply with a message for a custom answer."
        msg_id = await self._send_message_with_id_and_buttons(
            chat_id, text + hint, rows
        )
        if msg_id:
            self._question_message_ids[chat_id] = msg_id
        logger.info(
            "telegram_question_sent",
            chat_id=chat_id,
            interaction_id=interaction_id,
            option_count=len(options),
            has_header=bool(header),
        )

    async def send_plan_review(
        self,
        chat_id: str,
        interaction_id: str,
        description: str,
    ) -> None:
        logger.info(
            "telegram_plan_review_sending",
            chat_id=chat_id,
            description_length=len(description),
            will_split=len(description) > _MAX_MESSAGE_LENGTH,
        )
        plan_ids = await self.send_plan_messages(chat_id, description)

        buttons = [
            [
                InlineButton(
                    text="Yes, clear context and auto-accept edits",
                    callback_data=f"{_INTERACTION_PREFIX}{interaction_id}:clean_edit",
                ),
            ],
            [
                InlineButton(
                    text="Yes, auto-accept edits",
                    callback_data=f"{_INTERACTION_PREFIX}{interaction_id}:edit",
                ),
            ],
            [
                InlineButton(
                    text="Yes, manually approve edits",
                    callback_data=f"{_INTERACTION_PREFIX}{interaction_id}:default",
                ),
            ],
            [
                InlineButton(
                    text="Adjust the plan",
                    callback_data=f"{_INTERACTION_PREFIX}{interaction_id}:adjust",
                ),
            ],
        ]
        review_header = "Claude has written up a plan. Proceed with implementation?"
        review_msg_id = await self._send_message_with_id_and_buttons(
            chat_id, review_header, buttons
        )
        if review_msg_id:
            plan_ids.append(review_msg_id)
        self._plan_message_ids[chat_id] = plan_ids

    async def _on_command(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or not update.message.from_user:
            return
        if self._command_handler is None:
            return

        user_id = str(update.message.from_user.id)
        chat_id = str(update.message.chat_id)
        raw = update.message.text or ""
        tokens = raw.split()
        first_token = tokens[0] if tokens else ""
        command = first_token.lstrip("/").split("@")[0]
        args = raw[len(first_token) :].strip()

        try:
            response = await self._command_handler(user_id, command, args, chat_id)
            if response:
                await self.send_message(chat_id, response)
        except Exception:
            logger.exception("telegram_command_handler_error", chat_id=chat_id)

    async def _on_message(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or not update.message.text:
            return
        if not update.message.from_user:
            return
        if self._message_handler is None:
            return

        user_id = str(update.message.from_user.id)
        text = update.message.text
        chat_id = str(update.message.chat_id)
        message_id = str(update.message.message_id)

        logger.info(
            "telegram_message_received",
            user_id=user_id,
            chat_id=chat_id,
            text_length=len(text),
        )

        await self.send_typing_indicator(chat_id)
        try:
            result = await self._message_handler(user_id, text, chat_id)
            if result == "":
                await self.delete_message(chat_id, message_id)
        except Exception:
            logger.exception("telegram_message_handler_error", chat_id=chat_id)
            await self.send_message(
                chat_id, "An error occurred while processing your message."
            )

    async def _on_callback_query(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None:
            return

        try:
            await query.answer()
        except Exception:
            logger.debug("telegram_callback_answer_failed")

        data = query.data or ""

        if data.startswith(_GIT_PREFIX):
            await self._handle_git_callback(query, data)
            return

        if data.startswith(_INTERACTION_PREFIX):
            await self._handle_interaction_callback(query, data)
            return

        if not data.startswith(_APPROVAL_PREFIX):
            return

        suffix = data[len(_APPROVAL_PREFIX) :]
        if ":" not in suffix:
            return

        decision, rest = suffix.split(":", 1)
        if not rest:
            return

        if decision == "all":
            approval_id = rest
            tool_name = self._approval_tool_names.pop(approval_id, "")
        else:
            approval_id = rest
            tool_name = ""
            self._approval_tool_names.pop(approval_id, None)

        if not approval_id:
            return

        approved = decision in ("yes", "all")
        logger.info(
            "telegram_approval_resolved",
            approval_id=approval_id,
            approved=approved,
            auto_approve=decision == "all",
        )

        resolved = False
        if self._approval_resolver:
            try:
                resolved = await self._approval_resolver(approval_id, approved)
            except Exception:
                logger.exception(
                    "telegram_approval_resolver_error",
                    approval_id=approval_id,
                )

        if resolved and decision == "all" and self._auto_approve_handler:
            chat_id = str(query.message.chat_id)  # type: ignore[union-attr]
            self._auto_approve_handler(chat_id, tool_name)

        if resolved:
            if decision == "all":
                if tool_name.startswith("Bash::"):
                    cmd = tool_name.split("::", 1)[1]
                    status = f"Approved \u2713 (all future '{cmd}' cmds auto-approved)"
                elif tool_name:
                    status = f"Approved \u2713 (all future {tool_name} auto-approved)"
                else:
                    status = "Approved \u2713 (all future tools auto-approved)"
            elif approved:
                status = "Approved \u2713"
            else:
                status = "Rejected \u2717"
        else:
            status = "Expired (approval no longer active)"

        try:
            raw = f"{query.message.text}\n\n{status}"  # type: ignore[union-attr]
            await query.edit_message_text(raw)
            if resolved and isinstance(query.message, Message):
                chat_id = str(query.message.chat_id)
                msg_id = str(query.message.message_id)
                self._schedule_cleanup(chat_id, msg_id)
        except Exception:
            logger.exception("telegram_edit_approval_message_failed")

    async def _handle_interaction_callback(
        self, query: CallbackQuery, data: str
    ) -> None:
        suffix = data[len(_INTERACTION_PREFIX) :]
        if ":" not in suffix:
            return

        interaction_id, answer = suffix.split(":", 1)
        if not interaction_id or not answer:
            return

        logger.info(
            "telegram_interaction_resolved",
            interaction_id=interaction_id,
            answer=answer,
        )

        resolved = False
        if self._interaction_resolver:
            try:
                resolved = await self._interaction_resolver(interaction_id, answer)
            except Exception:
                logger.exception(
                    "telegram_interaction_resolver_error",
                    interaction_id=interaction_id,
                )

        if not isinstance(query.message, Message):
            return

        chat_id = str(query.message.chat_id)

        if not resolved:
            try:
                raw = f"{query.message.text}\n\nExpired (interaction no longer active)"
                await query.edit_message_text(raw)
            except Exception:
                logger.exception("telegram_edit_interaction_message_failed")
            return

        is_plan_review = answer in ("clean_edit", "edit", "default", "adjust")
        if is_plan_review:
            plan_ids = self._plan_message_ids.pop(chat_id, [])
            if plan_ids:
                await self.delete_messages(chat_id, plan_ids)
        else:
            msg_id = self._question_message_ids.pop(chat_id, None)
            if msg_id:
                await self.delete_message(chat_id, msg_id)

    async def _handle_git_callback(self, query: CallbackQuery, data: str) -> None:
        """Route git inline button callbacks to the registered git handler."""
        suffix = data[len(_GIT_PREFIX) :]
        if ":" not in suffix:
            action, payload = suffix, ""
        else:
            action, payload = suffix.split(":", 1)

        if not self._git_handler:
            return

        user_id = str(query.from_user.id) if query.from_user else ""
        chat_id = (
            str(query.message.chat_id) if isinstance(query.message, Message) else ""
        )

        if not user_id or not chat_id:
            return

        try:
            await self._git_handler(user_id, chat_id, action, payload)
        except Exception:
            logger.exception("telegram_git_callback_error", chat_id=chat_id)

    async def _on_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error(
            "telegram_error",
            error=str(context.error),
            update=str(update),
        )


def _split_text(text: str) -> list[str]:
    if not text:
        return [""]
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= _MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break

        split_at = text.rfind("\n", 0, _MAX_MESSAGE_LENGTH)
        if split_at <= 0:
            split_at = text.rfind(" ", 0, _MAX_MESSAGE_LENGTH)
        if split_at <= 0:
            split_at = _MAX_MESSAGE_LENGTH

        chunks.append(text[:split_at])
        text = (
            text[split_at + 1 :] if split_at < _MAX_MESSAGE_LENGTH else text[split_at:]
        )

    return chunks


def _to_telegram_markup(
    buttons: list[list[InlineButton]],
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(text=btn.text, callback_data=btn.callback_data)
                for btn in row
            ]
            for row in buttons
        ]
    )
