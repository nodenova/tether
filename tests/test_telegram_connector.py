"""Tests for the Telegram connector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Message

from tether.connectors.base import InlineButton
from tether.connectors.telegram import (
    _MAX_MESSAGE_LENGTH,
    TelegramConnector,
    _split_text,
    _to_telegram_markup,
)

# --- Pure function tests ---


class TestSplitText:
    def test_short_text_single_chunk(self):
        assert _split_text("hello") == ["hello"]

    def test_empty_text(self):
        assert _split_text("") == [""]

    def test_exact_limit_no_split(self):
        text = "a" * 4000
        assert _split_text(text) == [text]

    def test_splits_at_newline(self):
        line = "a" * 1500
        text = f"{line}\n{line}\n{line}"
        chunks = _split_text(text)
        assert len(chunks) == 2
        assert chunks[0] == f"{line}\n{line}"
        assert chunks[1] == line

    def test_splits_at_space_when_no_newline(self):
        word = "a" * 1999
        text = f"{word} {word} {word}"
        chunks = _split_text(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4000

    def test_hard_break_no_whitespace(self):
        text = "a" * 5000
        chunks = _split_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 4000
        assert chunks[1] == "a" * 1000

    def test_space_at_position_zero_no_infinite_loop(self):
        text = " " + "a" * 5000
        chunks = _split_text(text)
        assert len(chunks) >= 2
        assert all(chunk for chunk in chunks)  # no empty chunks
        assert "".join(chunks) == text  # all content preserved, no infinite loop

    def test_newline_at_position_zero(self):
        text = "\n" + "a" * 5000
        chunks = _split_text(text)
        assert len(chunks) >= 2
        assert all(chunk for chunk in chunks)  # no empty chunks


class TestToTelegramMarkup:
    def test_single_row(self):
        buttons = [[InlineButton(text="OK", callback_data="ok")]]
        markup = _to_telegram_markup(buttons)
        assert len(markup.inline_keyboard) == 1
        assert markup.inline_keyboard[0][0].text == "OK"
        assert markup.inline_keyboard[0][0].callback_data == "ok"

    def test_multiple_rows(self):
        buttons = [
            [InlineButton(text="A", callback_data="a")],
            [
                InlineButton(text="B", callback_data="b"),
                InlineButton(text="C", callback_data="c"),
            ],
        ]
        markup = _to_telegram_markup(buttons)
        assert len(markup.inline_keyboard) == 2
        assert len(markup.inline_keyboard[1]) == 2
        assert markup.inline_keyboard[1][1].text == "C"


# --- Connector method tests ---


def _make_mock_app():
    """Create a mock Application with bot and updater."""
    app = AsyncMock()
    app.bot = AsyncMock()
    app.updater = AsyncMock()
    app.add_handler = MagicMock()
    app.builder = MagicMock()
    return app


@pytest.fixture
def connector():
    return TelegramConnector("fake:token")


class TestStart:
    @pytest.mark.asyncio
    async def test_start_creates_app_and_starts_polling(self, connector):
        mock_app = _make_mock_app()
        mock_app.add_error_handler = MagicMock()
        mock_builder = MagicMock()
        mock_builder.token.return_value = mock_builder
        mock_builder.concurrent_updates.return_value = mock_builder
        mock_builder.build.return_value = mock_app

        with patch(
            "tether.connectors.telegram.Application.builder",
            return_value=mock_builder,
        ):
            await connector.start()

        mock_builder.token.assert_called_once_with("fake:token")
        mock_builder.concurrent_updates.assert_called_once_with(True)
        assert mock_app.add_handler.call_count == 3
        mock_app.add_error_handler.assert_called_once()
        mock_app.initialize.assert_awaited_once()
        mock_app.start.assert_awaited_once()
        mock_app.updater.start_polling.assert_awaited_once()


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_shuts_down_in_order(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.stop()

        mock_app.updater.stop.assert_awaited_once()
        mock_app.stop.assert_awaited_once()
        mock_app.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, connector):
        await connector.stop()  # should not raise


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_sends_short_message(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.send_message("123", "hello")

        mock_app.bot.send_message.assert_awaited_once()
        call_kwargs = mock_app.bot.send_message.await_args.kwargs
        assert call_kwargs["chat_id"] == 123
        assert "parse_mode" not in call_kwargs
        assert call_kwargs["reply_markup"] is None

    @pytest.mark.asyncio
    async def test_sends_long_message_in_chunks(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        text = "a" * 5000
        await connector.send_message("123", text)

        assert mock_app.bot.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_buttons_on_last_chunk_only(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        buttons = [[InlineButton(text="OK", callback_data="ok")]]
        text = "a" * 5000
        await connector.send_message("123", text, buttons=buttons)

        calls = mock_app.bot.send_message.await_args_list
        assert calls[0].kwargs["reply_markup"] is None
        assert calls[1].kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_no_app_is_noop(self, connector):
        await connector.send_message("123", "hello")  # should not raise

    @pytest.mark.asyncio
    async def test_send_message_exception_logged_not_raised(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_app.bot.send_message.side_effect = RuntimeError("network error")

        await connector.send_message("123", "hello")  # should not raise

    @pytest.mark.asyncio
    async def test_partial_chunk_failure(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        call_count = 0

        async def send_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("network error on chunk 2")

        mock_app.bot.send_message.side_effect = send_side_effect

        text = "a" * 5000  # will be split into 2 chunks
        await connector.send_message("123", text)  # should not raise

        assert call_count == 2


class TestSendTypingIndicator:
    @pytest.mark.asyncio
    async def test_sends_typing_action(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.send_typing_indicator("456")

        mock_app.bot.send_chat_action.assert_awaited_once()
        call_kwargs = mock_app.bot.send_chat_action.await_args.kwargs
        assert call_kwargs["chat_id"] == 456


class TestRequestApproval:
    @pytest.mark.asyncio
    async def test_sends_description_with_buttons(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.request_approval("123", "abc-123", "Run rm -rf?", "Bash")

        call_kwargs = mock_app.bot.send_message.await_args.kwargs
        assert "Run rm" in call_kwargs["text"]
        assert "rf" in call_kwargs["text"]
        markup = call_kwargs["reply_markup"]
        assert markup is not None
        buttons = markup.inline_keyboard[0]
        assert buttons[0].text == "Approve"
        assert "yes:abc-123" in buttons[0].callback_data
        assert buttons[1].text == "Reject"
        assert "no:abc-123" in buttons[1].callback_data


class TestSendFile:
    @pytest.mark.asyncio
    async def test_sends_document(self, connector, tmp_path):
        mock_app = _make_mock_app()
        connector._app = mock_app

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        await connector.send_file("123", str(test_file))

        mock_app.bot.send_document.assert_awaited_once()
        call_kwargs = mock_app.bot.send_document.await_args.kwargs
        assert call_kwargs["chat_id"] == 123

    @pytest.mark.asyncio
    async def test_send_nonexistent_file_logged_not_raised(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.send_file("123", "/nonexistent/path.txt")  # should not raise

        mock_app.bot.send_document.assert_not_awaited()


# --- Handler tests ---


def _make_update(user_id=1, text="hello", chat_id=100):
    """Create a mock Telegram Update for message handlers."""
    update = MagicMock()
    update.message.from_user.id = user_id
    update.message.text = text
    update.message.chat_id = chat_id
    return update


def _make_callback_update(data="approval:yes:abc-123"):
    """Create a mock Telegram Update for callback query handlers."""
    update = MagicMock()
    update.callback_query.answer = AsyncMock()
    update.callback_query.data = data
    update.callback_query.edit_message_text = AsyncMock()
    msg = MagicMock(spec=Message)
    msg.text = "Original message"
    update.callback_query.message = msg
    return update


class TestOnMessage:
    @pytest.mark.asyncio
    async def test_delegates_to_handler(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="response")
        connector.set_message_handler(handler)

        update = _make_update(user_id=42, text="fix bug", chat_id=99)
        await connector._on_message(update, MagicMock())

        handler.assert_awaited_once_with("42", "fix bug", "99")

    @pytest.mark.asyncio
    async def test_does_not_send_response_itself(self, connector):
        """Connector delegates delivery to Engine; it must not send the response."""
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="done")
        connector.set_message_handler(handler)

        update = _make_update(chat_id=99)
        await connector._on_message(update, MagicMock())

        for call in mock_app.bot.send_message.await_args_list:
            assert call.kwargs.get("text") != "done"

    @pytest.mark.asyncio
    async def test_no_handler_is_noop(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        update = _make_update()
        await connector._on_message(update, MagicMock())  # should not raise

    @pytest.mark.asyncio
    async def test_no_message_is_noop(self, connector):
        update = MagicMock()
        update.message = None
        await connector._on_message(update, MagicMock())  # should not raise

    @pytest.mark.asyncio
    async def test_handler_error_sends_error_message(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(side_effect=RuntimeError("boom"))
        connector.set_message_handler(handler)

        update = _make_update(chat_id=99)
        await connector._on_message(update, MagicMock())

        calls = mock_app.bot.send_message.await_args_list
        error_text = calls[-1].kwargs["text"]
        assert "error" in error_text.lower()

    @pytest.mark.asyncio
    async def test_message_without_from_user_is_noop(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="response")
        connector.set_message_handler(handler)

        update = MagicMock()
        update.message.text = "hello"
        update.message.from_user = None
        await connector._on_message(update, MagicMock())

        handler.assert_not_awaited()
        mock_app.bot.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sends_typing_indicator_before_handler(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        call_order = []

        async def fake_handler(user_id, text, chat_id):
            call_order.append("handler")
            return "ok"

        async def fake_typing(**kwargs):
            call_order.append("typing")

        mock_app.bot.send_chat_action.side_effect = fake_typing
        connector.set_message_handler(fake_handler)

        update = _make_update(chat_id=99)
        await connector._on_message(update, MagicMock())

        assert call_order == ["typing", "handler"]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_leak_details(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(side_effect=RuntimeError("secret db password"))
        connector.set_message_handler(handler)

        update = _make_update(chat_id=99)
        await connector._on_message(update, MagicMock())

        calls = mock_app.bot.send_message.await_args_list
        error_text = calls[-1].kwargs["text"]
        assert "secret db password" not in error_text
        assert "error" in error_text.lower()


class TestOnCallbackQuery:
    @pytest.mark.asyncio
    async def test_approval_yes_resolves_true(self, connector):
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:abc-123")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("abc-123", True)

    @pytest.mark.asyncio
    async def test_approval_no_resolves_false(self, connector):
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:no:abc-123")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("abc-123", False)

    @pytest.mark.asyncio
    async def test_non_approval_callback_ignored(self, connector):
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("other:data")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edits_message_with_status(self, connector):
        connector.set_approval_resolver(AsyncMock(return_value=True))

        update = _make_callback_update("approval:yes:abc-123")
        await connector._on_callback_query(update, MagicMock())

        update.callback_query.edit_message_text.assert_awaited_once()
        call_args = update.callback_query.edit_message_text.await_args
        edited_text = call_args[0][0]
        assert "Approved \u2713" in edited_text

    @pytest.mark.asyncio
    async def test_no_query_is_noop(self, connector):
        update = MagicMock()
        update.callback_query = None
        await connector._on_callback_query(update, MagicMock())

    @pytest.mark.asyncio
    async def test_answers_callback_query(self, connector):
        update = _make_callback_update("approval:yes:abc-123")
        connector.set_approval_resolver(AsyncMock())

        await connector._on_callback_query(update, MagicMock())

        update.callback_query.answer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_approval_prefix_only_ignored(self, connector):
        """Data 'approval:' with no colon in suffix is already handled."""
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_approval_id_ignored(self, connector):
        """Data 'approval:yes:' produces empty approval_id — must not call resolver."""
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_decision_value_resolves_as_rejected(self, connector):
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:maybe:abc-123")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("abc-123", False)

    @pytest.mark.asyncio
    async def test_approval_id_with_colons_preserved(self, connector):
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:id:with:colons")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("id:with:colons", True)

    @pytest.mark.asyncio
    async def test_resolver_exception_still_edits_message(self, connector):
        resolver = AsyncMock(side_effect=RuntimeError("resolver boom"))
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:abc-123")
        await connector._on_callback_query(update, MagicMock())

        update.callback_query.edit_message_text.assert_awaited_once()
        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "Expired" in edited_text

    @pytest.mark.asyncio
    async def test_no_resolver_set_still_edits_message(self, connector):
        update = _make_callback_update("approval:yes:abc-123")
        await connector._on_callback_query(update, MagicMock())

        update.callback_query.edit_message_text.assert_awaited_once()
        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "Expired" in edited_text

    @pytest.mark.asyncio
    async def test_query_answer_failure_does_not_abort_handler(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:abc-123")
        update.callback_query.answer = AsyncMock(
            side_effect=RuntimeError("Query is too old")
        )
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("abc-123", True)
        update.callback_query.edit_message_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolver_returns_false_shows_expired(self, connector):
        resolver = AsyncMock(return_value=False)
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:abc-123")
        await connector._on_callback_query(update, MagicMock())

        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "Expired" in edited_text
        assert "Approved \u2713" not in edited_text


class TestSendMessageWithId:
    @pytest.mark.asyncio
    async def test_returns_message_id(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        mock_app.bot.send_message.return_value = mock_msg

        result = await connector.send_message_with_id("123", "hello")

        assert result == "42"
        call_kwargs = mock_app.bot.send_message.await_args.kwargs
        assert call_kwargs["chat_id"] == 123
        assert "parse_mode" not in call_kwargs

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_app.bot.send_message.side_effect = RuntimeError("network")

        result = await connector.send_message_with_id("123", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_app_returns_none(self, connector):
        result = await connector.send_message_with_id("123", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_truncates_long_text(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        mock_app.bot.send_message.return_value = mock_msg

        long_text = "x" * 5000
        await connector.send_message_with_id("123", long_text)

        sent_text = mock_app.bot.send_message.await_args.kwargs["text"]
        assert len(sent_text) <= _MAX_MESSAGE_LENGTH


class TestEditMessage:
    @pytest.mark.asyncio
    async def test_edits_message(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.edit_message("123", "42", "updated text")

        mock_app.bot.edit_message_text.assert_awaited_once()
        call_kwargs = mock_app.bot.edit_message_text.await_args.kwargs
        assert call_kwargs["chat_id"] == 123
        assert call_kwargs["message_id"] == 42
        assert "parse_mode" not in call_kwargs

    @pytest.mark.asyncio
    async def test_no_app_is_noop(self, connector):
        await connector.edit_message("123", "42", "text")  # should not raise

    @pytest.mark.asyncio
    async def test_exception_caught(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_app.bot.edit_message_text.side_effect = RuntimeError("edit failed")

        await connector.edit_message("123", "42", "text")  # should not raise

    @pytest.mark.asyncio
    async def test_truncates_long_text(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        long_text = "x" * 5000
        await connector.edit_message("123", "42", long_text)

        sent_text = mock_app.bot.edit_message_text.await_args.kwargs["text"]
        assert len(sent_text) <= _MAX_MESSAGE_LENGTH


class TestTelegramConnectorEdgeCases:
    """Coverage gap closers for telegram.py."""

    @pytest.mark.asyncio
    async def test_send_typing_no_app_noop(self, connector):
        """send_typing_indicator with no app does nothing."""
        await connector.send_typing_indicator("123")  # should not raise

    @pytest.mark.asyncio
    async def test_send_typing_exception_logged(self, connector):
        """Exception in send_typing_indicator is caught."""
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_app.bot.send_chat_action.side_effect = RuntimeError("typing error")
        await connector.send_typing_indicator("123")  # should not raise

    @pytest.mark.asyncio
    async def test_send_file_no_app_noop(self, connector):
        """send_file with no app does nothing."""
        await connector.send_file("123", "/some/file.txt")  # should not raise

    @pytest.mark.asyncio
    async def test_callback_query_no_data_ignored(self, connector):
        """Callback query with None data is handled gracefully."""
        resolver = AsyncMock()
        connector.set_approval_resolver(resolver)
        update = MagicMock()
        update.callback_query.answer = AsyncMock()
        update.callback_query.data = None
        update.callback_query.edit_message_text = AsyncMock()
        await connector._on_callback_query(update, MagicMock())
        resolver.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_split_very_long_no_whitespace(self):
        """Very long string with no whitespace hits hard break at 4000."""
        text = "x" * 12000
        chunks = _split_text(text)
        assert len(chunks) == 3
        assert all(len(c) <= 4000 for c in chunks)
        assert "".join(chunks) == text


class TestDeleteMessage:
    @pytest.mark.asyncio
    async def test_deletes_message(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.delete_message("123", "42")

        mock_app.bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=42)

    @pytest.mark.asyncio
    async def test_no_app_is_noop(self, connector):
        await connector.delete_message("123", "42")  # should not raise

    @pytest.mark.asyncio
    async def test_exception_caught(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_app.bot.delete_message.side_effect = RuntimeError("delete failed")

        await connector.delete_message("123", "42")  # should not raise


class TestTextReplyDeletion:
    @pytest.mark.asyncio
    async def test_user_reply_deleted_when_consumed_as_interaction(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="")
        connector.set_message_handler(handler)

        update = _make_update(user_id=42, text="my answer", chat_id=99)
        update.message.message_id = 555

        await connector._on_message(update, MagicMock())

        mock_app.bot.delete_message.assert_awaited_once_with(chat_id=99, message_id=555)

    @pytest.mark.asyncio
    async def test_user_reply_not_deleted_for_normal_messages(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="some response")
        connector.set_message_handler(handler)

        update = _make_update(user_id=42, text="hello", chat_id=99)
        update.message.message_id = 555

        await connector._on_message(update, MagicMock())

        mock_app.bot.delete_message.assert_not_awaited()


class TestDelayedDelete:
    @pytest.mark.asyncio
    async def test_delayed_delete_waits_then_deletes(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector._delayed_delete("123", "42", 0.01)

        mock_app.bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=42)

    @pytest.mark.asyncio
    async def test_interaction_callback_deletes_question_immediately(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_interaction_resolver(resolver)

        update = _make_callback_update("interact:abc-123:FastAPI")
        update.callback_query.message.chat_id = 100
        update.callback_query.message.message_id = 42

        mock_app = _make_mock_app()
        connector._app = mock_app

        # Track the question message ID
        connector._question_message_ids["100"] = "42"

        await connector._on_callback_query(update, MagicMock())

        # Question message should be immediately deleted
        mock_app.bot.delete_message.assert_awaited_once_with(chat_id=100, message_id=42)

    @pytest.mark.asyncio
    async def test_approval_callback_schedules_deletion(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)

        update = _make_callback_update("approval:yes:abc-123")
        update.callback_query.message.chat_id = 100
        update.callback_query.message.message_id = 42

        mock_app = _make_mock_app()
        connector._app = mock_app

        with patch.object(
            connector, "_schedule_cleanup", wraps=connector._schedule_cleanup
        ) as mock_sched:
            await connector._on_callback_query(update, MagicMock())

            mock_sched.assert_called_once_with("100", "42")


class TestRequestApprovalReturnsMessageId:
    @pytest.mark.asyncio
    async def test_returns_message_id(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_msg = MagicMock()
        mock_msg.message_id = 99
        mock_app.bot.send_message.return_value = mock_msg

        result = await connector.request_approval(
            "123", "abc-123", "Run rm -rf?", "Bash"
        )

        assert result == "99"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_app(self, connector):
        result = await connector.request_approval(
            "123", "abc-123", "Run rm -rf?", "Bash"
        )
        assert result is None


class TestRequestApprovalButtons:
    @pytest.mark.asyncio
    async def test_approve_all_button_includes_tool_name(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.request_approval("123", "abc-123", "Write file?", "Write")

        calls = mock_app.bot.send_message.await_args_list
        markup = calls[-1].kwargs["reply_markup"]
        assert markup is not None
        assert len(markup.inline_keyboard) == 2
        assert markup.inline_keyboard[1][0].text == "Approve all Write"
        assert markup.inline_keyboard[1][0].callback_data == "approval:all:abc-123"

    @pytest.mark.asyncio
    async def test_approve_all_button_fallback_no_tool_name(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.request_approval("123", "abc-123", "Something?")

        calls = mock_app.bot.send_message.await_args_list
        markup = calls[-1].kwargs["reply_markup"]
        assert markup is not None
        assert markup.inline_keyboard[1][0].text == "Approve all in session"

    @pytest.mark.asyncio
    async def test_approve_all_button_scoped_bash_key(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.request_approval("123", "abc-123", "Run uv?", "Bash::uv run")

        calls = mock_app.bot.send_message.await_args_list
        markup = calls[-1].kwargs["reply_markup"]
        assert markup is not None
        assert markup.inline_keyboard[1][0].text == "Approve all 'uv run' cmds"
        assert markup.inline_keyboard[1][0].callback_data == "approval:all:abc-123"

    @pytest.mark.asyncio
    async def test_approve_all_callback_data_within_64_bytes(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        # Even with a very long tool_name, callback_data must stay under 64 bytes
        long_tool = "Bash::OPENAI_API_KEY=sk-test AZURE_OPENAI_API_KEY=az-test uv run"
        await connector.request_approval("123", "abc-123", "Run?", long_tool)

        calls = mock_app.bot.send_message.await_args_list
        markup = calls[-1].kwargs["reply_markup"]
        for row in markup.inline_keyboard:
            for btn in row:
                assert len(btn.callback_data.encode()) <= 64

    @pytest.mark.asyncio
    async def test_tool_name_stored_in_memory(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.request_approval("123", "abc-123", "Write?", "Write")

        assert connector._approval_tool_names["abc-123"] == "Write"


class TestApproveAllCallback:
    @pytest.mark.asyncio
    async def test_all_decision_resolves_as_approved(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Write"

        update = _make_callback_update("approval:all:abc-123")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("abc-123", True)

    @pytest.mark.asyncio
    async def test_all_decision_calls_auto_approve_handler_with_tool_name(
        self, connector
    ):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Write"

        handler_calls = []
        connector.set_auto_approve_handler(
            lambda chat_id, tool_name: handler_calls.append((chat_id, tool_name))
        )

        update = _make_callback_update("approval:all:abc-123")
        update.callback_query.message.chat_id = 999
        await connector._on_callback_query(update, MagicMock())

        assert handler_calls == [("999", "Write")]

    @pytest.mark.asyncio
    async def test_all_decision_shows_tool_specific_status(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector.set_auto_approve_handler(lambda chat_id, tool_name: None)
        connector._approval_tool_names["abc-123"] = "Bash"

        update = _make_callback_update("approval:all:abc-123")
        await connector._on_callback_query(update, MagicMock())

        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "Bash" in edited_text
        assert "Approved \u2713" in edited_text

    @pytest.mark.asyncio
    async def test_all_decision_scoped_bash_status(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector.set_auto_approve_handler(lambda chat_id, tool_name: None)
        connector._approval_tool_names["abc-123"] = "Bash::uv run"

        update = _make_callback_update("approval:all:abc-123")
        await connector._on_callback_query(update, MagicMock())

        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "Approved \u2713" in edited_text
        assert "'uv run' cmds auto-approved" in edited_text

    @pytest.mark.asyncio
    async def test_all_decision_scoped_bash_calls_handler_with_key(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Bash::uv run"

        handler_calls = []
        connector.set_auto_approve_handler(
            lambda chat_id, tool_name: handler_calls.append((chat_id, tool_name))
        )

        update = _make_callback_update("approval:all:abc-123")
        update.callback_query.message.chat_id = 999
        await connector._on_callback_query(update, MagicMock())

        assert handler_calls == [("999", "Bash::uv run")]

    @pytest.mark.asyncio
    async def test_all_decision_no_handler_still_resolves(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Edit"
        # No auto_approve_handler set

        update = _make_callback_update("approval:all:abc-123")
        await connector._on_callback_query(update, MagicMock())

        resolver.assert_awaited_once_with("abc-123", True)
        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "Approved \u2713" in edited_text

    @pytest.mark.asyncio
    async def test_tool_name_cleaned_up_on_all_callback(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Write"

        update = _make_callback_update("approval:all:abc-123")
        await connector._on_callback_query(update, MagicMock())

        assert "abc-123" not in connector._approval_tool_names

    @pytest.mark.asyncio
    async def test_tool_name_cleaned_up_on_yes_callback(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Write"

        update = _make_callback_update("approval:yes:abc-123")
        await connector._on_callback_query(update, MagicMock())

        assert "abc-123" not in connector._approval_tool_names

    @pytest.mark.asyncio
    async def test_tool_name_cleaned_up_on_no_callback(self, connector):
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector._approval_tool_names["abc-123"] = "Write"

        update = _make_callback_update("approval:no:abc-123")
        await connector._on_callback_query(update, MagicMock())

        assert "abc-123" not in connector._approval_tool_names

    @pytest.mark.asyncio
    async def test_missing_tool_name_defaults_to_empty(self, connector):
        """If approval_id has no stored tool_name, default to empty string."""
        resolver = AsyncMock(return_value=True)
        connector.set_approval_resolver(resolver)
        connector.set_auto_approve_handler(lambda chat_id, tool_name: None)

        update = _make_callback_update("approval:all:unknown-id")
        await connector._on_callback_query(update, MagicMock())

        edited_text = update.callback_query.edit_message_text.await_args[0][0]
        assert "all future tools auto-approved" in edited_text


class TestOnCommand:
    @pytest.mark.asyncio
    async def test_command_handler_called(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="Mode switched")
        connector.set_command_handler(handler)

        update = _make_update(user_id=42, text="/plan", chat_id=99)
        await connector._on_command(update, MagicMock())

        handler.assert_awaited_once_with("42", "plan", "", "99")

    @pytest.mark.asyncio
    async def test_command_with_args(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="OK")
        connector.set_command_handler(handler)

        update = _make_update(user_id=42, text="/edit extra args", chat_id=99)
        await connector._on_command(update, MagicMock())

        handler.assert_awaited_once_with("42", "edit", "extra args", "99")

    @pytest.mark.asyncio
    async def test_command_with_bot_mention(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="OK")
        connector.set_command_handler(handler)

        update = _make_update(user_id=42, text="/status@mybot", chat_id=99)
        await connector._on_command(update, MagicMock())

        handler.assert_awaited_once_with("42", "status", "", "99")

    @pytest.mark.asyncio
    async def test_command_response_sent(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="Switched to plan mode.")
        connector.set_command_handler(handler)

        update = _make_update(user_id=42, text="/plan", chat_id=99)
        await connector._on_command(update, MagicMock())

        mock_app.bot.send_message.assert_awaited_once()
        sent_text = mock_app.bot.send_message.await_args.kwargs["text"]
        assert "Switched to plan mode" in sent_text

    @pytest.mark.asyncio
    async def test_no_command_handler_is_noop(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        update = _make_update(user_id=42, text="/plan", chat_id=99)
        await connector._on_command(update, MagicMock())

        mock_app.bot.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_message_is_noop(self, connector):
        update = MagicMock()
        update.message = None
        handler = AsyncMock()
        connector.set_command_handler(handler)

        await connector._on_command(update, MagicMock())
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_command_handler_error_caught(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(side_effect=RuntimeError("boom"))
        connector.set_command_handler(handler)

        update = _make_update(user_id=42, text="/plan", chat_id=99)
        await connector._on_command(update, MagicMock())
        # Should not raise

    @pytest.mark.asyncio
    async def test_empty_response_not_sent(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        handler = AsyncMock(return_value="")
        connector.set_command_handler(handler)

        update = _make_update(user_id=42, text="/plan", chat_id=99)
        await connector._on_command(update, MagicMock())

        mock_app.bot.send_message.assert_not_awaited()


class TestClearPlanMessages:
    @pytest.mark.asyncio
    async def test_clears_tracked_plan_messages(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        connector._plan_message_ids["123"] = ["10", "11", "12"]

        await connector.clear_plan_messages("123")

        assert mock_app.bot.delete_message.await_count == 3
        mock_app.bot.delete_message.assert_any_await(chat_id=123, message_id=10)
        mock_app.bot.delete_message.assert_any_await(chat_id=123, message_id=11)
        mock_app.bot.delete_message.assert_any_await(chat_id=123, message_id=12)
        assert "123" not in connector._plan_message_ids

    @pytest.mark.asyncio
    async def test_no_tracked_messages_is_noop(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app

        await connector.clear_plan_messages("123")

        mock_app.bot.delete_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_app_is_noop(self, connector):
        connector._plan_message_ids["123"] = ["10", "11"]

        await connector.clear_plan_messages("123")  # should not raise


class TestSendPlanReviewLongContent:
    @pytest.mark.asyncio
    async def test_short_description_sends_plan_then_buttons(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        mock_app.bot.send_message.return_value = mock_msg

        await connector.send_plan_review("123", "abc-123", "Short plan.")

        # Plan text message + review message with buttons = 2 calls
        assert mock_app.bot.send_message.await_count == 2
        # Last call has buttons
        last_call = mock_app.bot.send_message.await_args_list[-1]
        assert last_call.kwargs["reply_markup"] is not None
        # Verify new button labels
        markup = last_call.kwargs["reply_markup"]
        button_texts = [btn.text for row in markup.inline_keyboard for btn in row]
        assert "Yes, auto-accept edits" in button_texts
        assert "Yes, clear context and auto-accept edits" in button_texts
        assert "Yes, manually approve edits" in button_texts
        assert "Adjust the plan" in button_texts

    @pytest.mark.asyncio
    async def test_long_description_split_into_chunks_and_buttons(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        mock_app.bot.send_message.return_value = mock_msg

        long_desc = "x" * 5000
        await connector.send_plan_review("123", "abc-123", long_desc)

        # Plan chunks + review message with buttons
        calls = mock_app.bot.send_message.await_args_list
        assert len(calls) >= 3  # 2 plan chunks + 1 review with buttons
        # Last call should have buttons
        last_call = calls[-1]
        assert last_call.kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_plan_message_ids_tracked(self, connector):
        mock_app = _make_mock_app()
        connector._app = mock_app
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        mock_app.bot.send_message.return_value = mock_msg

        await connector.send_plan_review("123", "abc-123", "Plan text")

        assert "123" in connector._plan_message_ids
        assert len(connector._plan_message_ids["123"]) >= 2  # plan msg + review msg
