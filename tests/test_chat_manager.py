from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai.base import AIResult, AIUsage
from domains.chat import manager as manager_module
from domains.chat.manager import chat_manager
from domains.chat.models import Conversation


@pytest.mark.asyncio
async def test_send_message_persists_both_messages_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    provider_manager = SimpleNamespace(
        provider="test-provider",
        chat=AsyncMock(
            return_value=AIResult(
                data="Готовый ответ",
                provider="test-provider",
                provider_host="https://provider.test",
                model="test-model",
                response_id="resp_123",
                usage=AIUsage(input_tokens=12, output_tokens=7, total_tokens=19),
            )
        ),
    )
    client = SimpleNamespace()
    user_id = uuid4()
    conversation = Conversation(
        id=uuid4(), user_id=user_id, title="Новый диалог", messages=[]
    )
    get_conversation = AsyncMock(return_value=conversation)
    add_message = MagicMock()
    get_remaining_tokens = AsyncMock(return_value=10_000)
    monkeypatch.setattr(manager_module.chat_service, "get_conversation", get_conversation)
    monkeypatch.setattr(manager_module.chat_service, "add_message", add_message)
    monkeypatch.setattr(
        manager_module.token_quota_service,
        "get_remaining_tokens",
        get_remaining_tokens,
    )

    user_message, assistant_message = await chat_manager.send_message(
        session,
        client,
        provider_manager,
        conversation.id,
        user_id,
        "user",
        "  Объясни времена  ",
    )

    assert user_message.content == "Объясни времена"
    assert assistant_message.content == "Готовый ответ"
    assert assistant_message.provider_response_id == "resp_123"
    assert assistant_message.model == "test-model"
    assert assistant_message.total_tokens == 19
    assert conversation.title == "Объясни времена"
    assert add_message.call_count == 2
    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()
    provider_manager.chat.assert_awaited_once()
    get_remaining_tokens.assert_awaited_once_with(session, user_id, "user")
    max_output_tokens = provider_manager.chat.await_args.kwargs["max_output_tokens"]
    assert 0 < max_output_tokens <= 8_000


@pytest.mark.asyncio
async def test_create_conversation_commits_in_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    add_conversation = MagicMock()
    monkeypatch.setattr(
        manager_module.chat_service, "add_conversation", add_conversation
    )

    conversation = await chat_manager.create_conversation(session, uuid4(), "Мой чат")

    assert conversation.title == "Мой чат"
    add_conversation.assert_called_once_with(session, conversation)
    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()


def test_missing_provider_usage_is_conservatively_accounted() -> None:
    messages = [{"role": "user", "content": "Explain articles"}]

    accounted = chat_manager._accounted_total_tokens(
        "user",
        None,
        messages,
        2_000,
    )

    assert accounted is not None
    assert accounted > 2_000
    assert chat_manager._accounted_total_tokens(
        "manager",
        None,
        messages,
        None,
    ) is None
