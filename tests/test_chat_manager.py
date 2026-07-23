from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai.base import AIResult, AIStreamEvent, AIUsage
from domains.chat import manager as manager_module
from domains.chat.manager import CHAT_INSTRUCTIONS, PreparedChatMessage, chat_manager
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


def test_chat_prompt_is_scoped_to_learning_english() -> None:
    assert "sole purpose" in CHAT_INSTRUCTIONS
    assert "learn, practise, and improve English" in CHAT_INSTRUCTIONS
    assert "always include useful English" in CHAT_INSTRUCTIONS
    assert "unrelated" in CHAT_INSTRUCTIONS


@pytest.mark.asyncio
async def test_streamed_message_is_persisted_after_final_event() -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    conversation = Conversation(
        id=uuid4(),
        user_id=uuid4(),
        title="Новый диалог",
        messages=[],
    )
    prepared = PreparedChatMessage(
        conversation=conversation,
        user_message=manager_module.ChatMessage(
            conversation_id=conversation.id,
            role="user",
            content="Question",
        ),
        user_role="user",
        openai_input=[{"role": "user", "content": "Question"}],
        max_output_tokens=500,
    )
    final_result = AIResult(
        data="Streamed answer",
        provider="test-provider",
        provider_host="https://provider.test",
        model="test-model",
        response_id="resp_stream",
        usage=AIUsage(input_tokens=4, output_tokens=2, total_tokens=6),
    )

    async def stream_chat(**_kwargs):
        yield AIStreamEvent(delta="Streamed ")
        yield AIStreamEvent(delta="answer")
        yield AIStreamEvent(result=final_result)

    provider_manager = SimpleNamespace(
        provider="test-provider",
        stream_chat=stream_chat,
    )
    add_message = MagicMock()
    original_add_message = manager_module.chat_service.add_message
    manager_module.chat_service.add_message = add_message
    try:
        events = [
            event
            async for event in chat_manager.stream_prepared_message(
                session,
                SimpleNamespace(),
                provider_manager,
                conversation.user_id,
                prepared,
            )
        ]
    finally:
        manager_module.chat_service.add_message = original_add_message

    assert [event.delta for event in events[:2]] == ["Streamed ", "answer"]
    assert events[-1].messages is not None
    assert events[-1].messages[1].content == "Streamed answer"
    add_message.assert_called_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupted_stream_is_persisted_and_accounted() -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    conversation = Conversation(
        id=uuid4(),
        user_id=uuid4(),
        title="Новый диалог",
        messages=[],
    )
    prepared = PreparedChatMessage(
        conversation=conversation,
        user_message=manager_module.ChatMessage(
            conversation_id=conversation.id,
            role="user",
            content="Question",
        ),
        user_role="user",
        openai_input=[{"role": "user", "content": "Question"}],
        max_output_tokens=500,
    )

    async def stream_chat(**_kwargs):
        yield AIStreamEvent(delta="Partial answer")
        yield AIStreamEvent(delta=" that should not arrive")

    provider_manager = SimpleNamespace(
        provider="test-provider",
        stream_chat=stream_chat,
    )
    add_message = MagicMock()
    original_add_message = manager_module.chat_service.add_message
    manager_module.chat_service.add_message = add_message
    try:
        stream = chat_manager.stream_prepared_message(
            session,
            SimpleNamespace(),
            provider_manager,
            conversation.user_id,
            prepared,
        )
        first = await anext(stream)
        await stream.aclose()
    finally:
        manager_module.chat_service.add_message = original_add_message

    assert first.delta == "Partial answer"
    assistant_message = add_message.call_args.args[1]
    assert assistant_message.content == "Partial answer"
    assert assistant_message.total_tokens > 500
    session.commit.assert_awaited_once()
