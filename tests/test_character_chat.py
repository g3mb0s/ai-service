from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai.base import AIResult, AIUsage
from domains.characters import manager as manager_module
from domains.characters.manager import character_chat_manager
from domains.characters.models import Character, CharacterConversation
from domains.characters.schemas import CharacterAIResponse, CharacterUpdateRequest


def test_perfect_message_has_no_feedback() -> None:
    response = CharacterAIResponse.model_validate(
        {
            "text": "That was a great pass! Do you play football?",
            "rate": {"quality": 10, "correction": "", "comment": ""},
        }
    )

    assert response.rate.quality == 10
    assert response.rate.correction == ""


def test_feedback_fields_must_be_filled_together() -> None:
    with pytest.raises(ValidationError):
        CharacterAIResponse.model_validate(
            {
                "text": "I train every day.",
                "rate": {
                    "quality": 6,
                    "correction": "I trained every day.",
                    "comment": "",
                },
            }
        )


def test_character_reply_and_comment_are_trimmed_to_three_sentences() -> None:
    response = CharacterAIResponse.model_validate(
        {
            "text": "One. Two. Three. Four.",
            "rate": {
                "quality": 7,
                "correction": "I like football.",
                "comment": "One. Two. Three. Four.",
            },
        }
    )

    assert response.text == "One. Two. Three."
    assert response.rate.comment == "One. Two. Three."


def test_lower_score_without_grammar_error_can_have_empty_feedback() -> None:
    response = CharacterAIResponse.model_validate(
        {
            "text": "Tell me more about your favourite team.",
            "rate": {"quality": 8, "correction": "", "comment": ""},
        }
    )

    assert response.rate.correction == ""


def test_null_feedback_fields_are_coerced_to_empty_strings() -> None:
    response = CharacterAIResponse.model_validate(
        {
            "text": "I train every day.",
            "rate": {"quality": 6, "correction": None, "comment": None},
        }
    )

    assert response.rate.correction == ""
    assert response.rate.comment == ""


@pytest.mark.asyncio
async def test_character_turn_persists_rating_and_short_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    user_id = uuid4()
    conversation = CharacterConversation(
        id=uuid4(),
        user_id=user_id,
        character_id="messi",
        title="Chat with Lionel Messi",
        messages=[],
    )
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "get_conversation",
        AsyncMock(return_value=conversation),
    )
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "get_character",
        AsyncMock(
            return_value=Character(
                id="messi",
                name="Lionel Messi",
                description="Football practice",
                greeting="Hello",
                instructions="Use the dynamic instructions from the database.",
                is_active=True,
            )
        ),
    )
    add_message = MagicMock()
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "add_message",
        add_message,
    )
    monkeypatch.setattr(
        manager_module.token_quota_service,
        "get_remaining_tokens",
        AsyncMock(return_value=10_000),
    )
    ai_manager = SimpleNamespace(
        provider="test-provider",
        generate_structured=AsyncMock(
            return_value=AIResult(
                data=CharacterAIResponse.model_validate(
                    {
                        "text": "I enjoy training with the team. What position do you play?",
                        "rate": {
                            "quality": 6,
                            "correction": "I played football yesterday.",
                            "comment": "Use the past form “played” with “yesterday”.",
                        },
                    }
                ),
                provider="test-provider",
                provider_host="https://provider.test",
                model="test-model",
                response_id="response-1",
                usage=AIUsage(input_tokens=20, output_tokens=30, total_tokens=50),
            )
        ),
    )

    user_message, assistant_message = await character_chat_manager.send_message(
        session,
        SimpleNamespace(),
        ai_manager,
        conversation.id,
        user_id,
        "user",
        "I play football yesterday",
    )

    assert user_message.quality == 6
    assert user_message.correction == "I played football yesterday."
    assert user_message.comment
    assert assistant_message.content.startswith("I enjoy training")
    assert assistant_message.total_tokens == 50
    assert add_message.call_count == 2
    session.commit.assert_awaited_once()
    assert (
        ai_manager.generate_structured.await_args.kwargs["response_model"]
        is CharacterAIResponse
    )
    assert (
        ai_manager.generate_structured.await_args.kwargs["instructions"]
        == "Use the dynamic instructions from the database."
    )


@pytest.mark.asyncio
async def test_character_prompt_is_loaded_from_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character = Character(
        id="detective",
        name="Detective",
        description="Mystery practice",
        greeting="What did you notice?",
        instructions="Use this editable prompt from the database.",
        is_active=True,
    )
    get_character = AsyncMock(return_value=character)
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "get_character",
        get_character,
    )

    result = await character_chat_manager.get_character(
        MagicMock(),
        "detective",
        active_only=True,
    )

    assert result.instructions == "Use this editable prompt from the database."
    get_character.assert_awaited_once()
    assert get_character.await_args.kwargs["active_only"] is True


@pytest.mark.asyncio
async def test_updated_character_is_refreshed_before_response_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character = Character(
        id="messi",
        name="Lionel Messi",
        description="Football practice",
        greeting="Hello",
        instructions="Use the original database prompt.",
        is_active=True,
    )
    session = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.commit = AsyncMock()
    monkeypatch.setattr(
        character_chat_manager,
        "get_character",
        AsyncMock(return_value=character),
    )

    result = await character_chat_manager.update_character(
        session,
        "messi",
        CharacterUpdateRequest(
            name="Updated Messi",
            description="Updated football practice",
            greeting="Hi!",
            instructions="Use the updated database prompt.",
            is_active=True,
        ),
    )

    assert result.name == "Updated Messi"
    session.flush.assert_awaited_once()
    session.refresh.assert_awaited_once_with(character)
    session.commit.assert_awaited_once()
