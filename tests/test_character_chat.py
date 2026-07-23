from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai.base import AIResult, AIUsage
from domains.characters import manager as manager_module
from domains.characters.manager import character_chat_manager
from domains.characters.models import CharacterConversation
from domains.characters.schemas import CharacterAIResponse


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


def test_messi_is_clearly_marked_as_ai_roleplay() -> None:
    character = character_chat_manager.get_character("messi")

    assert "fictional AI" in character.disclaimer
    assert "strictly in English" in character.instructions
    assert "one to three short sentences" in character.instructions
