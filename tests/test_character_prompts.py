from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai.base import AIResult, AIUsage
from domains.characters import manager as manager_module
from domains.characters.manager import (
    CHARACTER_BASE_INSTRUCTIONS,
    character_chat_manager,
    compose_character_instructions,
)
from domains.characters.models import Character, CharacterConversation
from domains.characters.schemas import (
    CharacterAIResponse,
    CharacterCreateRequest,
    CharacterPromptFields,
    CharacterUpdateRequest,
)
from basic_utils.exceptions import AIProviderError


def _character(
    *,
    character_id: str = "santa",
    instructions: str | None = None,
    character_prompt: str | None = None,
    is_active: bool = True,
) -> Character:
    return Character(
        id=character_id,
        name="Santa Claus",
        description="Christmas practice",
        greeting="Hello!",
        instructions=instructions,
        character_prompt=character_prompt,
        is_active=is_active,
    )


def _create_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "santa",
        "name": "Santa Claus",
        "description": "Christmas practice",
        "greeting": "Hello!",
        "character_prompt": "A cheerful Christmas gift-giver in a red suit.",
        "is_active": True,
    }
    payload.update(overrides)
    return payload


def _update_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "Santa Claus",
        "description": "Christmas practice",
        "greeting": "Hello!",
        "is_active": True,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------- compose ----


def test_compose_uses_base_plus_character_prompt() -> None:
    character = _character(
        character_prompt="A cheerful Christmas gift-giver in a red suit."
    )

    assert compose_character_instructions(character) == (
        CHARACTER_BASE_INSTRUCTIONS
        + "\n\nCharacter: A cheerful Christmas gift-giver in a red suit."
    )


def test_compose_falls_back_to_legacy_instructions() -> None:
    character = _character(
        instructions="Legacy full prompt stored in the database.",
        character_prompt=None,
    )

    assert compose_character_instructions(character) == (
        "Legacy full prompt stored in the database."
    )


def test_compose_returns_empty_string_when_both_missing() -> None:
    character = _character(instructions=None, character_prompt=None)

    assert compose_character_instructions(character) == ""


# ------------------------------------------------------------ coercion --------


def test_empty_and_whitespace_prompts_normalise_to_none() -> None:
    assert CharacterPromptFields(instructions="").instructions is None
    assert CharacterPromptFields(instructions="   ").instructions is None
    assert CharacterPromptFields(character_prompt="").character_prompt is None
    assert CharacterPromptFields(character_prompt=" \t ").character_prompt is None


def test_none_prompts_stay_none() -> None:
    assert CharacterPromptFields(instructions=None).instructions is None
    assert CharacterPromptFields(character_prompt=None).character_prompt is None


def test_non_string_prompt_is_coerced_not_rejected() -> None:
    fields = CharacterPromptFields(character_prompt=123)

    assert fields.character_prompt == "123"


def test_overlong_character_prompt_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CharacterCreateRequest.model_validate(
            _create_payload(character_prompt="x" * 2_001)
        )


# --------------------------------------------------------------- create -------


def test_create_accepts_only_character_prompt() -> None:
    request = CharacterCreateRequest.model_validate(
        _create_payload(character_prompt="A cheerful Christmas gift-giver.")
    )

    assert request.character_prompt == "A cheerful Christmas gift-giver."
    assert request.instructions is None


def test_create_rejects_both_prompt_fields_empty() -> None:
    with pytest.raises(ValidationError, match="character_prompt or instructions is required"):
        CharacterCreateRequest.model_validate(
            _create_payload(character_prompt=None, instructions=None)
        )


def test_create_rejects_both_prompt_fields_set() -> None:
    with pytest.raises(ValidationError, match="provide either character_prompt or instructions, not both"):
        CharacterCreateRequest.model_validate(
            _create_payload(
                character_prompt="A cheerful gift-giver.",
                instructions="Legacy prompt that is long enough to pass min_length.",
            )
        )


# --------------------------------------------------------------- update -------


@pytest.mark.asyncio
async def test_update_with_both_none_keeps_legacy_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character = _character(instructions="Original legacy prompt.", character_prompt=None)
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
        character.id,
        CharacterUpdateRequest.model_validate(_update_payload()),
    )

    assert result.instructions == "Original legacy prompt."
    assert result.character_prompt is None
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_with_character_prompt_clears_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character = _character(
        instructions="Original legacy prompt.",
        character_prompt=None,
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
        character.id,
        CharacterUpdateRequest.model_validate(
            _update_payload(character_prompt="A cheerful gift-giver.")
        ),
    )

    assert result.character_prompt == "A cheerful gift-giver."
    assert result.instructions is None


@pytest.mark.asyncio
async def test_update_with_instructions_clears_character_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character = _character(
        instructions=None,
        character_prompt="A cheerful gift-giver.",
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
        character.id,
        CharacterUpdateRequest.model_validate(
            _update_payload(instructions="Legacy prompt with enough characters.")
        ),
    )

    assert result.instructions == "Legacy prompt with enough characters."
    assert result.character_prompt is None


def test_update_rejects_both_prompt_fields_set() -> None:
    with pytest.raises(ValidationError, match="provide either character_prompt or instructions, not both"):
        CharacterUpdateRequest.model_validate(
            _update_payload(
                character_prompt="A cheerful gift-giver.",
                instructions="Legacy prompt that is long enough to pass min_length.",
            )
        )


def test_update_rejects_too_short_instructions() -> None:
    with pytest.raises(ValidationError):
        CharacterUpdateRequest.model_validate(
            _update_payload(instructions="short")
        )


# ---------------------------------------------------------- send_message -------


@pytest.mark.asyncio
async def test_send_message_uses_composed_instructions(
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
        character_id="santa",
        title="Chat with Santa",
        messages=[],
    )
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "get_conversation",
        AsyncMock(return_value=conversation),
    )
    character = _character(
        character_prompt="A cheerful Christmas gift-giver in a red suit."
    )
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "get_character",
        AsyncMock(return_value=character),
    )
    monkeypatch.setattr(
        manager_module.character_chat_service,
        "add_message",
        MagicMock(),
    )

    expected = (
        CHARACTER_BASE_INSTRUCTIONS
        + "\n\nCharacter: A cheerful Christmas gift-giver in a red suit."
    )
    captured: dict[str, str] = {}

    async def fake_max_output_tokens(
        session, user_id, user_role, instructions, input_text
    ):
        captured["max_output"] = instructions
        return 500

    def fake_accounted_tokens(
        user_role, reported_total, instructions, input_text, max_output_tokens
    ):
        captured["accounted"] = instructions
        return reported_total

    monkeypatch.setattr(
        character_chat_manager, "_max_output_tokens", fake_max_output_tokens
    )
    monkeypatch.setattr(
        character_chat_manager, "_accounted_tokens", fake_accounted_tokens
    )

    ai_manager = SimpleNamespace(
        provider="test-provider",
        generate_structured=AsyncMock(
            return_value=AIResult(
                data=CharacterAIResponse.model_validate(
                    {
                        "text": "Ho ho ho! What would you like to talk about?",
                        "rate": {
                            "quality": 10,
                            "correction": "",
                            "comment": "",
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

    await character_chat_manager.send_message(
        session,
        SimpleNamespace(),
        ai_manager,
        conversation.id,
        user_id,
        "user",
        "Merry Christmas!",
    )

    assert (
        ai_manager.generate_structured.await_args.kwargs["instructions"] == expected
    )
    assert captured["max_output"] == expected
    assert captured["accounted"] == expected
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_message_raises_when_no_prompt_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    user_id = uuid4()
    conversation = CharacterConversation(
        id=uuid4(),
        user_id=user_id,
        character_id="ghost",
        title="Chat with Ghost",
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
            return_value=_character(
                character_id="ghost",
                instructions=None,
                character_prompt=None,
            )
        ),
    )

    with pytest.raises(AIProviderError, match="Character has no prompt configured"):
        await character_chat_manager.send_message(
            session,
            SimpleNamespace(),
            SimpleNamespace(generate_structured=AsyncMock()),
            conversation.id,
            user_id,
            "user",
            "Hello?",
        )
