import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError, EntityNotFoundError
from domains.characters.models import CharacterConversation, CharacterMessage
from domains.characters.schemas import CharacterAIResponse, CharacterResponse
from domains.characters.service import character_chat_service
from domains.quota.service import token_quota_service


@dataclass(frozen=True, slots=True)
class CharacterDefinition:
    id: str
    name: str
    description: str
    greeting: str
    disclaimer: str
    instructions: str


MESSI = CharacterDefinition(
    id="messi",
    name="Lionel Messi",
    description="Practise everyday English with a calm football legend.",
    greeting="Hi! Let’s talk in English. You can ask me about football, training, goals, or daily life.",
    disclaimer="This is a fictional AI roleplay, not Lionel Messi or his representative.",
    instructions=(
        "Roleplay as a clearly fictional AI character inspired by Lionel Messi's "
        "widely known public football persona. Never claim to be the real person, "
        "never invent private information, and do not imply endorsement. "
        "The purpose is relaxed English conversation practice. Communicate strictly "
        "in English, even when the user writes in another language. Stay warm, calm, "
        "humble, and interested in football, training, teamwork, family-friendly daily "
        "life, and motivation. Keep the character reply in text to one to three short "
        "sentences. Ask at most one natural follow-up question. "
        "Evaluate only the English quality of the user's latest message. quality must "
        "be an integer from 0 to 10. If the message is natural and correct, use 10 and "
        "set correction and comment to empty strings. If it contains any meaningful "
        "grammar, word-choice, spelling, or naturalness issue, use 0-9, put the full "
        "natural corrected message in correction, and give a short grammar explanation "
        "in English in comment. If no correction is needed, correction and comment must "
        "both be empty. Never fill only one of those fields. The comment must be no more "
        "than three short sentences. "
        "Return only the structured object required by the schema."
    ),
)

CHARACTERS = {MESSI.id: MESSI}


class CharacterChatManager:
    def list_characters(self) -> list[CharacterResponse]:
        return [self._response(character) for character in CHARACTERS.values()]

    def get_character(self, character_id: str) -> CharacterDefinition:
        character = CHARACTERS.get(character_id)
        if character is None:
            raise EntityNotFoundError("Character not found")
        return character

    async def create_conversation(
        self,
        session: AsyncSession,
        character_id: str,
        user_id: UUID,
    ) -> CharacterConversation:
        character = self.get_character(character_id)
        conversation = CharacterConversation(
            user_id=user_id,
            character_id=character.id,
            title=f"Chat with {character.name}",
        )
        character_chat_service.add_conversation(session, conversation)
        await session.flush()
        await session.commit()
        return conversation

    async def list_conversations(
        self,
        session: AsyncSession,
        character_id: str,
        user_id: UUID,
    ) -> list[CharacterConversation]:
        self.get_character(character_id)
        return await character_chat_service.list_conversations(
            session,
            character_id,
            user_id,
        )

    async def get_conversation(
        self,
        session: AsyncSession,
        conversation_id: UUID,
        user_id: UUID,
    ) -> CharacterConversation:
        conversation = await character_chat_service.get_conversation(
            session,
            conversation_id,
            user_id,
        )
        if conversation is None:
            raise EntityNotFoundError("Character conversation not found")
        self.get_character(conversation.character_id)
        return conversation

    async def send_message(
        self,
        session: AsyncSession,
        client: AsyncOpenAI,
        ai_manager: AIManager,
        conversation_id: UUID,
        user_id: UUID,
        user_role: str | None,
        content: str,
    ) -> tuple[CharacterMessage, CharacterMessage]:
        conversation = await self.get_conversation(session, conversation_id, user_id)
        character = self.get_character(conversation.character_id)
        clean_content = content.strip()
        user_message = CharacterMessage(
            conversation_id=conversation.id,
            role="user",
            content=clean_content,
        )
        character_chat_service.add_message(session, user_message)

        history = conversation.messages[-settings.chat_history_limit :]
        input_payload = {
            "conversation": [
                {"role": message.role, "content": message.content}
                for message in history
            ],
            "latest_user_message": clean_content,
        }
        input_text = json.dumps(input_payload, ensure_ascii=False)
        max_output_tokens = await self._max_output_tokens(
            session,
            user_id,
            user_role,
            character.instructions,
            input_text,
        )

        try:
            result = await ai_manager.generate_structured(
                client=client,
                input_text=input_text,
                instructions=character.instructions,
                response_model=CharacterAIResponse,
                safety_identifier=self._safety_identifier(user_id),
                max_output_tokens=max_output_tokens,
            )
        except openai.APIError as exc:
            await session.rollback()
            raise AIProviderError(self._provider_error(ai_manager, exc)) from exc
        except AIProviderError:
            await session.rollback()
            raise

        user_message.quality = result.data.rate.quality
        user_message.correction = result.data.rate.correction or None
        user_message.comment = result.data.rate.comment or None
        assistant_message = CharacterMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=result.data.text,
            provider=result.provider,
            provider_host=result.provider_host,
            model=result.model,
            provider_response_id=result.response_id,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=self._accounted_tokens(
                user_role,
                result.usage.total_tokens,
                character.instructions,
                input_text,
                max_output_tokens,
            ),
        )
        character_chat_service.add_message(session, assistant_message)

        if not conversation.messages:
            conversation.title = clean_content[:80]

        await session.flush()
        await session.commit()
        return user_message, assistant_message

    async def _max_output_tokens(
        self,
        session: AsyncSession,
        user_id: UUID,
        user_role: str | None,
        instructions: str,
        input_text: str,
    ) -> int:
        remaining = await token_quota_service.get_remaining_tokens(
            session,
            user_id,
            user_role,
        )
        configured_max = settings.character_response_max_output_tokens
        if remaining is None:
            return configured_max
        allowed = min(
            configured_max,
            remaining - self._estimate_input_tokens(instructions, input_text),
        )
        if allowed <= 0:
            raise token_quota_service.limit_error()
        return allowed

    def _estimate_input_tokens(self, instructions: str, input_text: str) -> int:
        return len(instructions.encode()) + len(input_text.encode()) + 512

    def _accounted_tokens(
        self,
        user_role: str | None,
        reported_total: int | None,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
    ) -> int | None:
        if reported_total is not None or user_role != "user":
            return reported_total
        return (
            self._estimate_input_tokens(instructions, input_text)
            + max_output_tokens
        )

    def _safety_identifier(self, user_id: UUID) -> str:
        raw = f"{settings.openai_safety_salt}:character:{user_id}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _provider_error(self, ai_manager: AIManager, exc: openai.APIError) -> str:
        if isinstance(exc, openai.APIStatusError):
            return f"{ai_manager.provider} returned HTTP {exc.status_code}: {exc}"
        return f"{ai_manager.provider} request failed: {exc}"

    def _response(self, character: CharacterDefinition) -> CharacterResponse:
        return CharacterResponse(
            id="messi",
            name=character.name,
            description=character.description,
            greeting=character.greeting,
            disclaimer=character.disclaimer,
        )


character_chat_manager = CharacterChatManager()
