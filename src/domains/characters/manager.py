import hashlib
import json
from uuid import UUID

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError, EntityNotFoundError
from domains.characters.models import Character, CharacterConversation, CharacterMessage
from domains.characters.schemas import (
    CharacterAIResponse,
    CharacterCreateRequest,
    CharacterUpdateRequest,
)
from domains.characters.service import character_chat_service
from domains.quota.service import token_quota_service


class CharacterChatManager:
    async def list_characters(
        self,
        session: AsyncSession,
        *,
        active_only: bool = True,
    ) -> list[Character]:
        return await character_chat_service.list_characters(
            session,
            active_only=active_only,
        )

    async def get_character(
        self,
        session: AsyncSession,
        character_id: str,
        *,
        active_only: bool = False,
    ) -> Character:
        character = await character_chat_service.get_character(
            session,
            character_id,
            active_only=active_only,
        )
        if character is None:
            raise EntityNotFoundError("Character not found")
        return character

    async def create_character(
        self,
        session: AsyncSession,
        payload: CharacterCreateRequest,
    ) -> Character:
        existing = await character_chat_service.get_character(session, payload.id)
        if existing is not None:
            raise ValueError("A character with this ID already exists")
        character = Character(**payload.model_dump())
        character_chat_service.add_character(session, character)
        await session.flush()
        await session.commit()
        return character

    async def update_character(
        self,
        session: AsyncSession,
        character_id: str,
        payload: CharacterUpdateRequest,
    ) -> Character:
        character = await self.get_character(session, character_id)
        for field, value in payload.model_dump().items():
            setattr(character, field, value)
        await session.flush()
        await session.refresh(character)
        await session.commit()
        return character

    async def update_character_avatar(
        self,
        session: AsyncSession,
        character: Character,
        *,
        avatar_url: str | None,
        avatar_object_key: str | None,
    ) -> Character:
        character.avatar_url = avatar_url
        character.avatar_object_key = avatar_object_key
        await session.flush()
        await session.refresh(character)
        await session.commit()
        return character

    async def delete_character(
        self,
        session: AsyncSession,
        character_id: str,
    ) -> None:
        character = await self.get_character(session, character_id)
        conversation_count = await character_chat_service.count_conversations(
            session,
            character_id,
        )
        if conversation_count:
            raise ValueError(
                "Character has conversations and cannot be deleted; deactivate it instead"
            )
        await character_chat_service.delete_character(session, character)
        await session.commit()

    async def create_conversation(
        self,
        session: AsyncSession,
        character_id: str,
        user_id: UUID,
    ) -> CharacterConversation:
        character = await self.get_character(session, character_id, active_only=True)
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
        await self.get_character(session, character_id, active_only=True)
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
        await self.get_character(session, conversation.character_id)
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
        character = await self.get_character(session, conversation.character_id)
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

character_chat_manager = CharacterChatManager()
