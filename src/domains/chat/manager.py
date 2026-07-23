import hashlib
from uuid import UUID

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError, EntityNotFoundError
from domains.chat.models import ChatMessage, Conversation
from domains.chat.service import chat_service
from domains.quota.service import token_quota_service


CHAT_INSTRUCTIONS = (
    "You are a helpful learning assistant. Answer in the user's language. "
    "Be accurate, clear, and adapt explanations to the conversation context."
)


class ChatManager:
    async def create_conversation(
        self, session: AsyncSession, user_id: UUID, title: str | None
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=(title or "Новый диалог").strip() or "Новый диалог",
        )
        chat_service.add_conversation(session, conversation)
        await session.flush()
        await session.commit()
        return conversation

    async def list_conversations(
        self, session: AsyncSession, user_id: UUID
    ) -> list[Conversation]:
        return await chat_service.list_conversations(session, user_id)

    async def get_conversation(
        self, session: AsyncSession, conversation_id: UUID, user_id: UUID
    ) -> Conversation:
        conversation = await chat_service.get_conversation(
            session, conversation_id, user_id
        )
        if conversation is None:
            raise EntityNotFoundError("Conversation not found")
        return conversation

    async def send_message(
        self,
        session: AsyncSession,
        openai_client: AsyncOpenAI,
        ai_manager: AIManager,
        conversation_id: UUID,
        user_id: UUID,
        user_role: str | None,
        content: str,
    ) -> tuple[ChatMessage, ChatMessage]:
        conversation = await self.get_conversation(session, conversation_id, user_id)
        clean_content = content.strip()
        user_message = ChatMessage(
            conversation_id=conversation.id,
            role="user",
            content=clean_content,
        )
        chat_service.add_message(session, user_message)

        history = conversation.messages[-settings.chat_history_limit :]
        openai_input = [
            {"role": message.role, "content": message.content} for message in history
        ]
        openai_input.append({"role": "user", "content": clean_content})
        max_output_tokens = await self._get_max_output_tokens(
            session,
            user_id,
            user_role,
            openai_input,
        )

        try:
            ai_result = await ai_manager.chat(
                client=openai_client,
                messages=openai_input,
                instructions=CHAT_INSTRUCTIONS,
                safety_identifier=self._safety_identifier(user_id),
                max_output_tokens=max_output_tokens,
            )
        except openai.APIError as exc:
            await session.rollback()
            raise AIProviderError(self._provider_error(ai_manager, exc)) from exc
        except AIProviderError:
            await session.rollback()
            raise

        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_result.data,
            provider=ai_result.provider,
            provider_host=ai_result.provider_host,
            model=ai_result.model,
            provider_response_id=ai_result.response_id,
            input_tokens=ai_result.usage.input_tokens,
            output_tokens=ai_result.usage.output_tokens,
            total_tokens=self._accounted_total_tokens(
                user_role,
                ai_result.usage.total_tokens,
                openai_input,
                max_output_tokens,
            ),
        )
        chat_service.add_message(session, assistant_message)

        if not conversation.messages and conversation.title == "Новый диалог":
            conversation.title = clean_content[:80]

        await session.flush()
        await session.commit()
        return user_message, assistant_message

    async def _get_max_output_tokens(
        self,
        session: AsyncSession,
        user_id: UUID,
        user_role: str | None,
        messages: list[dict[str, str]],
    ) -> int | None:
        remaining = await token_quota_service.get_remaining_tokens(
            session,
            user_id,
            user_role,
        )
        if remaining is None:
            return None

        # A UTF-8 byte count is a conservative tokenizer-independent upper
        # estimate. The fixed reserve covers message framing and instructions.
        estimated_input_tokens = self._estimate_input_tokens(messages)
        max_output_tokens = min(
            remaining - estimated_input_tokens,
            settings.user_max_output_tokens_per_request,
        )
        if max_output_tokens <= 0:
            raise token_quota_service.limit_error()
        return max_output_tokens

    def _estimate_input_tokens(self, messages: list[dict[str, str]]) -> int:
        text_bytes = len(CHAT_INSTRUCTIONS.encode())
        text_bytes += sum(
            len(message["role"].encode()) + len(message["content"].encode())
            for message in messages
        )
        return text_bytes + 512

    def _accounted_total_tokens(
        self,
        user_role: str | None,
        reported_total: int | None,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
    ) -> int | None:
        if reported_total is not None or user_role != "user":
            return reported_total
        return self._estimate_input_tokens(messages) + (max_output_tokens or 0)

    def _safety_identifier(self, user_id: UUID) -> str:
        raw = f"{settings.openai_safety_salt}:{user_id}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _provider_error(self, ai_manager: AIManager, exc: openai.APIError) -> str:
        if isinstance(exc, openai.APIStatusError):
            return f"{ai_manager.provider} returned HTTP {exc.status_code}: {exc}"
        return f"{ai_manager.provider} request failed: {exc}"


chat_manager = ChatManager()
