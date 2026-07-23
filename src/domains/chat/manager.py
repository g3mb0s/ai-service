import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager, AIResult, AIUsage
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError, EntityNotFoundError
from domains.chat.models import ChatMessage, Conversation
from domains.chat.service import chat_service
from domains.quota.service import token_quota_service


CHAT_INSTRUCTIONS = (
    "You are Gembos AI, an English-language tutor. The sole purpose of every "
    "conversation is to help the user learn, practise, and improve English. "
    "Keep the conversation focused on English grammar, vocabulary, pronunciation, "
    "reading, writing, listening, speaking, translation, exam preparation, and "
    "English-language cultural context. If a request is unrelated, briefly explain "
    "that you specialise in learning English and offer to turn the topic into an "
    "English lesson or exercise. "
    "Explain difficult concepts in the user's language, but always include useful "
    "English words, phrases, examples, or practice when answering a learning "
    "question. Do not translate all English content unless the user asks. "
    "Encourage the user to produce English themselves. When they write in English, "
    "correct meaningful mistakes tactfully, show a natural corrected version, and "
    "briefly explain the correction. Adapt vocabulary, sentence length, and the "
    "amount of English to the user's apparent level and gradually increase it. "
    "Be accurate, supportive, concise, and never invent language rules."
)


@dataclass(slots=True)
class PreparedChatMessage:
    conversation: Conversation
    user_message: ChatMessage
    user_role: str | None
    openai_input: list[dict[str, str]]
    max_output_tokens: int | None


@dataclass(slots=True)
class ChatStreamEvent:
    delta: str | None = None
    messages: tuple[ChatMessage, ChatMessage] | None = None


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
        prepared = await self.prepare_message(
            session,
            conversation_id,
            user_id,
            user_role,
            content,
        )
        try:
            ai_result = await ai_manager.chat(
                client=openai_client,
                messages=prepared.openai_input,
                instructions=CHAT_INSTRUCTIONS,
                safety_identifier=self._safety_identifier(user_id),
                max_output_tokens=prepared.max_output_tokens,
            )
        except openai.APIError as exc:
            await session.rollback()
            raise AIProviderError(self._provider_error(ai_manager, exc)) from exc
        except AIProviderError:
            await session.rollback()
            raise

        return await self.complete_message(session, prepared, ai_result)

    async def prepare_message(
        self,
        session: AsyncSession,
        conversation_id: UUID,
        user_id: UUID,
        user_role: str | None,
        content: str,
    ) -> PreparedChatMessage:
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
        return PreparedChatMessage(
            conversation=conversation,
            user_message=user_message,
            user_role=user_role,
            openai_input=openai_input,
            max_output_tokens=max_output_tokens,
        )

    async def stream_prepared_message(
        self,
        session: AsyncSession,
        openai_client: AsyncOpenAI,
        ai_manager: AIManager,
        user_id: UUID,
        prepared: PreparedChatMessage,
    ) -> AsyncIterator[ChatStreamEvent]:
        final_result: AIResult[str] | None = None
        streamed_parts: list[str] = []
        try:
            async for event in ai_manager.stream_chat(
                client=openai_client,
                messages=prepared.openai_input,
                instructions=CHAT_INSTRUCTIONS,
                safety_identifier=self._safety_identifier(user_id),
                max_output_tokens=prepared.max_output_tokens,
            ):
                if event.delta:
                    streamed_parts.append(event.delta)
                    yield ChatStreamEvent(delta=event.delta)
                if event.result:
                    final_result = event.result
            if final_result is None:
                raise AIProviderError("AI provider did not finish the streamed response")
            messages = await self.complete_message(session, prepared, final_result)
            yield ChatStreamEvent(messages=messages)
        except (asyncio.CancelledError, GeneratorExit):
            interrupted_result = AIResult(
                data="".join(streamed_parts).strip() or "Генерация была прервана.",
                provider=ai_manager.provider,
                provider_host=settings.openai_base_url,
                model=settings.openai_model,
                response_id=None,
                usage=AIUsage(),
            )
            try:
                await asyncio.shield(
                    self.complete_message(session, prepared, interrupted_result)
                )
            except Exception:
                await asyncio.shield(session.rollback())
            raise
        except openai.APIError as exc:
            await session.rollback()
            raise AIProviderError(self._provider_error(ai_manager, exc)) from exc
        except AIProviderError:
            await session.rollback()
            raise
        except Exception:
            await session.rollback()
            raise

    async def complete_message(
        self,
        session: AsyncSession,
        prepared: PreparedChatMessage,
        ai_result: AIResult[str],
    ) -> tuple[ChatMessage, ChatMessage]:
        assistant_message = ChatMessage(
            conversation_id=prepared.conversation.id,
            role="assistant",
            content=ai_result.data,
            provider=ai_result.provider,
            provider_host=ai_result.provider_host,
            model=ai_result.model,
            provider_response_id=ai_result.response_id,
            input_tokens=ai_result.usage.input_tokens,
            output_tokens=ai_result.usage.output_tokens,
            total_tokens=self._accounted_total_tokens(
                prepared.user_role,
                ai_result.usage.total_tokens,
                prepared.openai_input,
                prepared.max_output_tokens,
            ),
        )
        chat_service.add_message(session, assistant_message)

        if (
            not prepared.conversation.messages
            and prepared.conversation.title == "Новый диалог"
        ):
            prepared.conversation.title = prepared.user_message.content[:80]

        await session.flush()
        await session.commit()
        return prepared.user_message, assistant_message

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
