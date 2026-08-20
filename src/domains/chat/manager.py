import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager, AIResult, AIStreamToolEvent, AIUsage
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError, DomainError, EntityNotFoundError
from domains.chat.models import ChatMessage, Conversation
from domains.chat.service import chat_service
from domains.quota.service import token_quota_service


GET_USER_WORDS_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_user_words",
        "description": (
            "Получить слова, которые изучает пользователь, вместе с их "
            "статистикой повторений. Фильтры: статус изучения, только "
            "проблемные слова. Вернёт items, total, offset, limit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "new",
                        "learning",
                        "single_review",
                        "recent",
                        "due",
                        "learned",
                        "long_learned",
                        "known",
                    ],
                },
                "errors_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "additionalProperties": False,
        },
    },
}


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
    "Be accurate, supportive, concise, and never invent language rules. "
    "If the user asks for examples with their own words, asks to analyse their "
    "problematic words, or asks to revise what they have already learned, call "
    "the get_user_words tool to fetch the user's words and their review "
    "statistics. Make at most one tool call per reply."
)


@dataclass(slots=True)
class PreparedChatMessage:
    conversation: Conversation
    user_message: ChatMessage
    user_role: str | None
    openai_input: list[dict[str, object]]
    max_output_tokens: int | None


@dataclass(slots=True)
class ChatToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class ChatStreamEvent:
    delta: str | None = None
    messages: tuple[ChatMessage, ChatMessage] | None = None
    tool_call: ChatToolCall | None = None
    note: str | None = None


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
        openai_input = self._serialize_history(
            [self._message_dict(message) for message in history]
        )
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

    async def prepare_tool_result(
        self,
        session: AsyncSession,
        conversation_id: UUID,
        user_id: UUID,
        user_role: str | None,
        tool_call_id: str,
        tool_result: str,
    ) -> PreparedChatMessage:
        conversation = await self.get_conversation(session, conversation_id, user_id)
        committed_call = self._find_committed_tool_call(conversation)
        if committed_call is None or committed_call["id"] != tool_call_id:
            raise DomainError(
                "tool_call_id does not match the committed assistant tool call"
            )

        history = conversation.messages[-settings.chat_history_limit :]
        tool_msg: dict[str, object] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result,
        }
        openai_input = self._serialize_history(
            [self._message_dict(message) for message in history] + [tool_msg]
        )
        user_message = self._last_user_message(conversation)
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
        self._maybe_set_title(prepared)

        await session.flush()
        await session.commit()
        return prepared.user_message, assistant_message

    async def stream_tool_message(
        self,
        session: AsyncSession,
        openai_client: AsyncOpenAI,
        ai_manager: AIManager,
        user_id: UUID,
        prepared: PreparedChatMessage,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Фаза 1: стрим с объявленным тулом get_user_words.

        Если модель ответила текстом — завершаем обычным complete_message +
        ``done``. Если модель вызвала тул — коммитим assistant-сообщение с
        tool_calls до отправки SSE, затем эмитим событие ``tool_call`` без
        ``done``.
        """
        tool_committed = False
        try:
            async for event in ai_manager.stream_tool_chat(
                client=openai_client,
                messages=prepared.openai_input,
                instructions=CHAT_INSTRUCTIONS,
                safety_identifier=self._safety_identifier(user_id),
                tools=[GET_USER_WORDS_TOOL],
                max_output_tokens=prepared.max_output_tokens,
            ):
                if event.delta:
                    yield ChatStreamEvent(delta=event.delta)
                if event.result is not None:
                    messages = await self.complete_message(session, prepared, event.result)
                    yield ChatStreamEvent(messages=messages)
                    return
                if event.tool_call_id is not None:
                    pre_tool_text = event.pre_tool_text or ""
                    await self._commit_tool_call_message(
                        session,
                        prepared,
                        ai_manager,
                        pre_tool_text=pre_tool_text,
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name or "",
                        tool_arguments=event.tool_arguments or "",
                        usage=event.usage or AIUsage(),
                        model=event.model or settings.openai_model,
                        response_id=event.response_id,
                    )
                    # Флаг выставляется только после успешного коммита: при
                    # отмене во время commit транзакция ещё может не завершиться.
                    tool_committed = True
                    yield ChatStreamEvent(
                        tool_call=ChatToolCall(
                            id=event.tool_call_id,
                            name=event.tool_name or "",
                            arguments=event.tool_arguments or "",
                        ),
                        note=self._truncate_note(event.note or pre_tool_text),
                    )
                    return
        except (asyncio.CancelledError, GeneratorExit):
            if tool_committed:
                # Сообщение уже закоммичено; повторно persist-ить не нужно.
                raise
            await session.rollback()
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

    async def _commit_tool_call_message(
        self,
        session: AsyncSession,
        prepared: PreparedChatMessage,
        ai_manager: AIManager,
        pre_tool_text: str,
        tool_call_id: str,
        tool_name: str,
        tool_arguments: str,
        usage: AIUsage,
        model: str,
        response_id: str | None,
    ) -> ChatMessage:
        assistant_message = ChatMessage(
            conversation_id=prepared.conversation.id,
            role="assistant",
            content=pre_tool_text,
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": tool_name,
                    "arguments": tool_arguments,
                }
            ],
            provider=ai_manager.provider,
            provider_host=settings.openai_base_url,
            model=model,
            provider_response_id=response_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=self._accounted_total_tokens(
                prepared.user_role,
                usage.total_tokens,
                prepared.openai_input,
                prepared.max_output_tokens,
            ),
        )
        chat_service.add_message(session, assistant_message)
        self._maybe_set_title(prepared)
        await session.flush()
        await session.commit()
        return assistant_message

    def _maybe_set_title(self, prepared: PreparedChatMessage) -> None:
        if (
            not prepared.conversation.messages
            and prepared.conversation.title == "Новый диалог"
        ):
            prepared.conversation.title = prepared.user_message.content[:80]

    def _message_dict(self, message: ChatMessage) -> dict[str, object]:
        base: dict[str, object] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            base["tool_calls"] = message.tool_calls
        return base

    def _serialize_history(
        self, input_messages: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Сериализует историю в канонический список словарей.

        - assistant с ``tool_calls`` включается только если для каждого call id
          позже в списке есть ``role: "tool"`` с тем же ``tool_call_id``;
          правило «пустой content → skip» к нему не применяется.
        - ``role: "tool"`` включается только если стоит сразу после своего
          assistant(tool_calls) и его ``tool_call_id`` всё ещё ожидается.
        - user/текстовые assistant — как есть, скип только при пустом content.
        """
        future_tool_ids: list[set[str]] = [set() for _ in input_messages]
        seen_later: set[str] = set()
        for index in range(len(input_messages) - 1, -1, -1):
            future_tool_ids[index] = set(seen_later)
            if input_messages[index].get("role") == "tool":
                seen_later.add(str(input_messages[index].get("tool_call_id") or ""))

        output: list[dict[str, object]] = []
        pending_call_ids: set[str] = set()
        last_assistant_call_ids: set[str] = set()
        for index, message in enumerate(input_messages):
            role = message.get("role")
            tool_calls = message.get("tool_calls")
            if role == "assistant" and tool_calls:
                call_ids = {str(call["id"]) for call in tool_calls}
                if call_ids and call_ids <= future_tool_ids[index]:
                    output.append(message)
                    pending_call_ids = set(call_ids)
                    last_assistant_call_ids = set(call_ids)
                else:
                    pending_call_ids = set()
                    last_assistant_call_ids = set()
            elif role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if call_id in last_assistant_call_ids and call_id in pending_call_ids:
                    output.append(message)
                    pending_call_ids.discard(call_id)
                    if not pending_call_ids:
                        last_assistant_call_ids = set()
            elif role in ("user", "assistant"):
                content = message.get("content")
                if isinstance(content, str) and content != "":
                    output.append(message)
                pending_call_ids = set()
                last_assistant_call_ids = set()
            else:
                output.append(message)
                pending_call_ids = set()
                last_assistant_call_ids = set()
        return output

    def _find_committed_tool_call(
        self, conversation: Conversation
    ) -> dict[str, object] | None:
        for message in reversed(conversation.messages):
            if message.role == "assistant" and message.tool_calls:
                if message.tool_calls:
                    return message.tool_calls[0]
        return None

    def _last_user_message(self, conversation: Conversation) -> ChatMessage:
        for message in reversed(conversation.messages):
            if message.role == "user":
                return message
        raise DomainError("Conversation has no user message")

    def _truncate_note(self, text: str) -> str:
        max_length = settings.tool_call_note_max_length
        if len(text) <= max_length:
            return text
        return text[:max_length] + "…"

    async def _get_max_output_tokens(
        self,
        session: AsyncSession,
        user_id: UUID,
        user_role: str | None,
        messages: list[dict[str, object]],
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

    def _estimate_input_tokens(self, messages: list[dict[str, object]]) -> int:
        # A UTF-8 byte count is a conservative tokenizer-independent upper
        # estimate. The fixed reserve covers message framing and instructions.
        # content считается по байтам; для assistant с tool_calls и role:"tool"
        # дополнительно учитываются tool_calls/tool_call_id (см. замечание 5).
        text_bytes = len(CHAT_INSTRUCTIONS.encode())
        for message in messages:
            text_bytes += len(str(message.get("role", "")).encode())
            content = message.get("content")
            if isinstance(content, str):
                text_bytes += len(content.encode())
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                text_bytes += len(str(call.get("name", "")).encode())
                text_bytes += len(str(call.get("arguments", "")).encode())
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str):
                text_bytes += len(tool_call_id.encode())
        return text_bytes + 512

    def _accounted_total_tokens(
        self,
        user_role: str | None,
        reported_total: int | None,
        messages: list[dict[str, object]],
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
