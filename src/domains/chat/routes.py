import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from ai.selector import get_ai_manager
from basic_utils.auth_dependencies import AuthenticatedUser, get_current_user
from basic_utils.database import get_async_session_generator
from basic_utils.exceptions import AIProviderError
from basic_utils.openai_client import get_openai_client
from domains.chat.manager import chat_manager
from domains.chat.schemas import (
    ChatTurnResponse,
    ConversationResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    SendMessageRequest,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/conversations",
    response_model=ConversationSummaryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: CreateConversationRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationSummaryResponse:
    conversation = await chat_manager.create_conversation(session, user.id, payload.title)
    return ConversationSummaryResponse.model_validate(conversation)


@router.get("/conversations", response_model=list[ConversationSummaryResponse])
async def list_conversations(
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> list[ConversationSummaryResponse]:
    conversations = await chat_manager.list_conversations(session, user.id)
    return [ConversationSummaryResponse.model_validate(item) for item in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationResponse:
    conversation = await chat_manager.get_conversation(
        session, conversation_id, user.id
    )
    return ConversationResponse.model_validate(conversation)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=ChatTurnResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    conversation_id: UUID,
    payload: SendMessageRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
    provider_manager: AIManager = Depends(get_ai_manager),
) -> ChatTurnResponse:
    user_message, assistant_message = await chat_manager.send_message(
        session,
        openai_client,
        provider_manager,
        conversation_id,
        user.id,
        user.role,
        payload.content,
    )
    return ChatTurnResponse(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )


@router.post("/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: UUID,
    payload: SendMessageRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
    provider_manager: AIManager = Depends(get_ai_manager),
) -> StreamingResponse:
    # Preparation happens before the response starts so auth, ownership and quota
    # errors preserve their normal HTTP status codes.
    prepared = await chat_manager.prepare_message(
        session,
        conversation_id,
        user.id,
        user.role,
        payload.content,
    )

    async def events() -> AsyncIterator[str]:
        yield _sse("ready", {})
        try:
            async for event in chat_manager.stream_prepared_message(
                session,
                openai_client,
                provider_manager,
                user.id,
                prepared,
            ):
                if event.delta is not None:
                    yield _sse("delta", {"delta": event.delta})
                if event.messages is not None:
                    user_message, assistant_message = event.messages
                    turn = ChatTurnResponse(
                        conversation_id=conversation_id,
                        user_message=user_message,
                        assistant_message=assistant_message,
                    )
                    yield _sse("done", turn.model_dump(mode="json"))
        except AIProviderError as exc:
            yield _sse("error", {"message": str(exc)})
        except Exception:
            yield _sse(
                "error",
                {"message": "Не удалось завершить потоковый ответ ИИ"},
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
