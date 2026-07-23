from uuid import UUID

from fastapi import APIRouter, Depends, status
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from ai.selector import get_ai_manager
from basic_utils.auth_dependencies import AuthenticatedUser, get_current_user
from basic_utils.database import get_async_session_generator
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
