from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from ai.selector import get_ai_manager
from basic_utils.auth_dependencies import (
    AuthenticatedUser,
    get_admin,
    get_current_user,
)
from basic_utils.database import get_async_session_generator
from basic_utils.openai_client import get_openai_client
from domains.characters.manager import character_chat_manager
from domains.characters.schemas import (
    CharacterConversationResponse,
    CharacterConversationSummaryResponse,
    CharacterAdminResponse,
    CharacterCreateRequest,
    CharacterResponse,
    CharacterTurnResponse,
    CharacterUpdateRequest,
    CreateCharacterConversationRequest,
    SendCharacterMessageRequest,
)


router = APIRouter(tags=["characters"])


@router.get("/characters", response_model=list[CharacterResponse])
async def list_characters(
    session: AsyncSession = Depends(get_async_session_generator),
    _user: AuthenticatedUser = Depends(get_current_user),
) -> list[CharacterResponse]:
    return await character_chat_manager.list_characters(session)


@router.get(
    "/admin/characters",
    response_model=list[CharacterAdminResponse],
)
async def list_admin_characters(
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
) -> list[CharacterAdminResponse]:
    return await character_chat_manager.list_characters(
        session,
        active_only=False,
    )


@router.get(
    "/admin/characters/{character_id}",
    response_model=CharacterAdminResponse,
)
async def get_admin_character(
    character_id: str,
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
) -> CharacterAdminResponse:
    return await character_chat_manager.get_character(session, character_id)


@router.post(
    "/admin/characters",
    response_model=CharacterAdminResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_admin_character(
    payload: CharacterCreateRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
) -> CharacterAdminResponse:
    try:
        return await character_chat_manager.create_character(session, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.put(
    "/admin/characters/{character_id}",
    response_model=CharacterAdminResponse,
)
async def update_admin_character(
    character_id: str,
    payload: CharacterUpdateRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
) -> CharacterAdminResponse:
    return await character_chat_manager.update_character(
        session,
        character_id,
        payload,
    )


@router.delete(
    "/admin/characters/{character_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_admin_character(
    character_id: str,
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
) -> Response:
    try:
        await character_chat_manager.delete_character(session, character_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/characters/{character_id}/conversations",
    response_model=CharacterConversationSummaryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_character_conversation(
    character_id: str,
    _payload: CreateCharacterConversationRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> CharacterConversationSummaryResponse:
    conversation = await character_chat_manager.create_conversation(
        session,
        character_id,
        user.id,
    )
    return CharacterConversationSummaryResponse.model_validate(conversation)


@router.get(
    "/characters/{character_id}/conversations",
    response_model=list[CharacterConversationSummaryResponse],
)
async def list_character_conversations(
    character_id: str,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> list[CharacterConversationSummaryResponse]:
    conversations = await character_chat_manager.list_conversations(
        session,
        character_id,
        user.id,
    )
    return [
        CharacterConversationSummaryResponse.model_validate(conversation)
        for conversation in conversations
    ]


@router.get(
    "/character-conversations/{conversation_id}",
    response_model=CharacterConversationResponse,
)
async def get_character_conversation(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> CharacterConversationResponse:
    conversation = await character_chat_manager.get_conversation(
        session,
        conversation_id,
        user.id,
    )
    return CharacterConversationResponse.model_validate(conversation)


@router.post(
    "/character-conversations/{conversation_id}/messages",
    response_model=CharacterTurnResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_character_message(
    conversation_id: UUID,
    payload: SendCharacterMessageRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_current_user),
    client: AsyncOpenAI = Depends(get_openai_client),
    ai_manager: AIManager = Depends(get_ai_manager),
) -> CharacterTurnResponse:
    user_message, assistant_message = await character_chat_manager.send_message(
        session,
        client,
        ai_manager,
        conversation_id,
        user.id,
        user.role,
        payload.content,
    )
    return CharacterTurnResponse(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )
