import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
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
from infrastructure.object_storage import ObjectStorage, get_object_storage


router = APIRouter(tags=["characters"])
logger = logging.getLogger(__name__)
MAX_AVATAR_BYTES = 5 * 1024 * 1024
AVATAR_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _matches_image_signature(content: bytes, content_type: str) -> bool:
    if content_type == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    if content_type == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if content_type == "image/webp":
        return (
            len(content) >= 12
            and content.startswith(b"RIFF")
            and content[8:12] == b"WEBP"
        )
    return False


async def _remove_object_quietly(
    storage: ObjectStorage,
    object_key: str | None,
) -> None:
    if not object_key:
        return
    try:
        await run_in_threadpool(storage.remove, object_key)
    except Exception:
        logger.exception("Failed to remove object %s", object_key)


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


@router.put(
    "/admin/characters/{character_id}/avatar",
    response_model=CharacterAdminResponse,
)
async def update_admin_character_avatar(
    character_id: str,
    avatar: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
    storage: ObjectStorage = Depends(get_object_storage),
) -> CharacterAdminResponse:
    extension = AVATAR_EXTENSIONS.get(avatar.content_type or "")
    if extension is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Avatar must be a JPEG, PNG, or WebP image",
        )
    content = await avatar.read(MAX_AVATAR_BYTES + 1)
    await avatar.close()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Avatar file is empty",
        )
    if len(content) > MAX_AVATAR_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Avatar must not exceed 5 MB",
        )
    if not _matches_image_signature(content, avatar.content_type):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File content does not match its image type",
        )

    character = await character_chat_manager.get_character(session, character_id)
    previous_key = character.avatar_object_key
    object_key = f"characters/{character_id}/{uuid4().hex}.{extension}"
    avatar_url = await run_in_threadpool(
        storage.put,
        object_key,
        content,
        avatar.content_type,
    )
    try:
        character = await character_chat_manager.update_character_avatar(
            session,
            character,
            avatar_url=avatar_url,
            avatar_object_key=object_key,
        )
    except Exception:
        await _remove_object_quietly(storage, object_key)
        raise
    await _remove_object_quietly(storage, previous_key)
    return character


@router.delete(
    "/admin/characters/{character_id}/avatar",
    response_model=CharacterAdminResponse,
)
async def delete_admin_character_avatar(
    character_id: str,
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
    storage: ObjectStorage = Depends(get_object_storage),
) -> CharacterAdminResponse:
    character = await character_chat_manager.get_character(session, character_id)
    previous_key = character.avatar_object_key
    character = await character_chat_manager.update_character_avatar(
        session,
        character,
        avatar_url=None,
        avatar_object_key=None,
    )
    await _remove_object_quietly(storage, previous_key)
    return character


@router.delete(
    "/admin/characters/{character_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_admin_character(
    character_id: str,
    session: AsyncSession = Depends(get_async_session_generator),
    _admin: AuthenticatedUser = Depends(get_admin),
    storage: ObjectStorage = Depends(get_object_storage),
) -> Response:
    character = await character_chat_manager.get_character(session, character_id)
    avatar_object_key = character.avatar_object_key
    try:
        await character_chat_manager.delete_character(session, character_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    await _remove_object_quietly(storage, avatar_object_key)
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
