from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from domains.characters.models import Character, CharacterConversation, CharacterMessage


class CharacterChatService:
    def add_character(self, session: AsyncSession, character: Character) -> None:
        session.add(character)

    async def get_character(
        self,
        session: AsyncSession,
        character_id: str,
        *,
        active_only: bool = False,
    ) -> Character | None:
        query = select(Character).where(Character.id == character_id)
        if active_only:
            query = query.where(Character.is_active.is_(True))
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def list_characters(
        self,
        session: AsyncSession,
        *,
        active_only: bool,
    ) -> list[Character]:
        query = select(Character)
        if active_only:
            query = query.where(Character.is_active.is_(True))
        result = await session.execute(query.order_by(Character.name, Character.id))
        return list(result.scalars().all())

    async def count_conversations(
        self,
        session: AsyncSession,
        character_id: str,
    ) -> int:
        result = await session.execute(
            select(func.count(CharacterConversation.id)).where(
                CharacterConversation.character_id == character_id
            )
        )
        return int(result.scalar_one())

    async def delete_character(
        self,
        session: AsyncSession,
        character: Character,
    ) -> None:
        await session.delete(character)

    def add_conversation(
        self,
        session: AsyncSession,
        conversation: CharacterConversation,
    ) -> None:
        session.add(conversation)

    def add_message(self, session: AsyncSession, message: CharacterMessage) -> None:
        session.add(message)

    async def get_conversation(
        self,
        session: AsyncSession,
        conversation_id: UUID,
        user_id: UUID,
    ) -> CharacterConversation | None:
        result = await session.execute(
            select(CharacterConversation)
            .where(
                CharacterConversation.id == conversation_id,
                CharacterConversation.user_id == user_id,
            )
            .options(selectinload(CharacterConversation.messages))
        )
        return result.scalar_one_or_none()

    async def list_conversations(
        self,
        session: AsyncSession,
        character_id: str,
        user_id: UUID,
    ) -> list[CharacterConversation]:
        result = await session.execute(
            select(CharacterConversation)
            .where(
                CharacterConversation.character_id == character_id,
                CharacterConversation.user_id == user_id,
            )
            .order_by(CharacterConversation.updated_at.desc())
        )
        return list(result.scalars().all())


character_chat_service = CharacterChatService()
