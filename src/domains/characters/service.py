from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from domains.characters.models import CharacterConversation, CharacterMessage


class CharacterChatService:
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
