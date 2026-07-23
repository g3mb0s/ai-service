from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from domains.chat.models import ChatMessage, Conversation


class ChatService:
    """Только операции чтения и записи в БД, без бизнес-логики."""

    def add_conversation(
        self, session: AsyncSession, conversation: Conversation
    ) -> None:
        session.add(conversation)

    def add_message(self, session: AsyncSession, message: ChatMessage) -> None:
        session.add(message)

    async def get_conversation(
        self, session: AsyncSession, conversation_id: UUID, user_id: UUID
    ) -> Conversation | None:
        result = await session.execute(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()

    async def list_conversations(
        self, session: AsyncSession, user_id: UUID
    ) -> list[Conversation]:
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())


chat_service = ChatService()
