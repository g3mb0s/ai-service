from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from basic_utils.config import settings
from basic_utils.exceptions import DailyTokenLimitError
from domains.chat.models import ChatMessage, Conversation
from domains.characters.models import CharacterConversation, CharacterMessage


class TokenQuotaService:
    """Checks the persistent daily token quota for regular users."""

    async def get_remaining_tokens(
        self,
        session: AsyncSession,
        user_id: UUID,
        role: str | None,
    ) -> int | None:
        if role != "user":
            return None

        day_start = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        day_end = day_start + timedelta(days=1)

        # Serialize calls made by the same user during the same UTC day.
        # This transaction-level lock is released by commit or rollback.
        await session.execute(
            select(func.pg_advisory_xact_lock(self._lock_key(user_id, day_start)))
        )

        regular_chat_usage = (
            select(ChatMessage.total_tokens.label("total_tokens"))
            .join(Conversation, Conversation.id == ChatMessage.conversation_id)
            .where(
                Conversation.user_id == user_id,
                ChatMessage.role == "assistant",
                ChatMessage.created_at >= day_start,
                ChatMessage.created_at < day_end,
            )
        )
        character_chat_usage = (
            select(CharacterMessage.total_tokens.label("total_tokens"))
            .join(
                CharacterConversation,
                CharacterConversation.id == CharacterMessage.conversation_id,
            )
            .where(
                CharacterConversation.user_id == user_id,
                CharacterMessage.role == "assistant",
                CharacterMessage.created_at >= day_start,
                CharacterMessage.created_at < day_end,
            )
        )
        all_usage = union_all(regular_chat_usage, character_chat_usage).subquery()
        result = await session.execute(
            select(func.coalesce(func.sum(all_usage.c.total_tokens), 0))
        )
        used_tokens = int(result.scalar_one())
        remaining_tokens = settings.user_daily_token_limit - used_tokens
        if remaining_tokens <= 0:
            raise self.limit_error()
        return remaining_tokens

    def limit_error(self) -> DailyTokenLimitError:
        formatted_limit = f"{settings.user_daily_token_limit:,}".replace(",", " ")
        return DailyTokenLimitError(
            f"Дневной лимит ИИ ({formatted_limit} токенов) исчерпан. "
            "Он обновится в 00:00 UTC."
        )

    def _lock_key(self, user_id: UUID, day_start: datetime) -> int:
        raw = f"{user_id}:{day_start.date().isoformat()}".encode()
        unsigned = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        return unsigned if unsigned < 2**63 else unsigned - 2**64


token_quota_service = TokenQuotaService()
