from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from basic_utils.exceptions import DailyTokenLimitError
from domains.quota.service import TokenQuotaService


@pytest.mark.asyncio
async def test_regular_user_is_rejected_after_daily_limit() -> None:
    service = TokenQuotaService()
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(),
            SimpleNamespace(scalar_one=lambda: 50_000),
        ]
    )

    with pytest.raises(DailyTokenLimitError):
        await service.get_remaining_tokens(session, uuid4(), "user")

    assert session.execute.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["manager", "admin", None])
async def test_non_user_roles_are_not_limited(role: str | None) -> None:
    service = TokenQuotaService()
    session = MagicMock()
    session.execute = AsyncMock()

    remaining = await service.get_remaining_tokens(session, uuid4(), role)

    assert remaining is None
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_usage_below_limit_returns_remaining_tokens() -> None:
    service = TokenQuotaService()
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(),
            SimpleNamespace(scalar_one=lambda: 12_500),
        ]
    )

    remaining = await service.get_remaining_tokens(session, uuid4(), "user")

    assert remaining == 37_500


def test_daily_lock_key_is_stable() -> None:
    service = TokenQuotaService()
    user_id = uuid4()
    day = datetime(2026, 7, 23, tzinfo=timezone.utc)

    assert service._lock_key(user_id, day) == service._lock_key(user_id, day)
