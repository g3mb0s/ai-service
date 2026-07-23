from uuid import uuid4

import pytest
from fastapi import HTTPException

from basic_utils.auth_dependencies import AuthenticatedUser, get_manager_or_admin


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["manager", "admin"])
async def test_manager_and_admin_have_exercise_access(role: str) -> None:
    user = AuthenticatedUser(id=uuid4(), email="staff@example.com", role=role)

    result = await get_manager_or_admin(user)

    assert result is user


@pytest.mark.asyncio
async def test_regular_user_has_no_exercise_access() -> None:
    user = AuthenticatedUser(id=uuid4(), email="user@example.com", role="user")

    with pytest.raises(HTTPException) as error:
        await get_manager_or_admin(user)

    assert error.value.status_code == 403
