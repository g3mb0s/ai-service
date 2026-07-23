import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from basic_utils.config import settings

ASYNC_DATABASE_URL = os.getenv(
    "ASYNC_DATABASE_URL",
    f"{settings.db_async_driver}://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}",
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    max_overflow=10,
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session_generator() -> AsyncGenerator[AsyncSession, None]:
    """Создаёт сессию для FastAPI; транзакцией управляет manager."""

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
