from functools import lru_cache

from openai import AsyncOpenAI

from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """Возвращает один async-клиент с общим пулом соединений."""

    if not settings.openai_api_key:
        raise AIProviderError("OPENAI_API_KEY is not configured")

    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )


async def close_openai_client() -> None:
    if get_openai_client.cache_info().currsize:
        await get_openai_client().close()
        get_openai_client.cache_clear()
