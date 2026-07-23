from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class AIUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AIResult(Generic[ResultT]):
    data: ResultT
    provider: str
    provider_host: str
    model: str
    response_id: str | None
    usage: AIUsage


@dataclass(frozen=True, slots=True)
class AIStreamEvent:
    delta: str | None = None
    result: AIResult[str] | None = None


class AIManager(ABC):
    """Единый контракт для AI-провайдеров."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Стабильное имя провайдера для конфигурации и БД."""

    @abstractmethod
    async def chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, str]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[str]:
        """Возвращает обычный текстовый ответ."""

    @abstractmethod
    def stream_chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, str]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[AIStreamEvent]:
        """Постепенно возвращает текст и итоговые usage-метаданные."""

    @abstractmethod
    async def generate_structured(
        self,
        client: AsyncOpenAI,
        input_text: str,
        instructions: str,
        response_model: type[StructuredModelT],
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[StructuredModelT]:
        """Возвращает ответ, провалидированный Pydantic-моделью."""
