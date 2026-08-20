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


@dataclass(frozen=True, slots=True)
class AIStreamToolEvent:
    """Потоковый результат вызова модели с возможностью вызова тула.

    Провайдер эмитит события ``delta`` по мере генерации pre-tool текста, а
    затем одно финальное событие: либо вызов тула (``tool_call_id`` +
    ``tool_name`` + ``tool_arguments`` + ``usage``), либо обычный текстовый
    ответ (``result``). Поле ``note`` равно ``pre_tool_text`` (замечание 1).
    """

    delta: str | None = None
    pre_tool_text: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None
    result: AIResult[str] | None = None
    usage: AIUsage | None = None
    note: str = ""
    model: str = ""
    response_id: str | None = None


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
        messages: list[dict[str, object]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[str]:
        """Возвращает обычный текстовый ответ."""

    @abstractmethod
    def stream_chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, object]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[AIStreamEvent]:
        """Постепенно возвращает текст и итоговые usage-метаданные."""

    @abstractmethod
    def stream_tool_chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, object]],
        instructions: str,
        safety_identifier: str,
        tools: list[dict[str, object]],
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[AIStreamToolEvent]:
        """Стримит ответ модели с объявленными тулами.

        Если модель вызвала тул — финальное событие содержит вызов
        (``tool_call_id``/``tool_name``/``tool_arguments``). Иначе финальное
        событие содержит ``result`` с текстовым ответом.
        """

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
