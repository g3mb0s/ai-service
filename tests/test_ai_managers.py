from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from ai.chatgpt import chatgpt_ai_manager
from ai.deepseek import deepseek_ai_manager
from ai.selector import get_ai_manager
from basic_utils.config import settings


class StructuredAnswer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_chatgpt_uses_responses_api() -> None:
    response = SimpleNamespace(
        id="resp_openai",
        model="gpt-test",
        output_text="answer",
        usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )

    result = await chatgpt_ai_manager.chat(
        client=client,
        messages=[{"role": "user", "content": "question"}],
        instructions="help",
        safety_identifier="user-hash",
        max_output_tokens=123,
    )

    assert result.data == "answer"
    assert result.provider == "openai"
    assert result.usage.total_tokens == 5
    client.responses.create.assert_awaited_once()
    assert client.responses.create.await_args.kwargs["max_output_tokens"] == 123


@pytest.mark.asyncio
async def test_deepseek_uses_chat_completions_and_parses_json() -> None:
    completion = SimpleNamespace(
        id="resp_deepseek",
        model="deepseek-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"structured"}')
            )
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10),
    )
    create = AsyncMock(return_value=completion)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await deepseek_ai_manager.generate_structured(
        client=client,
        input_text="question",
        instructions="return an answer",
        response_model=StructuredAnswer,
        safety_identifier="user-hash",
    )

    assert result.data.answer == "structured"
    assert result.provider == "deepseek"
    assert result.usage.input_tokens == 4
    assert create.await_args.kwargs["response_format"] == {"type": "json_object"}


def test_selector_returns_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_provider", "deepseek")
    assert get_ai_manager() is deepseek_ai_manager

    monkeypatch.setattr(settings, "ai_provider", "chatgpt")
    assert get_ai_manager() is chatgpt_ai_manager
