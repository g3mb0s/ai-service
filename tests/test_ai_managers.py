from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
async def test_chatgpt_streams_deltas_and_final_usage() -> None:
    response = SimpleNamespace(
        id="resp_stream",
        model="gpt-test",
        output_text="streamed answer",
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
    )
    stream = FakeResponseStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="streamed "),
            SimpleNamespace(type="response.output_text.delta", delta="answer"),
            SimpleNamespace(type="response.completed", delta=None),
        ],
        response,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(stream=MagicMock(return_value=stream))
    )

    events = [
        event
        async for event in chatgpt_ai_manager.stream_chat(
            client=client,
            messages=[{"role": "user", "content": "question"}],
            instructions="help",
            safety_identifier="user-hash",
            max_output_tokens=500,
        )
    ]

    assert [event.delta for event in events[:-1]] == ["streamed ", "answer"]
    assert events[-1].result is not None
    assert events[-1].result.usage.total_tokens == 6
    assert client.responses.stream.call_args.kwargs["max_output_tokens"] == 500


@pytest.mark.asyncio
async def test_deepseek_streams_chat_completion_chunks() -> None:
    chunks = [
        SimpleNamespace(
            id="chat-stream",
            model="deepseek-test",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="streamed "))],
            usage=None,
        ),
        SimpleNamespace(
            id="chat-stream",
            model="deepseek-test",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))],
            usage=None,
        ),
        SimpleNamespace(
            id="chat-stream",
            model="deepseek-test",
            choices=[],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        ),
    ]
    stream = FakeChatStream(chunks)
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    events = [
        event
        async for event in deepseek_ai_manager.stream_chat(
            client=client,
            messages=[{"role": "user", "content": "question"}],
            instructions="help",
            safety_identifier="user-hash",
            max_output_tokens=500,
        )
    ]

    assert [event.delta for event in events[:-1]] == ["streamed ", "answer"]
    assert events[-1].result is not None
    assert events[-1].result.usage.total_tokens == 5
    assert create.await_args.kwargs["stream"] is True
    assert create.await_args.kwargs["stream_options"] == {"include_usage": True}
    stream.close.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_deepseek_accepts_json_wrapped_in_markdown_fence() -> None:
    completion = SimpleNamespace(
        id="resp_deepseek",
        model="deepseek-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='```json\n{"answer":"structured"}\n```'
                )
            )
        ],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=completion))
        )
    )

    result = await deepseek_ai_manager.generate_structured(
        client=client,
        input_text="question",
        instructions="return an answer",
        response_model=StructuredAnswer,
        safety_identifier="user-hash",
    )

    assert result.data.answer == "structured"


def test_selector_returns_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_provider", "deepseek")
    assert get_ai_manager() is deepseek_ai_manager

    monkeypatch.setattr(settings, "ai_provider", "chatgpt")
    assert get_ai_manager() is chatgpt_ai_manager


class FakeResponseStream:
    def __init__(self, events: list[SimpleNamespace], response: SimpleNamespace):
        self.events = events
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event

    async def get_final_response(self):
        return self.response


class FakeChatStream:
    def __init__(self, chunks: list[SimpleNamespace]):
        self.chunks = chunks
        self.close = AsyncMock()

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk
