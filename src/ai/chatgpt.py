from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ai.base import (
    AIManager,
    AIResult,
    AIStreamEvent,
    AIStreamToolEvent,
    AIUsage,
    StructuredModelT,
)
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError
from basic_utils.logger import get_logger


logger = get_logger("openai")


class ChatGPTAIManager(AIManager):
    @property
    def provider(self) -> str:
        return "openai"

    async def chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, object]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[str]:
        request: dict[str, object] = dict(
            model=settings.openai_model,
            instructions=instructions,
            input=self._responses_input(messages),
            safety_identifier=safety_identifier,
            store=settings.openai_store,
        )
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        response = await client.responses.create(**request)
        content = response.output_text.strip()
        if not content:
            raise AIProviderError("OpenAI returned an empty response")
        return AIResult(
            data=content,
            provider=self.provider,
            provider_host=settings.openai_base_url,
            model=response.model or settings.openai_model,
            response_id=response.id,
            usage=self._get_usage(response.usage),
        )

    async def stream_chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, object]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[AIStreamEvent]:
        request: dict[str, object] = dict(
            model=settings.openai_model,
            instructions=instructions,
            input=self._responses_input(messages),
            safety_identifier=safety_identifier,
            store=settings.openai_store,
        )
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        async with client.responses.stream(**request) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta" and event.delta:
                    yield AIStreamEvent(delta=event.delta)
            response = await stream.get_final_response()

        content = response.output_text.strip()
        if not content:
            raise AIProviderError("OpenAI returned an empty response")
        yield AIStreamEvent(
            result=AIResult(
                data=content,
                provider=self.provider,
                provider_host=settings.openai_base_url,
                model=response.model or settings.openai_model,
                response_id=response.id,
                usage=self._get_usage(response.usage),
            )
        )

    async def stream_tool_chat(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, object]],
        instructions: str,
        safety_identifier: str,
        tools: list[dict[str, object]],
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[AIStreamToolEvent]:
        request: dict[str, object] = dict(
            model=settings.openai_model,
            instructions=instructions,
            input=self._responses_input(messages),
            tools=tools,
            safety_identifier=safety_identifier,
            store=settings.openai_store,
        )
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        pre_tool_parts: list[str] = []
        arguments_buffer: list[str] = []
        tool_call: tuple[str, str, str] | None = None
        saw_function_call = False
        async with client.responses.stream(**request) as stream:
            async for event in stream:
                if (
                    event.type == "response.output_text.delta"
                    and event.delta
                    and not saw_function_call
                ):
                    pre_tool_parts.append(event.delta)
                    yield AIStreamToolEvent(delta=event.delta)
                elif (
                    event.type == "response.function_call_arguments.delta"
                    and event.delta
                ):
                    arguments_buffer.append(event.delta)
                elif event.type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", None) == "function_call":
                        call_id = getattr(item, "call_id", None)
                        name = getattr(item, "name", None)
                        arguments = getattr(item, "arguments", None)
                        if arguments is None:
                            arguments = "".join(arguments_buffer)
                        if tool_call is None:
                            tool_call = (call_id or "", name or "", arguments or "")
                            saw_function_call = True
                        else:
                            logger.warning(
                                "Ignoring an additional function call in stream_tool_chat",
                                {"name": name or "", "call_id": call_id or ""},
                            )
            response = await stream.get_final_response()

        usage = self._get_usage(response.usage)
        if tool_call is not None:
            pre_tool_text = "".join(pre_tool_parts)
            yield AIStreamToolEvent(
                pre_tool_text=pre_tool_text,
                tool_call_id=tool_call[0],
                tool_name=tool_call[1],
                tool_arguments=tool_call[2],
                usage=usage,
                note=pre_tool_text,
                model=response.model or settings.openai_model,
                response_id=response.id,
            )
            return

        content = response.output_text.strip()
        if not content:
            raise AIProviderError("OpenAI returned an empty response")
        yield AIStreamToolEvent(
            result=AIResult(
                data=content,
                provider=self.provider,
                provider_host=settings.openai_base_url,
                model=response.model or settings.openai_model,
                response_id=response.id,
                usage=usage,
            )
        )

    async def generate_structured(
        self,
        client: AsyncOpenAI,
        input_text: str,
        instructions: str,
        response_model: type[StructuredModelT],
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[StructuredModelT]:
        request: dict[str, object] = dict(
            model=settings.openai_model,
            instructions=instructions,
            input=input_text,
            text_format=response_model,
            safety_identifier=safety_identifier,
            store=settings.openai_store,
        )
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        response = await client.responses.parse(**request)
        if response.output_parsed is None:
            raise AIProviderError("OpenAI did not return structured data")
        return AIResult(
            data=response.output_parsed,
            provider=self.provider,
            provider_host=settings.openai_base_url,
            model=response.model or settings.openai_model,
            response_id=response.id,
            usage=self._get_usage(response.usage),
        )

    def _responses_input(
        self, messages: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Конвертирует канонические словари в формат Responses API input.

        Обычные user/assistant сообщения передаются как есть; assistant с
        ``tool_calls`` разворачивается в assistant-элемент (с ``content: []``,
        если pre-tool текст пуст — см. замечание 4) плюс элементы
        ``function_call``; ``role: "tool"`` становится ``function_call_output``.
        """
        output: list[dict[str, object]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if role == "assistant" and tool_calls:
                assistant_item: dict[str, object] = {"role": "assistant"}
                assistant_item["content"] = content if content else []
                output.append(assistant_item)
                for call in tool_calls:
                    output.append(
                        {
                            "type": "function_call",
                            "call_id": call["id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                        }
                    )
            elif role == "tool":
                output.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id"),
                        "output": content if content is not None else "",
                    }
                )
            else:
                output.append(
                    {
                        "role": role,
                        "content": content if content is not None else "",
                    }
                )
        return output

    def _get_usage(self, usage: object | None) -> AIUsage:
        if usage is None:
            return AIUsage()
        return AIUsage(
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


chatgpt_ai_manager = ChatGPTAIManager()
