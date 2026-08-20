import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from pydantic import ValidationError

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


logger = get_logger("deepseek")


class DeepSeekAIManager(AIManager):
    @property
    def provider(self) -> str:
        return "deepseek"

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
            messages=[
                {"role": "system", "content": instructions},
                *self._chat_completions_messages(messages),
            ],
            user=safety_identifier,
        )
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens
        completion = await client.chat.completions.create(**request)
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise AIProviderError("DeepSeek returned an empty response")
        return AIResult(
            data=content.strip(),
            provider=self.provider,
            provider_host=settings.openai_base_url,
            model=completion.model or settings.openai_model,
            response_id=completion.id,
            usage=self._get_usage(completion.usage),
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
            messages=[
                {"role": "system", "content": instructions},
                *self._chat_completions_messages(messages),
            ],
            user=safety_identifier,
            stream=True,
            stream_options={"include_usage": True},
        )
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens

        stream = await client.chat.completions.create(**request)
        parts: list[str] = []
        response_id: str | None = None
        response_model = settings.openai_model
        usage: object | None = None
        try:
            async for chunk in stream:
                response_id = getattr(chunk, "id", None) or response_id
                response_model = getattr(chunk, "model", None) or response_model
                usage = getattr(chunk, "usage", None) or usage
                choices = getattr(chunk, "choices", [])
                delta = choices[0].delta.content if choices else None
                if delta:
                    parts.append(delta)
                    yield AIStreamEvent(delta=delta)
        finally:
            await stream.close()

        content = "".join(parts).strip()
        if not content:
            raise AIProviderError("DeepSeek returned an empty response")
        yield AIStreamEvent(
            result=AIResult(
                data=content,
                provider=self.provider,
                provider_host=settings.openai_base_url,
                model=response_model,
                response_id=response_id,
                usage=self._get_usage(usage),
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
            messages=[
                {"role": "system", "content": instructions},
                *self._chat_completions_messages(messages),
            ],
            tools=tools,
            tool_choice="auto",
            user=safety_identifier,
            stream=True,
            stream_options={"include_usage": True},
        )
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens

        stream = await client.chat.completions.create(**request)
        pre_tool_parts: list[str] = []
        # index -> {"id", "name", "arguments"}; id/name берутся из первого чанка
        # данного index и бэкфиллятся при конкатенации arguments (замечание 3).
        calls: dict[int, dict[str, str]] = {}
        call_order: list[int] = []
        saw_tool_fragment = False
        response_id: str | None = None
        response_model = settings.openai_model
        usage: object | None = None
        try:
            async for chunk in stream:
                response_id = getattr(chunk, "id", None) or response_id
                response_model = getattr(chunk, "model", None) or response_model
                usage = getattr(chunk, "usage", None) or usage
                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue
                delta = choices[0].delta
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    saw_tool_fragment = True
                    for tc in tool_calls:
                        index = getattr(tc, "index", 0)
                        entry = calls.get(index)
                        if entry is None:
                            entry = {"id": "", "name": "", "arguments": ""}
                            calls[index] = entry
                            call_order.append(index)
                        tc_id = getattr(tc, "id", None)
                        function = getattr(tc, "function", None)
                        tc_name = (
                            getattr(function, "name", None)
                            if function is not None
                            else None
                        )
                        args_delta = (
                            getattr(function, "arguments", None)
                            if function is not None
                            else None
                        )
                        if tc_id:
                            entry["id"] = tc_id
                        if tc_name:
                            entry["name"] = tc_name
                        if args_delta:
                            entry["arguments"] += args_delta
                elif content and not saw_tool_fragment:
                    pre_tool_parts.append(content)
                    yield AIStreamToolEvent(delta=content)
        finally:
            await stream.close()

        if calls:
            first_call = calls[call_order[0]]
            pre_tool_text = "".join(pre_tool_parts)
            yield AIStreamToolEvent(
                pre_tool_text=pre_tool_text,
                tool_call_id=first_call["id"],
                tool_name=first_call["name"],
                tool_arguments=first_call["arguments"],
                usage=self._get_usage(usage),
                note=pre_tool_text,
                model=response_model,
                response_id=response_id,
            )
            return

        content = "".join(pre_tool_parts).strip()
        if not content:
            raise AIProviderError("DeepSeek returned an empty response")
        yield AIStreamToolEvent(
            result=AIResult(
                data=content,
                provider=self.provider,
                provider_host=settings.openai_base_url,
                model=response_model,
                response_id=response_id,
                usage=self._get_usage(usage),
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
        schema_obj = response_model.model_json_schema()
        schema = json.dumps(schema_obj, ensure_ascii=False)
        system_content = (
            f"{instructions}\n\n"
            "Return your answer as a single valid JSON object and nothing else. "
            "Do not wrap it in markdown code fences. Do not add any explanation, "
            "comment, or prose before or after the JSON object. Use exactly the "
            "field names and value types declared in the JSON Schema below: strings "
            "as JSON strings, integers as JSON integers, and null only where the "
            "schema explicitly allows it.\n"
            f"JSON Schema: {schema}"
        )
        examples = schema_obj.get("examples")
        if examples:
            system_content += (
                "\nExample of a valid response:\n"
                + json.dumps(examples[0], ensure_ascii=False)
            )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": input_text},
        ]
        request: dict[str, object] = dict(
            model=settings.openai_model,
            messages=messages,
            response_format={"type": "json_object"},
            user=safety_identifier,
        )
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens

        last_error: ValidationError | None = None
        for attempt in range(2):
            completion = await client.chat.completions.create(**request)
            content = completion.choices[0].message.content
            if not content or not content.strip():
                if attempt == 0:
                    request["messages"] = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was empty. Reply again with "
                                "only the valid JSON object."
                            ),
                        },
                    ]
                    continue
                raise AIProviderError(
                    "DeepSeek returned an empty structured response"
                )
            try:
                parsed = response_model.model_validate_json(
                    self._extract_json_object(content)
                )
                return AIResult(
                    data=parsed,
                    provider=self.provider,
                    provider_host=settings.openai_base_url,
                    model=completion.model or settings.openai_model,
                    response_id=completion.id,
                    usage=self._get_usage(completion.usage),
                )
            except ValidationError as exc:
                last_error = exc
                if attempt == 0:
                    request["messages"] = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not a valid JSON object "
                                "matching the schema. Reply again with only the "
                                "valid JSON object, using the exact field names and "
                                "value types from the schema."
                            ),
                        },
                    ]
                    continue
        logger.warning(
            "Structured response validation failed",
            {
                "schema": response_model.__name__,
                "errors": (
                    last_error.errors(include_url=False, include_input=False)
                    if last_error is not None
                    else []
                ),
            },
        )
        raise AIProviderError(
            f"DeepSeek returned invalid structured data for "
            f"{response_model.__name__}"
        ) from last_error

    def _chat_completions_messages(
        self, messages: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Конвертирует канонические tool_calls в формат chat.completions.

        Каноническая форма (как хранится в БД): assistant c
        ``tool_calls=[{id, name, arguments}]``. Chat Completions (и DeepSeek)
        ожидает ``tool_calls=[{id, type: "function", function: {name, arguments}}]``.
        Сообщения ``role: "tool"`` уже в правильном формате.
        """
        converted: list[dict[str, object]] = []
        for message in messages:
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                converted.append(message)
                continue
            chat_completions_calls: list[dict[str, object]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                if "type" in call and "function" in call:
                    chat_completions_calls.append(call)
                    continue
                chat_completions_calls.append(
                    {
                        "id": call.get("id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": call.get("arguments"),
                        },
                    }
                )
            converted.append({**message, "tool_calls": chat_completions_calls})
        return converted

    def _extract_json_object(self, content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            cleaned = cleaned[first_newline + 1 :] if first_newline != -1 else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end >= start:
            return cleaned[start : end + 1]
        return cleaned

    def _get_usage(self, usage: object | None) -> AIUsage:
        if usage is None:
            return AIUsage()
        return AIUsage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


deepseek_ai_manager = DeepSeekAIManager()
