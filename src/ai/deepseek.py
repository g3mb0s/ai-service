import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from pydantic import ValidationError

from ai.base import AIManager, AIResult, AIStreamEvent, AIUsage, StructuredModelT
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
        messages: list[dict[str, str]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[str]:
        request: dict[str, object] = dict(
            model=settings.openai_model,
            messages=[{"role": "system", "content": instructions}, *messages],
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
        messages: list[dict[str, str]],
        instructions: str,
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[AIStreamEvent]:
        request: dict[str, object] = dict(
            model=settings.openai_model,
            messages=[{"role": "system", "content": instructions}, *messages],
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

    async def generate_structured(
        self,
        client: AsyncOpenAI,
        input_text: str,
        instructions: str,
        response_model: type[StructuredModelT],
        safety_identifier: str,
        max_output_tokens: int | None = None,
    ) -> AIResult[StructuredModelT]:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        request: dict[str, object] = dict(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{instructions}\nReturn only a valid JSON object matching "
                        f"this JSON Schema: {schema}"
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            response_format={"type": "json_object"},
            user=safety_identifier,
        )
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens
        completion = await client.chat.completions.create(**request)
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise AIProviderError("DeepSeek returned an empty structured response")
        try:
            parsed = response_model.model_validate_json(
                self._extract_json_object(content)
            )
        except ValidationError as exc:
            logger.warning(
                "Structured response validation failed",
                {
                    "schema": response_model.__name__,
                    "errors": exc.errors(include_url=False, include_input=False),
                },
            )
            raise AIProviderError(
                f"DeepSeek returned invalid structured data for "
                f"{response_model.__name__}"
            ) from exc
        return AIResult(
            data=parsed,
            provider=self.provider,
            provider_host=settings.openai_base_url,
            model=completion.model or settings.openai_model,
            response_id=completion.id,
            usage=self._get_usage(completion.usage),
        )

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
