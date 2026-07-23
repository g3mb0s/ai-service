import json

from openai import AsyncOpenAI
from pydantic import ValidationError

from ai.base import AIManager, AIResult, AIUsage, StructuredModelT
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError


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

    async def generate_structured(
        self,
        client: AsyncOpenAI,
        input_text: str,
        instructions: str,
        response_model: type[StructuredModelT],
        safety_identifier: str,
    ) -> AIResult[StructuredModelT]:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        completion = await client.chat.completions.create(
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
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise AIProviderError("DeepSeek returned an empty structured response")
        try:
            parsed = response_model.model_validate_json(content)
        except ValidationError as exc:
            raise AIProviderError(
                "DeepSeek returned JSON that does not match the exercise schema"
            ) from exc
        return AIResult(
            data=parsed,
            provider=self.provider,
            provider_host=settings.openai_base_url,
            model=completion.model or settings.openai_model,
            response_id=completion.id,
            usage=self._get_usage(completion.usage),
        )

    def _get_usage(self, usage: object | None) -> AIUsage:
        if usage is None:
            return AIUsage()
        return AIUsage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


deepseek_ai_manager = DeepSeekAIManager()
