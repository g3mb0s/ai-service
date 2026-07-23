from openai import AsyncOpenAI

from ai.base import AIManager, AIResult, AIUsage, StructuredModelT
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError


class ChatGPTAIManager(AIManager):
    @property
    def provider(self) -> str:
        return "openai"

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
            instructions=instructions,
            input=messages,
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

    async def generate_structured(
        self,
        client: AsyncOpenAI,
        input_text: str,
        instructions: str,
        response_model: type[StructuredModelT],
        safety_identifier: str,
    ) -> AIResult[StructuredModelT]:
        response = await client.responses.parse(
            model=settings.openai_model,
            instructions=instructions,
            input=input_text,
            text_format=response_model,
            safety_identifier=safety_identifier,
            store=settings.openai_store,
        )
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

    def _get_usage(self, usage: object | None) -> AIUsage:
        if usage is None:
            return AIUsage()
        return AIUsage(
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


chatgpt_ai_manager = ChatGPTAIManager()
