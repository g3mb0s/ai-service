from ai.base import AIManager
from ai.chatgpt import chatgpt_ai_manager
from ai.deepseek import deepseek_ai_manager
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError


def get_ai_manager() -> AIManager:
    provider = settings.ai_provider.strip().lower()
    if provider in {"openai", "chatgpt"}:
        return chatgpt_ai_manager
    if provider == "deepseek":
        return deepseek_ai_manager
    raise AIProviderError(f"Unsupported AI_PROVIDER: {settings.ai_provider}")
