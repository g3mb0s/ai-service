class DomainError(Exception):
    """Ожидаемая ошибка бизнес-слоя."""


class EntityNotFoundError(DomainError):
    pass


class AIProviderError(DomainError):
    pass


class DailyTokenLimitError(DomainError):
    pass
