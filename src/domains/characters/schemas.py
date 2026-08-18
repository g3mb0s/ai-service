import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CharacterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    greeting: str
    avatar_url: str | None


class CharacterAdminResponse(CharacterResponse):
    instructions: str | None
    character_prompt: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class CharacterPromptFields(BaseModel):
    """Shared prompt fields for the two request models.

    Both fields are optional in the wire format: an empty/whitespace string is
    normalised to ``None`` (``mode="before"``, so ``"  "`` never fails the
    ``min_length`` constraint), and ``None`` means "field not provided".
    """

    instructions: str | None = Field(default=None, min_length=20, max_length=20_000)
    character_prompt: str | None = Field(default=None, min_length=1, max_length=2_000)

    @field_validator("instructions", "character_prompt", mode="before")
    @classmethod
    def empty_prompt_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            # pydantic v2 lax mode does not coerce non-strings to str (unlike
            # v1), so coerce explicitly; this keeps e.g. a JSON number 123 from
            # causing a 500 and makes it accepted as "123".
            return str(value)
        if not value.strip():
            return None
        return value


class CharacterCreateRequest(CharacterPromptFields):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(
        min_length=2,
        max_length=50,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    greeting: str = Field(min_length=1, max_length=500)
    is_active: bool = True

    @model_validator(mode="after")
    def validate_prompt_fields(self) -> "CharacterCreateRequest":
        if self.character_prompt is not None and self.instructions is not None:
            raise ValueError("provide either character_prompt or instructions, not both")
        if self.character_prompt is None and self.instructions is None:
            raise ValueError("character_prompt or instructions is required")
        return self


class CharacterUpdateRequest(CharacterPromptFields):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    greeting: str = Field(min_length=1, max_length=500)
    is_active: bool

    @model_validator(mode="after")
    def validate_prompt_fields(self) -> "CharacterUpdateRequest":
        if self.character_prompt is not None and self.instructions is not None:
            raise ValueError("provide either character_prompt or instructions, not both")
        # Both fields None is allowed: "keep the existing prompt".
        return self


class CreateCharacterConversationRequest(BaseModel):
    pass


class SendCharacterMessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=2_000)


class CharacterRate(BaseModel):
    quality: int = Field(ge=0, le=10)
    correction: str = Field(max_length=300)
    comment: str = Field(max_length=400)

    @field_validator("correction", "comment", mode="before")
    @classmethod
    def null_feedback_to_empty(cls, value: object) -> object:
        if value is None:
            return ""
        return value

    @model_validator(mode="after")
    def validate_feedback(self) -> "CharacterRate":
        correction = self.correction.strip()
        comment = self.comment.strip()
        if self.quality == 10:
            correction = ""
            comment = ""
        if bool(correction) != bool(comment):
            raise ValueError("correction and comment must be both empty or both filled")
        self.correction = correction
        self.comment = _limit_sentences(comment)
        return self


class CharacterAIResponse(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "text": "Nice to meet you! What would you like to talk about?",
                    "rate": {"quality": 10, "correction": "", "comment": ""},
                }
            ]
        },
    )

    text: str = Field(min_length=1, max_length=500)
    rate: CharacterRate

    @model_validator(mode="after")
    def validate_length(self) -> "CharacterAIResponse":
        self.text = _limit_sentences(self.text)
        return self


class CharacterMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: Literal["user", "assistant"]
    content: str
    quality: int | None
    correction: str | None
    comment: str | None
    created_at: datetime


class CharacterConversationSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    character_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class CharacterConversationResponse(CharacterConversationSummaryResponse):
    messages: list[CharacterMessageResponse]


class CharacterTurnResponse(BaseModel):
    conversation_id: UUID
    user_message: CharacterMessageResponse
    assistant_message: CharacterMessageResponse


def _sentence_count(value: str) -> int:
    if not value.strip():
        return 0
    endings = re.findall(r"[.!?]+(?:\s|$)", value.strip())
    return max(1, len(endings))


def _limit_sentences(value: str, limit: int = 3) -> str:
    value = value.strip()
    if not value or _sentence_count(value) <= limit:
        return value
    parts = re.split(r"(?<=[.!?])\s+", value)
    return " ".join(parts[:limit]).strip()
