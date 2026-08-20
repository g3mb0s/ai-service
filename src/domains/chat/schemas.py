from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = Field(default=None, max_length=200)


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str | None = Field(default=None, max_length=20_000)
    tool_call_id: str | None = Field(default=None, max_length=64)
    tool_result: str | None = Field(default=None, max_length=100_000)

    @model_validator(mode="after")
    def _validate_phase_fields(self) -> "SendMessageRequest":
        if self.tool_call_id is not None or self.tool_result is not None:
            if self.tool_call_id is None or self.tool_result is None:
                raise ValueError(
                    "tool_call_id and tool_result must be provided together"
                )
            if self.content is not None and self.content != "":
                raise ValueError(
                    "content must be absent when tool_call_id is provided"
                )
            return self
        if self.content is None or self.content == "":
            raise ValueError("content is required")
        return self


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: Literal["user", "assistant"]
    content: str
    tool_calls: list[dict] | None = None
    provider: str | None
    provider_host: str | None
    model: str | None
    provider_response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    created_at: datetime


class ConversationSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationResponse(ConversationSummaryResponse):
    messages: list[MessageResponse]


class ChatTurnResponse(BaseModel):
    conversation_id: UUID
    user_message: MessageResponse
    assistant_message: MessageResponse
