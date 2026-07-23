from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


ExerciseType = Literal[
    "fill_gap_choice",
    "fill_gap_input",
    "matching",
    "sentence_from_audio",
    "sentence_from_translation",
]
ContentStatus = Literal["draft", "published", "archived"]


class GenerateExercisesRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    topic: str = Field(min_length=1, max_length=300)
    level: str = Field(default="A1", min_length=1, max_length=50)
    exercise_type: ExerciseType = "fill_gap_choice"
    count: int = Field(default=5, ge=1, le=10)
    tags: list[str] = Field(default_factory=list, max_length=20)
    extra_instructions: str | None = Field(default=None, max_length=2_000)
    audio_url: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_audio_url(self) -> "GenerateExercisesRequest":
        if self.exercise_type == "sentence_from_audio" and not self.audio_url:
            raise ValueError("audio_url is required for sentence_from_audio")
        return self


class ExerciseSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    shuffle_items: bool = Field(alias="shuffleItems")
    shuffle_options: bool = Field(alias="shuffleOptions")
    case_sensitive: bool = Field(alias="caseSensitive")


class ExerciseScoring(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mode: str
    max_score: float = Field(alias="maxScore", ge=0)


class ExerciseMetadata(BaseModel):
    version: int = Field(ge=1)
    status: ContentStatus


class ExerciseContractBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    title: str = Field(min_length=1, max_length=300)
    level: str = Field(min_length=1, max_length=50)
    language: Literal["en"]
    tags: list[str]
    settings: ExerciseSettings
    scoring: ExerciseScoring
    metadata: ExerciseMetadata


class FillGapChoiceGap(BaseModel):
    key: str = Field(min_length=1)
    options: list[str] = Field(min_length=1)
    answers: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answers(self) -> "FillGapChoiceGap":
        if not set(self.answers).issubset(self.options):
            raise ValueError("fill-gap answers must be present in options")
        return self


class FillGapInputGap(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str = Field(min_length=1)
    answers: list[str] = Field(min_length=1)
    accepted_answers: list[str] = Field(alias="acceptedAnswers")


class FillGapChoiceItem(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    gaps: list[FillGapChoiceGap] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_markers(self) -> "FillGapChoiceItem":
        _validate_gap_markers(self.text, [gap.key for gap in self.gaps])
        return self


class FillGapInputItem(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    gaps: list[FillGapInputGap] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_markers(self) -> "FillGapInputItem":
        _validate_gap_markers(self.text, [gap.key for gap in self.gaps])
        return self


class FillGapChoiceContent(BaseModel):
    items: list[FillGapChoiceItem] = Field(min_length=1)


class FillGapInputContent(BaseModel):
    items: list[FillGapInputItem] = Field(min_length=1)


class FillGapChoiceExercise(ExerciseContractBase):
    type: Literal["fill_gap_choice"]
    content: FillGapChoiceContent


class FillGapInputExercise(ExerciseContractBase):
    type: Literal["fill_gap_input"]
    content: FillGapInputContent


class MatchEntry(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MatchingItem(BaseModel):
    id: str = Field(min_length=1)
    left: list[MatchEntry] = Field(min_length=1)
    right: list[MatchEntry] = Field(min_length=1)
    pairs: list[list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pairs(self) -> "MatchingItem":
        left_ids = [entry.id for entry in self.left]
        right_ids = [entry.id for entry in self.right]
        _validate_unique_ids([*left_ids, *right_ids])
        for pair in self.pairs:
            if len(pair) != 2:
                raise ValueError("each matching pair must contain exactly two ids")
            if pair[0] not in left_ids or pair[1] not in right_ids:
                raise ValueError("matching pair references an unknown id")
        return self


class MatchingContent(BaseModel):
    items: list[MatchingItem] = Field(min_length=1)


class MatchingExercise(ExerciseContractBase):
    type: Literal["matching"]
    content: MatchingContent


class WordEntry(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AudioPrompt(BaseModel):
    url: str = Field(min_length=1)


class TranslationPrompt(BaseModel):
    ru: str = Field(min_length=1)


class SentenceFromAudioItem(BaseModel):
    audio: AudioPrompt
    words: list[WordEntry] = Field(min_length=1)
    answer: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answer(self) -> "SentenceFromAudioItem":
        _validate_word_answer(self.words, self.answer)
        return self


class SentenceFromTranslationItem(BaseModel):
    translation: TranslationPrompt
    words: list[WordEntry] = Field(min_length=1)
    answer: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answer(self) -> "SentenceFromTranslationItem":
        _validate_word_answer(self.words, self.answer)
        return self


class SentenceFromAudioContent(BaseModel):
    items: list[SentenceFromAudioItem] = Field(min_length=1)


class SentenceFromTranslationContent(BaseModel):
    items: list[SentenceFromTranslationItem] = Field(min_length=1)


class SentenceFromAudioExercise(ExerciseContractBase):
    type: Literal["sentence_from_audio"]
    content: SentenceFromAudioContent


class SentenceFromTranslationExercise(ExerciseContractBase):
    type: Literal["sentence_from_translation"]
    content: SentenceFromTranslationContent


ExerciseContract = Annotated[
    FillGapChoiceExercise
    | FillGapInputExercise
    | MatchingExercise
    | SentenceFromAudioExercise
    | SentenceFromTranslationExercise,
    Field(discriminator="type"),
]


class GeneratedExerciseSet(BaseModel):
    exercises: list[ExerciseContract] = Field(min_length=1)


class ExerciseGenerationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    topic: str
    language: Literal["en"]
    level: str
    exercise_type: ExerciseType
    extra_instructions: str | None
    exercises: list[ExerciseContract]
    provider: str
    provider_host: str | None
    model: str
    provider_response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    created_at: datetime


def _validate_gap_markers(text: str, keys: list[str]) -> None:
    if "___" in text:
        raise ValueError("use {{gap-key}} markers instead of underscores")
    _validate_unique_ids(keys)
    for key in keys:
        if f"{{{{{key}}}}}" not in text:
            raise ValueError(f"gap marker {{{{{key}}}}} is missing from text")


def _validate_unique_ids(values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError("ids and gap keys must be unique inside an exercise")


def _validate_word_answer(words: list[WordEntry], answer: list[str]) -> None:
    word_ids = [word.id for word in words]
    _validate_unique_ids(word_ids)
    if any(word_id not in word_ids for word_id in answer):
        raise ValueError("sentence answer references an unknown word id")
