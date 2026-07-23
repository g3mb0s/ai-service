import hashlib
from uuid import UUID

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from basic_utils.config import settings
from basic_utils.exceptions import AIProviderError, EntityNotFoundError
from domains.exercises.models import ExerciseGeneration
from domains.exercises.schemas import (
    GenerateExercisesRequest,
    GeneratedExerciseSet,
    SentenceFromAudioExercise,
)
from domains.exercises.service import exercise_service


EXERCISE_INSTRUCTIONS = (
    "You create English-language exercises for an existing content service. "
    "Return payloads that match the supplied schema exactly. The language field must "
    "always be 'en'. Generated exercises must remain drafts. Never represent a gap "
    "with underscores. For fill-gap exercises, place every gap in text using the exact "
    "{{gap-key}} marker referenced by its gaps entry. Keep nested ids unique."
)


class ExerciseManager:
    async def generate(
        self,
        session: AsyncSession,
        openai_client: AsyncOpenAI,
        ai_manager: AIManager,
        user_id: UUID,
        payload: GenerateExercisesRequest,
    ) -> ExerciseGeneration:
        prompt = (
            f"Topic: {payload.topic}\n"
            "Exercise language: English (language=en)\n"
            f"Learner level: {payload.level}\n"
            f"Exercise type: {payload.exercise_type}\n"
            f"Number of separate exercise payloads: {payload.count}\n"
            f"Tags: {payload.tags}\n"
            f"Type-specific rules: {self._type_rules(payload.exercise_type)}\n"
            f"Audio URL: {payload.audio_url or 'not applicable'}\n"
            f"Additional requirements: {payload.extra_instructions or 'none'}"
        )
        try:
            ai_result = await ai_manager.generate_structured(
                client=openai_client,
                input_text=prompt,
                instructions=EXERCISE_INSTRUCTIONS,
                response_model=GeneratedExerciseSet,
                safety_identifier=self._safety_identifier(user_id),
            )
        except openai.APIError as exc:
            await session.rollback()
            raise AIProviderError(self._provider_error(ai_manager, exc)) from exc
        except AIProviderError:
            await session.rollback()
            raise

        parsed = ai_result.data
        if len(parsed.exercises) != payload.count:
            raise AIProviderError("AI provider returned an unexpected exercise count")
        if any(
            exercise.type != payload.exercise_type for exercise in parsed.exercises
        ):
            raise AIProviderError("AI provider returned an unexpected exercise type")

        for exercise in parsed.exercises:
            exercise.language = "en"
            exercise.level = payload.level
            exercise.tags = list(payload.tags)
            exercise.metadata.version = 1
            exercise.metadata.status = "draft"
            if isinstance(exercise, SentenceFromAudioExercise):
                for item in exercise.content.items:
                    item.audio.url = payload.audio_url or ""

        generation = ExerciseGeneration(
            user_id=user_id,
            topic=payload.topic.strip(),
            language="en",
            level=payload.level.strip(),
            exercise_type=payload.exercise_type,
            extra_instructions=(
                payload.extra_instructions.strip() if payload.extra_instructions else None
            ),
            exercises=[
                exercise.model_dump(mode="json", by_alias=True)
                for exercise in parsed.exercises
            ],
            provider=ai_result.provider,
            provider_host=ai_result.provider_host,
            model=ai_result.model,
            provider_response_id=ai_result.response_id,
            input_tokens=ai_result.usage.input_tokens,
            output_tokens=ai_result.usage.output_tokens,
            total_tokens=ai_result.usage.total_tokens,
        )
        exercise_service.add_generation(session, generation)
        await session.flush()
        await session.commit()
        return generation

    async def get_generation(
        self, session: AsyncSession, generation_id: UUID, user_id: UUID
    ) -> ExerciseGeneration:
        generation = await exercise_service.get_generation(
            session, generation_id, user_id
        )
        if generation is None:
            raise EntityNotFoundError("Exercise generation not found")
        return generation

    async def list_generations(
        self, session: AsyncSession, user_id: UUID
    ) -> list[ExerciseGeneration]:
        return await exercise_service.list_generations(session, user_id)

    def _safety_identifier(self, user_id: UUID) -> str:
        raw = f"{settings.openai_safety_salt}:{user_id}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _provider_error(self, ai_manager: AIManager, exc: openai.APIError) -> str:
        if isinstance(exc, openai.APIStatusError):
            return f"{ai_manager.provider} returned HTTP {exc.status_code}: {exc}"
        return f"{ai_manager.provider} request failed: {exc}"

    def _type_rules(self, exercise_type: str) -> str:
        rules = {
            "fill_gap_choice": (
                "Use text such as 'She {{gap-1}} to school.'; options and answers "
                "belong to the gap object, and every answer must be one of options."
            ),
            "fill_gap_input": (
                "Use {{gap-key}} markers in text; provide answers and acceptedAnswers "
                "for every gap. Never use blank lines or underscores as placeholders."
            ),
            "matching": (
                "Create left and right entries with unique ids; pairs contain exactly "
                "[leftId, rightId] and may reference only existing entries."
            ),
            "sentence_from_audio": (
                "Use the provided audio URL unchanged. Words have unique ids and "
                "answer is the ordered list of those ids forming the English sentence."
            ),
            "sentence_from_translation": (
                "translation.ru contains the Russian prompt; words have unique ids and "
                "answer is the ordered list of ids forming the English sentence."
            ),
        }
        return rules[exercise_type]


exercise_manager = ExerciseManager()
