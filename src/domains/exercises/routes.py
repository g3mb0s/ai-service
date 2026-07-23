from uuid import UUID

from fastapi import APIRouter, Depends, status
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ai.base import AIManager
from ai.selector import get_ai_manager
from basic_utils.auth_dependencies import AuthenticatedUser, get_manager_or_admin
from basic_utils.database import get_async_session_generator
from basic_utils.openai_client import get_openai_client
from domains.exercises.manager import exercise_manager
from domains.exercises.schemas import (
    ExerciseGenerationResponse,
    GenerateExercisesRequest,
)

router = APIRouter(prefix="/exercises", tags=["exercises"])


@router.post(
    "/generate",
    response_model=ExerciseGenerationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_exercises(
    payload: GenerateExercisesRequest,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_manager_or_admin),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
    provider_manager: AIManager = Depends(get_ai_manager),
) -> ExerciseGenerationResponse:
    generation = await exercise_manager.generate(
        session, openai_client, provider_manager, user.id, payload
    )
    return ExerciseGenerationResponse.model_validate(generation)


@router.get("", response_model=list[ExerciseGenerationResponse])
async def list_generations(
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_manager_or_admin),
) -> list[ExerciseGenerationResponse]:
    generations = await exercise_manager.list_generations(session, user.id)
    return [ExerciseGenerationResponse.model_validate(item) for item in generations]


@router.get("/{generation_id}", response_model=ExerciseGenerationResponse)
async def get_generation(
    generation_id: UUID,
    session: AsyncSession = Depends(get_async_session_generator),
    user: AuthenticatedUser = Depends(get_manager_or_admin),
) -> ExerciseGenerationResponse:
    generation = await exercise_manager.get_generation(
        session, generation_id, user.id
    )
    return ExerciseGenerationResponse.model_validate(generation)
