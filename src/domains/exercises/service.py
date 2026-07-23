from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.exercises.models import ExerciseGeneration


class ExerciseService:
    """Только операции чтения и записи в БД, без бизнес-логики."""

    def add_generation(
        self, session: AsyncSession, generation: ExerciseGeneration
    ) -> None:
        session.add(generation)

    async def get_generation(
        self, session: AsyncSession, generation_id: UUID, user_id: UUID
    ) -> ExerciseGeneration | None:
        result = await session.execute(
            select(ExerciseGeneration).where(
                ExerciseGeneration.id == generation_id,
                ExerciseGeneration.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_generations(
        self, session: AsyncSession, user_id: UUID
    ) -> list[ExerciseGeneration]:
        result = await session.execute(
            select(ExerciseGeneration)
            .where(ExerciseGeneration.user_id == user_id)
            .order_by(ExerciseGeneration.created_at.desc())
        )
        return list(result.scalars().all())


exercise_service = ExerciseService()
