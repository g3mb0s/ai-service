from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai.base import AIResult, AIUsage
from domains.exercises import manager as manager_module
from domains.exercises.manager import exercise_manager
from domains.exercises.schemas import (
    FillGapChoiceExercise,
    GenerateExercisesRequest,
    GeneratedExerciseSet,
)


@pytest.mark.asyncio
async def test_generate_persists_structured_exercises_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    generated = GeneratedExerciseSet(
        exercises=[
            FillGapChoiceExercise.model_validate(
                {
                    "type": "fill_gap_choice",
                    "title": "Present Simple",
                    "level": "A1",
                    "language": "en",
                    "tags": ["grammar"],
                    "content": {
                        "items": [
                            {
                                "id": "item-1",
                                "text": "She {{gap-1}} to school every day.",
                                "gaps": [
                                    {
                                        "key": "gap-1",
                                        "options": ["go", "goes"],
                                        "answers": ["goes"],
                                    }
                                ],
                            }
                        ]
                    },
                    "settings": {
                        "shuffleItems": False,
                        "shuffleOptions": True,
                        "caseSensitive": False,
                    },
                    "scoring": {"mode": "per_item", "maxScore": 1},
                    "metadata": {"version": 1, "status": "draft"},
                }
            )
        ]
    )
    provider_manager = SimpleNamespace(
        provider="test-provider",
        generate_structured=AsyncMock(
            return_value=AIResult(
                data=generated,
                provider="test-provider",
                provider_host="https://provider.test",
                model="test-model",
                response_id="resp_exercise",
                usage=AIUsage(input_tokens=20, output_tokens=30, total_tokens=50),
            )
        )
    )
    client = SimpleNamespace()
    add_generation = MagicMock()
    monkeypatch.setattr(
        manager_module.exercise_service, "add_generation", add_generation
    )
    request = GenerateExercisesRequest(
        topic="Present Simple", count=1, tags=["grammar"]
    )

    generation = await exercise_manager.generate(
        session, client, provider_manager, uuid4(), request
    )

    assert generation.language == "en"
    assert generation.exercises[0]["content"]["items"][0]["text"] == (
        "She {{gap-1}} to school every day."
    )
    assert generation.provider_response_id == "resp_exercise"
    assert generation.model == "test-model"
    assert generation.total_tokens == 50
    add_generation.assert_called_once_with(session, generation)
    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()
    provider_manager.generate_structured.assert_awaited_once()
