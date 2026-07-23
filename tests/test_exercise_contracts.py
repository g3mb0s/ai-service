import pytest
from pydantic import ValidationError

from domains.exercises.schemas import (
    FillGapChoiceItem,
    GenerateExercisesRequest,
    MatchingItem,
    MatchEntry,
)


def test_language_is_not_part_of_generation_request() -> None:
    request = GenerateExercisesRequest(topic="Present Simple")

    assert "language" not in request.model_dump()

    with pytest.raises(ValidationError):
        GenerateExercisesRequest.model_validate(
            {"topic": "Present Simple", "language": "ru"}
        )


def test_fill_gap_uses_content_service_marker() -> None:
    item = FillGapChoiceItem.model_validate(
        {
            "id": "item-1",
            "text": "She {{gap-1}} every day.",
            "gaps": [
                {
                    "key": "gap-1",
                    "options": ["works", "work"],
                    "answers": ["works"],
                }
            ],
        }
    )

    assert item.gaps[0].key == "gap-1"


def test_fill_gap_rejects_underscore_placeholder() -> None:
    with pytest.raises(ValidationError):
        FillGapChoiceItem.model_validate(
            {
                "id": "item-1",
                "text": "She ___ every day.",
                "gaps": [
                    {
                        "key": "gap-1",
                        "options": ["works", "work"],
                        "answers": ["works"],
                    }
                ],
            }
        )


def test_matching_pairs_reference_existing_ids() -> None:
    item = MatchingItem(
        id="matching-1",
        left=[MatchEntry(id="left-1", text="cat")],
        right=[MatchEntry(id="right-1", text="кошка")],
        pairs=[["left-1", "right-1"]],
    )

    assert item.pairs == [["left-1", "right-1"]]
