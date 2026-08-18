"""add character_prompt (shared base prompt flow)

Revision ID: 20260818000000
Revises: 20260728000000
Create Date: 2026-08-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260818000000"
down_revision: str | None = "20260728000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Frozen copy of the persona-neutral shared base prompt from
# domains/characters/manager.py (CHARACTER_BASE_INSTRUCTIONS). Downgrade is
# best-effort and lossy, so slight drift between this copy and the live constant
# is acceptable. The separator ("\n\nCharacter: ") is folded into LEGACY_BASE
# below so the downgrade SQL carries no inline E'\n\n...' literal.
LEGACY_BASE_INSTRUCTIONS = (
    "Roleplay as a clearly fictional AI character described below. Never claim to "
    "be a real person, never invent private information, and do not imply "
    "endorsement. The purpose is relaxed English conversation practice. Communicate "
    "strictly in English, even when the user writes in another language. Stay in "
    "character at all times, following the supplied character description. Keep the "
    "character reply in text to one to three short sentences. Ask at most one "
    "natural follow-up question. Evaluate only the English quality of the user's "
    "latest message. quality must be an integer from 0 to 10. If the message is "
    "natural and correct, use 10 and set correction and comment to empty strings. "
    "If it contains any meaningful grammar, word-choice, spelling, or naturalness "
    "issue, use 0-9, put the full natural corrected message in correction, and give "
    "a short grammar explanation in English in comment. If no correction is needed, "
    "correction and comment must both be empty. Never fill only one of those "
    "fields. The comment must be no more than three short sentences. Return only "
    "the structured object required by the schema."
)

LEGACY_BASE = LEGACY_BASE_INSTRUCTIONS + "\n\nCharacter: "


def upgrade() -> None:
    op.add_column(
        "characters",
        sa.Column("character_prompt", sa.String(length=2000), nullable=True),
    )
    op.alter_column("characters", "instructions", existing_type=sa.Text(), nullable=True)
    # Migrate the seeded Messi character to the new flow: short persona in
    # character_prompt, legacy full instructions cleared. Other rows keep their
    # stored instructions as the legacy fallback.
    op.execute(
        sa.text(
            "UPDATE characters "
            "SET character_prompt = :persona, instructions = NULL "
            "WHERE id = :character_id"
        ).bindparams(
            persona=(
                "A warm, calm, humble football legend persona who talks about "
                "football, training, teamwork, family-friendly daily life, and "
                "motivation."
            ),
            character_id="messi",
        )
    )


def downgrade() -> None:
    # Correctness of this downgrade depends on the invariant that no row has
    # both instructions and character_prompt NULL (enforced by the app layer).
    # New-flow rows (character_prompt set) are folded back into a single legacy
    # instructions string: base + character prompt. Legacy rows (character_prompt
    # NULL) are left untouched. The operation is best-effort and lossy.
    op.execute(
        sa.text(
            "UPDATE characters "
            "SET instructions = :base || COALESCE(character_prompt, '') "
            "WHERE character_prompt IS NOT NULL"
        ).bindparams(base=LEGACY_BASE)
    )
    op.alter_column("characters", "instructions", existing_type=sa.Text(), nullable=False)
    op.drop_column("characters", "character_prompt")
