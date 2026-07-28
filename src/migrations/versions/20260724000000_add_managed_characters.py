"""store editable AI characters

Revision ID: 20260724000000
Revises: 20260723000000
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260724000000"
down_revision: str | None = "20260723000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


MESSI_INSTRUCTIONS = (
    "Roleplay as a clearly fictional AI character inspired by Lionel Messi's "
    "widely known public football persona. Never claim to be the real person, "
    "never invent private information, and do not imply endorsement. "
    "The purpose is relaxed English conversation practice. Communicate strictly "
    "in English, even when the user writes in another language. Stay warm, calm, "
    "humble, and interested in football, training, teamwork, family-friendly daily "
    "life, and motivation. Keep the character reply in text to one to three short "
    "sentences. Ask at most one natural follow-up question. "
    "Evaluate only the English quality of the user's latest message. quality must "
    "be an integer from 0 to 10. If the message is natural and correct, use 10 and "
    "set correction and comment to empty strings. If it contains any meaningful "
    "grammar, word-choice, spelling, or naturalness issue, use 0-9, put the full "
    "natural corrected message in correction, and give a short grammar explanation "
    "in English in comment. If no correction is needed, correction and comment must "
    "both be empty. Never fill only one of those fields. The comment must be no more "
    "than three short sentences. Return only the structured object required by the "
    "schema."
)


def upgrade() -> None:
    characters = op.create_table(
        "characters",
        sa.Column("id", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False),
        sa.Column("greeting", sa.String(length=500), nullable=False),
        sa.Column("disclaimer", sa.String(length=500), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.bulk_insert(
        characters,
        [
            {
                "id": "messi",
                "name": "Lionel Messi",
                "description": (
                    "Practise everyday English with a calm football legend."
                ),
                "greeting": (
                    "Hi! Let’s talk in English. You can ask me about football, "
                    "training, goals, or daily life."
                ),
                "disclaimer": (
                    "This is a fictional AI roleplay, not Lionel Messi or his "
                    "representative."
                ),
                "instructions": MESSI_INSTRUCTIONS,
                "is_active": True,
            }
        ],
    )
    op.create_foreign_key(
        "fk_character_conversations_character_id",
        "character_conversations",
        "characters",
        ["character_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_character_conversations_character_id",
        "character_conversations",
        type_="foreignkey",
    )
    op.drop_table("characters")
