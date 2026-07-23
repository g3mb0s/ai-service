"""align generated exercises with content service contract

Revision ID: 20260722002000
Revises: 20260722001000
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260722002000"
down_revision: str | None = "20260722001000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "exercise_generations",
        "items",
        new_column_name="exercises",
        existing_type=sa.JSON(),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "exercise_generations",
        "exercises",
        new_column_name="items",
        existing_type=sa.JSON(),
        existing_nullable=False,
    )
