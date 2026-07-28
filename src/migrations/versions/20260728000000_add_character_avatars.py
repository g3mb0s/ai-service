"""add character avatars

Revision ID: 20260728000000
Revises: 20260724000000
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260728000000"
down_revision: str | None = "20260724000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "characters",
        sa.Column("avatar_url", sa.String(length=1000), nullable=True),
    )
    op.add_column(
        "characters",
        sa.Column("avatar_object_key", sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("characters", "avatar_object_key")
    op.drop_column("characters", "avatar_url")
