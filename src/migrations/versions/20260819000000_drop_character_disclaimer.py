"""drop per-character disclaimer (moved to shared frontend text)

Revision ID: 20260819000000
Revises: 20260818000000
Create Date: 2026-08-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260819000000"
down_revision: str | None = "20260818000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("characters", "disclaimer")


def downgrade() -> None:
    op.add_column(
        "characters",
        sa.Column("disclaimer", sa.String(length=500), nullable=False, server_default=""),
    )
    op.alter_column("characters", "disclaimer", server_default=None)
