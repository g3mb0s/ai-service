"""add provider, model and token usage metadata

Revision ID: 20260722001000
Revises: 20260722000000
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260722001000"
down_revision: str | None = "20260722000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "chat_messages",
        "openai_response_id",
        new_column_name="provider_response_id",
        existing_type=sa.String(length=100),
        existing_nullable=True,
    )
    op.add_column(
        "chat_messages", sa.Column("provider", sa.String(length=50), nullable=True)
    )
    op.add_column(
        "chat_messages",
        sa.Column("provider_host", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "chat_messages", sa.Column("model", sa.String(length=100), nullable=True)
    )
    op.add_column(
        "chat_messages", sa.Column("input_tokens", sa.BigInteger(), nullable=True)
    )
    op.add_column(
        "chat_messages", sa.Column("output_tokens", sa.BigInteger(), nullable=True)
    )
    op.add_column(
        "chat_messages", sa.Column("total_tokens", sa.BigInteger(), nullable=True)
    )

    op.add_column(
        "exercise_generations",
        sa.Column(
            "provider",
            sa.String(length=50),
            server_default="openai",
            nullable=False,
        ),
    )
    op.add_column(
        "exercise_generations",
        sa.Column("provider_host", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "exercise_generations",
        sa.Column("provider_response_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "exercise_generations",
        sa.Column("input_tokens", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "exercise_generations",
        sa.Column("output_tokens", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "exercise_generations",
        sa.Column("total_tokens", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("exercise_generations", "total_tokens")
    op.drop_column("exercise_generations", "output_tokens")
    op.drop_column("exercise_generations", "input_tokens")
    op.drop_column("exercise_generations", "provider_response_id")
    op.drop_column("exercise_generations", "provider_host")
    op.drop_column("exercise_generations", "provider")

    op.drop_column("chat_messages", "total_tokens")
    op.drop_column("chat_messages", "output_tokens")
    op.drop_column("chat_messages", "input_tokens")
    op.drop_column("chat_messages", "model")
    op.drop_column("chat_messages", "provider_host")
    op.drop_column("chat_messages", "provider")
    op.alter_column(
        "chat_messages",
        "provider_response_id",
        new_column_name="openai_response_id",
        existing_type=sa.String(length=100),
        existing_nullable=True,
    )
