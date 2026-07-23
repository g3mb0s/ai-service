"""add AI character conversations

Revision ID: 20260723000000
Revises: 20260722002000
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260723000000"
down_revision: str | None = "20260722002000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "character_conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("character_id", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
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
    op.create_index(
        "ix_character_conversations_user_id",
        "character_conversations",
        ["user_id"],
    )
    op.create_index(
        "ix_character_conversations_character_id",
        "character_conversations",
        ["character_id"],
    )

    op.create_table(
        "character_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("quality", sa.Integer(), nullable=True),
        sa.Column("correction", sa.Text(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("provider_host", sa.String(length=500), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("provider_response_id", sa.String(length=100), nullable=True),
        sa.Column("input_tokens", sa.BigInteger(), nullable=True),
        sa.Column("output_tokens", sa.BigInteger(), nullable=True),
        sa.Column("total_tokens", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["character_conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_character_messages_conversation_id",
        "character_messages",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_character_messages_conversation_id",
        table_name="character_messages",
    )
    op.drop_table("character_messages")
    op.drop_index(
        "ix_character_conversations_character_id",
        table_name="character_conversations",
    )
    op.drop_index(
        "ix_character_conversations_user_id",
        table_name="character_conversations",
    )
    op.drop_table("character_conversations")
