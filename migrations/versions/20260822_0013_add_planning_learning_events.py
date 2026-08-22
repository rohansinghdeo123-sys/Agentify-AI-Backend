"""add durable Planning learning evidence

Revision ID: 20260822_0013
Revises: 20260704_0012
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0013"
down_revision = "20260704_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "planning_learning_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("curriculum_key", sa.String(), nullable=False),
        sa.Column("unit_id", sa.String(), nullable=False),
        sa.Column("interaction_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False, server_default="study_answer"),
        sa.Column("source_session_id", sa.String(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "curriculum_key",
            "unit_id",
            "interaction_id",
            name="uq_planning_learning_event_identity",
        ),
    )
    op.create_index(
        "ix_planning_learning_events_id",
        "planning_learning_events",
        ["id"],
    )
    op.create_index(
        "ix_planning_learning_events_user_id",
        "planning_learning_events",
        ["user_id"],
    )
    op.create_index(
        "ix_planning_learning_events_curriculum_key",
        "planning_learning_events",
        ["curriculum_key"],
    )
    op.create_index(
        "ix_planning_learning_events_unit_id",
        "planning_learning_events",
        ["unit_id"],
    )
    op.create_index(
        "ix_planning_learning_events_user_curriculum",
        "planning_learning_events",
        ["user_id", "curriculum_key"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_planning_learning_events_user_curriculum",
        table_name="planning_learning_events",
    )
    op.drop_index("ix_planning_learning_events_unit_id", table_name="planning_learning_events")
    op.drop_index(
        "ix_planning_learning_events_curriculum_key",
        table_name="planning_learning_events",
    )
    op.drop_index("ix_planning_learning_events_user_id", table_name="planning_learning_events")
    op.drop_index("ix_planning_learning_events_id", table_name="planning_learning_events")
    op.drop_table("planning_learning_events")
