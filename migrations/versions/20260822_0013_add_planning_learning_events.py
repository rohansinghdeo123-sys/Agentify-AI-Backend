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


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {
        item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table_name)
    }


def _create_index_if_missing(name: str, columns: list[str]) -> None:
    if not _has_index("planning_learning_events", name):
        op.create_index(name, "planning_learning_events", columns)


def upgrade() -> None:
    if not _has_table("planning_learning_events"):
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
    _create_index_if_missing(
        "ix_planning_learning_events_id",
        ["id"],
    )
    _create_index_if_missing(
        "ix_planning_learning_events_user_id",
        ["user_id"],
    )
    _create_index_if_missing(
        "ix_planning_learning_events_curriculum_key",
        ["curriculum_key"],
    )
    _create_index_if_missing(
        "ix_planning_learning_events_unit_id",
        ["unit_id"],
    )
    _create_index_if_missing(
        "ix_planning_learning_events_user_curriculum",
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
