"""add study route states for Planning <-> Study Lab integration

Revision ID: 20261011_0015
Revises: 20260826_0014
Create Date: 2026-10-11
"""

from alembic import op
import sqlalchemy as sa


revision = "20261011_0015"
down_revision = "20260826_0014"
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


def _create_index_if_missing(name: str, table_name: str, columns: list[str]) -> None:
    if not _has_index(table_name, name):
        op.create_index(name, table_name, columns)


def upgrade() -> None:
    if not _has_table("study_route_states"):
        op.create_table(
            "study_route_states",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("route_date", sa.Date(), nullable=False),
            sa.Column("class_level", sa.String(), nullable=False, server_default=""),
            sa.Column("subject", sa.String(), nullable=False, server_default=""),
            sa.Column("selections_json", sa.JSON(), nullable=True),
            sa.Column("study_time_today", sa.String(), nullable=True),
            sa.Column("session_duration_minutes", sa.Integer(), nullable=True),
            sa.Column("goal_json", sa.JSON(), nullable=True),
            sa.Column("active_curriculum_key", sa.String(), nullable=False, server_default=""),
            sa.Column("active_chapter_slug", sa.String(), nullable=False, server_default=""),
            sa.Column("current_unit_id", sa.String(), nullable=False, server_default=""),
            sa.Column("current_step_index", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("current_step_role", sa.String(), nullable=False, server_default=""),
            sa.Column("last_position_json", sa.JSON(), nullable=True),
            sa.Column("override_reason", sa.String(), nullable=False, server_default=""),
            sa.Column("override_until_unit_id", sa.String(), nullable=False, server_default=""),
            sa.Column("effective_proficiency_json", sa.JSON(), nullable=True),
            sa.Column("needs_review_concept_ids_json", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="not_started"),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("last_active_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "route_date",
                name="uq_study_route_state_user_date",
            ),
        )

    _create_index_if_missing(
        "ix_study_route_states_id",
        "study_route_states",
        ["id"],
    )
    _create_index_if_missing(
        "ix_study_route_states_user_id",
        "study_route_states",
        ["user_id"],
    )
    _create_index_if_missing(
        "ix_study_route_states_route_date",
        "study_route_states",
        ["route_date"],
    )
    _create_index_if_missing(
        "ix_study_route_states_active_curriculum_key",
        "study_route_states",
        ["active_curriculum_key"],
    )
    _create_index_if_missing(
        "ix_study_route_states_current_unit_id",
        "study_route_states",
        ["current_unit_id"],
    )
    _create_index_if_missing(
        "ix_study_route_states_override_reason",
        "study_route_states",
        ["override_reason"],
    )
    _create_index_if_missing(
        "ix_study_route_states_status",
        "study_route_states",
        ["status"],
    )
    _create_index_if_missing(
        "ix_study_route_states_user_date",
        "study_route_states",
        ["user_id", "route_date"],
    )


def downgrade() -> None:
    if _has_table("study_route_states"):
        if _has_index("study_route_states", "ix_study_route_states_user_date"):
            op.drop_index("ix_study_route_states_user_date", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_status"):
            op.drop_index("ix_study_route_states_status", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_override_reason"):
            op.drop_index("ix_study_route_states_override_reason", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_current_unit_id"):
            op.drop_index("ix_study_route_states_current_unit_id", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_active_curriculum_key"):
            op.drop_index("ix_study_route_states_active_curriculum_key", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_route_date"):
            op.drop_index("ix_study_route_states_route_date", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_user_id"):
            op.drop_index("ix_study_route_states_user_id", table_name="study_route_states")
        if _has_index("study_route_states", "ix_study_route_states_id"):
            op.drop_index("ix_study_route_states_id", table_name="study_route_states")
        op.drop_table("study_route_states")