"""Persist Agent HQ runs, contributions, recommendation decisions and scheduling."""

import sqlalchemy as sa
from alembic import op

revision = "20261117_0020"
down_revision = "20261116_0019"
branch_labels = None
depends_on = None


def upgrade():
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "agent_hq_runs" not in tables:
        op.create_table(
            "agent_hq_runs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("parent_run_id", sa.String(36), sa.ForeignKey("agent_hq_runs.id", ondelete="SET NULL")),
            sa.Column("actor_uid", sa.String(128), nullable=False),
            sa.Column("request_id", sa.String(128), nullable=False),
            sa.Column("kind", sa.String(16), nullable=False),
            sa.Column("topic", sa.Text(), nullable=False),
            sa.Column("participant_ids", sa.JSON(), nullable=False),
            sa.Column("days", sa.Integer(), nullable=False),
            sa.Column("ending_on", sa.Date()),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime()),
            sa.Column("finished_at", sa.DateTime()),
            sa.Column("result", sa.JSON(), nullable=False),
            sa.UniqueConstraint("actor_uid", "request_id", name="uq_agent_hq_run_request"),
            sa.CheckConstraint("kind IN ('direct','meeting','standup','report')", name="ck_agent_hq_run_kind"),
            sa.CheckConstraint("status IN ('queued','running','completed','failed')", name="ck_agent_hq_run_status"),
            sa.CheckConstraint("days BETWEEN 1 AND 90", name="ck_agent_hq_run_days"),
        )
        op.create_index("ix_agent_hq_runs_created", "agent_hq_runs", ["created_at", "id"])
        op.create_index("ix_agent_hq_runs_status_created", "agent_hq_runs", ["status", "created_at"])
        op.create_index("ix_agent_hq_runs_updated", "agent_hq_runs", ["updated_at"])
        op.create_index("ix_agent_hq_runs_parent", "agent_hq_runs", ["parent_run_id"])
    if "agent_hq_messages" not in tables:
        op.create_table(
            "agent_hq_messages",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_hq_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("agent_id", sa.String(128)),
            sa.Column("kind", sa.String(32), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("evidence", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_agent_hq_messages_run_order", "agent_hq_messages", ["run_id", "id"])
    if "agent_hq_recommendation_decisions" not in tables:
        op.create_table(
            "agent_hq_recommendation_decisions",
            sa.Column("recommendation_id", sa.String(128), primary_key=True),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("actor_uid", sa.String(128), nullable=False),
            sa.Column("note", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
    if "agent_hq_schedules" not in tables:
        op.create_table(
            "agent_hq_schedules",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("timezone", sa.String(64), nullable=False),
            sa.Column("hour", sa.Integer(), nullable=False),
            sa.Column("minute", sa.Integer(), nullable=False),
            sa.Column("next_run_at", sa.DateTime()),
            sa.Column("actor_uid", sa.String(128), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint("id = 1", name="ck_agent_hq_schedule_singleton"),
            sa.CheckConstraint("hour BETWEEN 0 AND 23", name="ck_agent_hq_schedule_hour"),
            sa.CheckConstraint("minute BETWEEN 0 AND 59", name="ck_agent_hq_schedule_minute"),
        )


def downgrade():
    op.drop_table("agent_hq_schedules")
    op.drop_table("agent_hq_recommendation_decisions")
    op.drop_table("agent_hq_messages")
    op.drop_table("agent_hq_runs")
