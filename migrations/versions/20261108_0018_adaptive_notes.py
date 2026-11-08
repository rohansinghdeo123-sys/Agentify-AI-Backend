"""Source-bound adaptive notes and authoritative topic duration configuration."""
from alembic import op
import sqlalchemy as sa

revision = "20261108_0018"
down_revision = "20261030_0017"
branch_labels = None
depends_on = None


def upgrade():
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "adaptive_note_lessons" not in tables:
        op.create_table("adaptive_note_lessons",
            sa.Column("id",sa.Integer(),primary_key=True),
            sa.Column("content_version",sa.String(96),nullable=False),
            sa.Column("curriculum_key",sa.String(),nullable=False),
            sa.Column("chapter_slug",sa.String(),nullable=False),
            sa.Column("unit_id",sa.String(),nullable=False),
            sa.Column("proficiency",sa.String(),nullable=False),
            sa.Column("explanation_style",sa.String(),nullable=False),
            sa.Column("source_fingerprint",sa.String(64),nullable=False),
            sa.Column("minimum_minutes",sa.Integer(),nullable=False),
            sa.Column("maximum_minutes",sa.Integer(),nullable=False),
            sa.Column("payload_json",sa.JSON(),nullable=False),
            sa.Column("created_at",sa.DateTime()),
        )
        for column in ["content_version","curriculum_key","unit_id","source_fingerprint"]:
            op.create_index(f"ix_adaptive_note_lessons_{column}","adaptive_note_lessons",[column],unique=column=="content_version")
    if "topic_duration_overrides" not in tables:
        op.create_table("topic_duration_overrides",
            sa.Column("id",sa.Integer(),primary_key=True),
            sa.Column("curriculum_key",sa.String(),nullable=False),
            sa.Column("unit_id",sa.String(),nullable=False),
            sa.Column("minimum_minutes",sa.Integer(),nullable=False),
            sa.Column("maximum_minutes",sa.Integer(),nullable=False),
            sa.Column("configured_by",sa.String(),nullable=False),
            sa.Column("updated_at",sa.DateTime()),
            sa.UniqueConstraint("curriculum_key","unit_id",name="uq_topic_duration_scope"),
        )
        op.create_index("ix_topic_duration_overrides_curriculum_key","topic_duration_overrides",["curriculum_key"])


def downgrade():
    op.drop_table("topic_duration_overrides")
    op.drop_table("adaptive_note_lessons")
