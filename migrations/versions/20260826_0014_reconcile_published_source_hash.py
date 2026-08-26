"""reconcile the published content source hash column

Revision ID: 20260826_0014
Revises: 20260822_0013
Create Date: 2026-08-26

Some long-lived databases were stamped past revision 0008 after their tables
had originally been created by SQLAlchemy.  Those databases can therefore be
at the current Alembic revision while still missing ``published_source_hash``.
Revision 0014 intentionally repeats the idempotent schema check so those
deployments are repaired without changing or discarding existing content.
"""

from alembic import op
import sqlalchemy as sa


revision = "20260826_0014"
down_revision = "20260822_0013"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return column_name in {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def upgrade() -> None:
    if not _has_table("content_chapters") or _has_column(
        "content_chapters", "published_source_hash"
    ):
        return

    op.add_column(
        "content_chapters",
        sa.Column("published_source_hash", sa.String(), nullable=True, server_default=""),
    )
    # A previously approved source is the only trustworthy historical value.
    # Draft rows remain empty until their first publish.
    op.execute(
        "UPDATE content_chapters SET published_source_hash = source_hash "
        "WHERE status IN ('approved', 'published')"
    )


def downgrade() -> None:
    # This is a reconciliation migration for a column owned by revision 0008.
    # Dropping it here would corrupt a normal downgrade from 0014 to 0013.
    pass
