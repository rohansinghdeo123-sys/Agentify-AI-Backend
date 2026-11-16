"""Remove retired notes source data and cap saved unlimited routes."""
from alembic import op
import sqlalchemy as sa

revision = "20261116_0019"
down_revision = "20261108_0018"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    chapters = sa.table("content_chapters", sa.column("id"), sa.column("status"),
                        sa.column("board"), sa.column("validation_report"))
    retired = sa.or_(
        chapters.c.status == "indexed_master",
        chapters.c.board.in_(["Master Notes", "Amit Master Notes"]),
        sa.cast(chapters.c.validation_report, sa.Text).like('%"MASTER_NOTES"%'),
    )
    ids = sa.select(chapters.c.id).where(retired)
    for name in ("content_pages", "content_concepts", "content_chunks"):
        child = sa.table(name, sa.column("chapter_id"))
        connection.execute(child.delete().where(child.c.chapter_id.in_(ids)))
    connection.execute(chapters.delete().where(retired))

    cache = sa.table("adaptive_note_lessons", sa.column("payload_json"))
    connection.execute(cache.delete().where(sa.cast(cache.c.payload_json, sa.Text).like('%"MASTER_NOTES"%')))
    routes = sa.table("study_route_states", sa.column("study_time_today"))
    connection.execute(routes.update().where(routes.c.study_time_today == "no_limit").values(study_time_today="30"))


def downgrade():
    # Deleted source data is intentionally unrecoverable through schema rollback.
    pass
