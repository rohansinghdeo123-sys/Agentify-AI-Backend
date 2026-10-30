"""Persist route snapshots, assessment receipts and return-to-route positions.

Revision ID: 20261030_0016
Revises: 20261011_0015
"""
from alembic import op
import sqlalchemy as sa

revision = "20261030_0016"
down_revision = "20261011_0015"
branch_labels = None
depends_on = None


def upgrade():
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("study_route_states")}
    if "runtime_json" not in columns:
        op.add_column("study_route_states", sa.Column("runtime_json", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("study_route_states") as batch:
        batch.drop_column("runtime_json")
