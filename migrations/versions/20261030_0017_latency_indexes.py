"""Indexes for latest learning evidence, sessions and topic performance.
Revision ID: 20261030_0017
Revises: 20261030_0016
"""
from alembic import op
import sqlalchemy as sa
revision = '20261030_0017'
down_revision = '20261030_0016'
branch_labels = None
depends_on = None
INDEXES = [
 ('ix_test_history_user_date_id','test_history',['user_id','date','id']),
 ('ix_test_history_user_topic_id','test_history',['user_id','topic','id']),
 ('ix_topic_performance_user_topic','topic_performance',['user_id','topic']),
 ('ix_session_details_test_id','session_details',['test_id']),
]
def upgrade():
 for name,table,columns in INDEXES:
  if name not in {i['name'] for i in sa.inspect(op.get_bind()).get_indexes(table)}:
   op.create_index(name,table,columns)
def downgrade():
 for name,table,_ in reversed(INDEXES):
  if name in {i['name'] for i in sa.inspect(op.get_bind()).get_indexes(table)}:
   op.drop_index(name,table_name=table)
