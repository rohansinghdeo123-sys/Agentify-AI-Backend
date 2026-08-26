"""Bring the configured database to the current AgentifyAI schema safely.

The historical migration chain starts from an existing SQLAlchemy schema.
For a genuinely empty database we therefore create the current declarative
schema once and stamp it at Alembic head. Existing databases continue through
the normal, data-preserving Alembic upgrade path.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

import models  # noqa: F401 - registers every model on Base.metadata
from database import Base, engine


def _alembic_config() -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    return config


def migrate() -> str:
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    application_tables = set(Base.metadata.tables)
    config = _alembic_config()

    if not existing.intersection(application_tables):
        Base.metadata.create_all(bind=engine)
        command.stamp(config, "head")
        return "bootstrapped"

    command.upgrade(config, "head")
    return "upgraded"


if __name__ == "__main__":
    result = migrate()
    print(f"DATABASE: schema {result} to Alembic head.")
