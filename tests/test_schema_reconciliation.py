"""Regression coverage for additive repairs on long-lived databases."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

import models  # noqa: F401 - register the complete legacy schema
from database import Base


MIGRATION_PATH = (
    Path(__file__).parents[1]
    / "migrations"
    / "versions"
    / "20260826_0014_reconcile_published_source_hash.py"
)


def _migration_module():
    spec = importlib.util.spec_from_file_location("schema_reconciliation_0014", MIGRATION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_content_table(connection: sa.Connection) -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE content_chapters ("
            "id INTEGER PRIMARY KEY, source_hash VARCHAR, status VARCHAR)"
        )
    )
    connection.execute(
        sa.text(
            "INSERT INTO content_chapters (id, source_hash, status) VALUES "
            "(1, 'approved-hash', 'approved'), "
            "(2, 'published-hash', 'published'), "
            "(3, 'draft-hash', 'draft')"
        )
    )


def test_reconciliation_adds_and_backfills_missing_column() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _legacy_content_table(connection)
        migration = _migration_module()
        migration.op = Operations(MigrationContext.configure(connection))

        migration.upgrade()

        columns = {item["name"] for item in sa.inspect(connection).get_columns("content_chapters")}
        assert "published_source_hash" in columns
        rows = connection.execute(
            sa.text(
                "SELECT id, published_source_hash FROM content_chapters ORDER BY id"
            )
        ).all()
        assert rows == [(1, "approved-hash"), (2, "published-hash"), (3, "")]


def test_reconciliation_is_idempotent() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _legacy_content_table(connection)
        migration = _migration_module()
        migration.op = Operations(MigrationContext.configure(connection))

        migration.upgrade()
        migration.upgrade()

        columns = [item["name"] for item in sa.inspect(connection).get_columns("content_chapters")]
        assert columns.count("published_source_hash") == 1


def test_migration_entrypoint_bootstraps_and_reopens_a_clean_database(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    database_path = tmp_path / "clean-agentify.db"
    env = {
        **os.environ,
        "ALLOW_SQLITE_FALLBACK": "true",
        "DATABASE_URL": f"sqlite:///{database_path.as_posix()}",
    }

    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "migrate.py")],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    engine = sa.create_engine(env["DATABASE_URL"])
    with engine.connect() as connection:
        inspector = sa.inspect(connection)
        tables = set(inspector.get_table_names())
        assert {"user_progress", "content_chapters", "planning_learning_events"} <= tables
        columns = {
            item["name"] for item in inspector.get_columns("content_chapters")
        }
        assert "published_source_hash" in columns
        version = connection.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
        assert version == "20260826_0014"


def test_deployment_commands_run_schema_entrypoint_before_server() -> None:
    root = Path(__file__).parents[1]
    assert "python scripts/migrate.py && uvicorn" in (root / "Procfile").read_text()
    assert "python scripts/migrate.py && uvicorn" in (root / "Dockerfile").read_text()


def test_migration_entrypoint_accepts_empty_local_database_url(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    env = {
        **os.environ,
        "ALLOW_SQLITE_FALLBACK": "true",
        "DATABASE_URL": "",
    }
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "migrate.py")],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    engine = sa.create_engine(f"sqlite:///{(tmp_path / 'ai_educator.db').as_posix()}")
    with engine.connect() as connection:
        version = connection.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
        assert version == "20260826_0014"


def test_migration_entrypoint_upgrades_an_unversioned_legacy_database(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    database_path = tmp_path / "legacy-agentify.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    legacy_engine = sa.create_engine(database_url)
    Base.metadata.create_all(bind=legacy_engine)
    with legacy_engine.begin() as connection:
        connection.execute(
            sa.text("ALTER TABLE content_chapters DROP COLUMN published_source_hash")
        )

    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "migrate.py")],
        cwd=root,
        env={
            **os.environ,
            "ALLOW_SQLITE_FALLBACK": "true",
            "DATABASE_URL": database_url,
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with legacy_engine.connect() as connection:
        inspector = sa.inspect(connection)
        columns = {
            item["name"] for item in inspector.get_columns("content_chapters")
        }
        assert "published_source_hash" in columns
        version = connection.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
        assert version == "20260826_0014"
