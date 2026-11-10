"""Migration 0049 repairs conflict keys that a Git relocation left on the old path."""

from importlib import import_module

from alembic import command
from sqlalchemy import text
from sqlalchemy.engine import Engine

from basic_memory.utils import file_path_conflict_keys
from tests.test_entity_file_path_conflict_keys_migration import (  # noqa: F401 (fixture)
    _alembic_config,
    migration_database,
)

migration = import_module(
    "basic_memory.alembic.versions.0049_repair_entity_file_path_conflict_keys"
)
PREVIOUS_REVISION = str(migration.down_revision)

# (file_path now, path the stored keys were computed from)
ROWS = [
    # Relocated before the fix: the keys still name the old path.
    ("archive/Moved Note.md", "drafts/Moved Note.md"),
    ("Archiv/Café.md", "Café.md"),
    # Written through the ORM: already right.
    ("keep/Kept.md", "keep/Kept.md"),
]
# Enough rows that the repair pages through several id-ordered batches.
ROWS += [
    (f"bulk/{index}/Note.md", f"old/{index}/Note.md" if index % 7 == 0 else f"bulk/{index}/Note.md")
    for index in range(2 * migration.REPAIR_BATCH_SIZE + 5)
]


def _seed(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO project (id, external_id, name, permalink, path, is_active, "
                "created_at, updated_at) VALUES (1, 'project-1', 'main', 'main', '/tmp/main', "
                "true, '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO entity (id, external_id, title, note_type, content_type, "
                "project_id, permalink, file_path, file_path_key, file_path_permalink, "
                "created_at, updated_at) VALUES (:id, :external_id, 'Note', 'note', "
                "'text/markdown', 1, :permalink, :file_path, :path_key, :key_permalink, "
                "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            ),
            [
                {
                    "id": index,
                    "external_id": f"entity-{index}",
                    "permalink": f"main/stored-{index}",
                    "file_path": file_path,
                    "path_key": file_path_conflict_keys(keyed).path_key,
                    "key_permalink": file_path_conflict_keys(keyed).permalink,
                }
                for index, (file_path, keyed) in enumerate(ROWS, start=1)
            ],
        )


def _keys(engine: Engine) -> list[tuple[str, str, str]]:
    with engine.connect() as connection:
        return [
            (row[0], row[1], row[2])
            for row in connection.execute(
                text("SELECT file_path, file_path_key, file_path_permalink FROM entity ORDER BY id")
            )
        ]


def test_upgrade_recomputes_stale_keys_from_the_current_path(
    migration_database: tuple[str, Engine],  # noqa: F811 (fixture)
) -> None:
    database_url, engine = migration_database
    config = _alembic_config(database_url)
    command.upgrade(config, PREVIOUS_REVISION)
    _seed(engine)
    assert _keys(engine)[0][1] == "drafts/moved note.md"

    command.upgrade(config, migration.revision)

    repaired = _keys(engine)
    assert [row[0] for row in repaired] == [row[0] for row in ROWS]
    for file_path, path_key, permalink in repaired:
        keys = file_path_conflict_keys(file_path)
        assert (path_key, permalink) == (keys.path_key, keys.permalink), file_path
    assert repaired[0][1:] == ("archive/moved note.md", "archive/moved-note")

    # Running it again (downgrade is a no-op) finds nothing to change.
    command.downgrade(config, PREVIOUS_REVISION)
    command.upgrade(config, migration.revision)
    assert _keys(engine) == repaired
