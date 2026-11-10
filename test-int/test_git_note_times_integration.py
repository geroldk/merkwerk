"""BM-010: `created`/`modified` live in the accepted bytes and survive moves and rebuilds."""

import asyncio
from typing import Any
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastmcp import Client
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import ProjectEntry
from basic_memory.file_utils import parse_frontmatter
from basic_memory.models import Entity
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.project_repository import ProjectRepository

from tests.git_host_support import host_setup as host_setup, ready


@pytest.fixture
async def second_repository(host_setup, config_manager):
    root = host_setup.repository.path.parent / "second"
    root.mkdir()
    config = host_setup.config
    config.projects["second"] = ProjectEntry(path=str(root))
    _, factory = await db.get_or_create_db(config.database_path)
    async with db.scoped_session(factory) as session:
        second = await ProjectRepository().create(
            session, {"name": "second", "path": str(root), "is_active": True}
        )
    repository = GitProjectRepository.create(
        root.parent / "second.git", project_external_id=UUID(second.external_id)
    )
    config.git_repositories[UUID(second.external_id)] = repository.path
    config_manager.save_config(config)
    return second, repository


async def caught_up(*project_ids: str) -> None:
    async with asyncio.timeout(30):
        for project_id in project_ids:
            host = resolve_container().git_host
            assert host is not None
            worker = host.runtime_for(project_id).worker
            while not (await worker.status()).caught_up:
                worker.notify()
                await asyncio.sleep(0.03)


async def entity_times(path: str, database_path: Path) -> tuple[datetime, datetime]:
    _, factory = await db.get_or_create_db(database_path)
    async with db.scoped_session(factory) as session:
        entity = (await session.scalars(select(Entity).where(Entity.file_path == path))).one()
        return entity.created_at, entity.updated_at


def times(repository: GitProjectRepository, path: str) -> tuple[str, str]:
    metadata = parse_frontmatter(repository.snapshot().files[path])
    return metadata["created"], metadata["modified"]


async def call(client: Client[Any], name: str, arguments: dict[str, Any]):
    result = await client.call_tool(name, {**arguments, "output_format": "json"})
    assert not result.is_error, result.content
    return result


@pytest.mark.asyncio
async def test_write_edit_and_move_keep_the_times_in_git_and_sql(host_setup):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    repository, project = host_setup.repository, host_setup.project
    async with Client(mcp) as client:
        await call(
            client,
            "bm_write",
            {
                "project": "test-project",
                "title": "Zeit",
                "directory": "a",
                "content": "- [fact] x\n",
            },
        )
        await caught_up(project.external_id)
        created, modified = times(repository, "a/Zeit.md")
        assert datetime.fromisoformat(created).tzinfo is not None
        assert created == modified or datetime.fromisoformat(modified) >= datetime.fromisoformat(
            created
        )
        sql_created, sql_updated = await entity_times("a/Zeit.md", host_setup.config.database_path)
        assert sql_created == datetime.fromisoformat(created)
        assert sql_updated == datetime.fromisoformat(modified)

        await asyncio.sleep(1.1)  # times are stored to the second
        await call(
            client,
            "bm_edit",
            {
                "project": "test-project",
                "identifier": "a/Zeit.md",
                "operation": "append",
                "content": "\n- [fact] y\n",
            },
        )
        await caught_up(project.external_id)
        created_2, modified_2 = times(repository, "a/Zeit.md")
        assert created_2 == created  # created never changes
        assert datetime.fromisoformat(modified_2) > datetime.fromisoformat(modified)

        await asyncio.sleep(1.1)
        await call(
            client,
            "bm_move",
            {"project": "test-project", "identifier": "a/Zeit.md", "destination": "b/Zeit.md"},
        )
        await caught_up(project.external_id)
    # A plain move changes neither time, in Git or in SQL.
    assert times(repository, "b/Zeit.md") == (created_2, modified_2)
    sql_created, sql_updated = await entity_times("b/Zeit.md", host_setup.config.database_path)
    assert sql_created == datetime.fromisoformat(created_2)
    assert sql_updated == datetime.fromisoformat(modified_2)


@pytest.mark.asyncio
async def test_a_move_into_another_project_keeps_both_times(host_setup, second_repository):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    second, second_repo = second_repository
    first_repo = host_setup.repository
    async with Client(mcp) as client:
        await call(
            client,
            "bm_write",
            {
                "project": "test-project",
                "title": "Reise",
                "directory": "n",
                "content": "- [fact] r\n",
            },
        )
        await caught_up(host_setup.project.external_id)
        before = times(first_repo, "n/Reise.md")
        await asyncio.sleep(1.1)
        await call(
            client,
            "bm_move",
            {
                "project": "test-project",
                "identifier": "n/Reise.md",
                "destination": "m/Reise.md",
                "destination_project": "second",
            },
        )
        await caught_up(host_setup.project.external_id, second.external_id)
    assert times(second_repo, "m/Reise.md") == before


@pytest.mark.asyncio
async def test_a_write_that_changes_nothing_keeps_modified(host_setup):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    repository, project = host_setup.repository, host_setup.project
    async with Client(mcp) as client:
        await call(
            client,
            "bm_write",
            {
                "project": "test-project",
                "title": "Gleich",
                "directory": "g",
                "content": "- [fact] same\n",
            },
        )
        await caught_up(project.external_id)
        before = times(repository, "g/Gleich.md")
        await asyncio.sleep(1.1)
        await call(
            client,
            "bm_edit",
            {
                "project": "test-project",
                "identifier": "g/Gleich.md",
                "operation": "find_replace",
                "find_text": "same",
                "content": "same",
            },
        )
        await caught_up(project.external_id)
    assert times(repository, "g/Gleich.md") == before


@pytest.mark.asyncio
async def test_a_write_of_only_the_date_fields_never_renews_modified(host_setup):
    """The one-off backfill sets exactly the old values; later metadata edits renew them."""
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    repository, project = host_setup.repository, host_setup.project
    old_created, old_modified = "2026-09-01T10:00:00+02:00", "2026-09-23T11:31:45+02:00"
    async with Client(mcp) as client:
        await call(
            client,
            "bm_write",
            {
                "project": "test-project",
                "title": "Alt",
                "directory": "d",
                "content": "- [fact] a\n",
            },
        )
        await caught_up(project.external_id)
        await call(
            client,
            "bm_edit",
            {
                "project": "test-project",
                "identifier": "d/Alt.md",
                "operation": "append",
                "content": "",
                "metadata": {"created": old_created, "modified": old_modified},
            },
        )
        await caught_up(project.external_id)
        assert times(repository, "d/Alt.md") == (old_created, old_modified)
        sql_created, sql_updated = await entity_times("d/Alt.md", host_setup.config.database_path)
        assert sql_created == datetime.fromisoformat(old_created)
        assert sql_updated == datetime.fromisoformat(old_modified)

        # Any other frontmatter change renews modified and leaves created alone.
        await call(
            client,
            "bm_edit",
            {
                "project": "test-project",
                "identifier": "d/Alt.md",
                "operation": "append",
                "content": "",
                "metadata": {"status": "geprueft"},
            },
        )
        await caught_up(project.external_id)
    created, modified = times(repository, "d/Alt.md")
    assert created == old_created
    assert datetime.fromisoformat(modified) > datetime.fromisoformat(old_modified)


@pytest.mark.asyncio
async def test_transaction_write_and_move_stamp_and_keep_the_times(host_setup):
    from httpx import ASGITransport, AsyncClient

    from basic_memory.api.app import app

    repository = host_setup.repository
    text = "---\ntitle: Tx\ntype: note\npermalink: test-project/tx\n---\n\n- [fact] tx\n"
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as http:
            prefix = f"/v2/projects/{host_setup.project.external_id}/git/transactions"

            async def run(steps):
                started = (await http.post(prefix, json={})).json()
                head = {"X-BM-Transaction-Key": started["transaction_key"]}
                revision = 0
                for action, body in steps:
                    response = await http.post(
                        f"{prefix}/{started['transaction_id']}/{action}",
                        headers=head,
                        json={**body, "expected_revision": revision},
                    )
                    assert response.status_code == 200, response.text
                    revision = response.json()["revision"]
                committed = await http.post(
                    f"{prefix}/{started['transaction_id']}/commit",
                    headers=head,
                    json={"expected_revision": revision, "message": "t"},
                )
                assert committed.status_code == 200, committed.text
                await ready(http, host_setup.project.external_id)

            await run([("write", {"path": "t/Tx.md", "content": text, "operation": "replace"})])
            created, modified = times(repository, "t/Tx.md")
            assert datetime.fromisoformat(created).tzinfo is not None
            await asyncio.sleep(1.1)
            await run([("move", {"path": "t/Tx.md", "destination": "u/Tx.md"})])
    assert times(repository, "u/Tx.md") == (created, modified)
    sql_created, sql_updated = await entity_times("u/Tx.md", host_setup.config.database_path)
    assert sql_created == datetime.fromisoformat(created)
    assert sql_updated == datetime.fromisoformat(modified)


@pytest.mark.asyncio
async def test_a_note_without_time_keys_keeps_its_indexed_times_across_projects(
    host_setup, second_repository
):
    """I1: the source entity's times go into a note that lacks the keys; not the move time."""
    from uuid import uuid4

    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    second, second_repo = second_repository
    first_repo = host_setup.repository
    note = uuid4()
    # Published without `created`/`modified`, as notes from before BM-010 are.
    base = first_repo.snapshot()
    prepared = first_repo.prepare(
        transaction_id=uuid4(),
        base_commit=base.commit_id,
        files={"alt/Alt.md": "---\ntitle: Alt\ntype: note\npermalink: alt/alt\n---\n\nAlt\n"},
        note_ids={"alt/Alt.md": note},
    )
    first_repo.publish(prepared)
    async with Client(mcp) as client:
        await caught_up(host_setup.project.external_id)
        indexed = await entity_times("alt/Alt.md", host_setup.config.database_path)
        await asyncio.sleep(1.1)  # a move-time stamp would differ by at least a second
        moved = await call(
            client,
            "bm_move",
            {
                "project": "test-project",
                "identifier": "alt/Alt.md",
                "destination": "neu/Alt.md",
                "destination_project": "second",
            },
        )
        await caught_up(host_setup.project.external_id, second.external_id)
    assert moved.data["external_id"] == str(note) == moved.data["previous_external_id"]
    created, modified = times(second_repo, "neu/Alt.md")
    # Note times are stored to the second (format_note_time).
    assert datetime.fromisoformat(created).timestamp() == int(indexed[0].timestamp())
    assert datetime.fromisoformat(modified).timestamp() == int(indexed[1].timestamp())
