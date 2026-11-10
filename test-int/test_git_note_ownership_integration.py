"""One note UUID in two Git projects on one host (I1 part A): wait, refuse, never lose.

A cross-project move keeps the note's UUID: the target publishes it while the source may
still hold it (its SQL lags, or a peer synced the target first). The target then waits
for exactly those notes and projects everything else; a real duplicate is refused as
before. These tests drive the normal API lifespan with two real project repositories.
"""

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.models import Entity
from basic_memory.repository.git_project_repository import GitProjectRepository

from tests.git_host_support import add_git_project, host_setup as host_setup, ready


def canonical(title: str, path: str, extra: str = "") -> str:
    return f"---\ntitle: {title}\ntype: note\npermalink: {path[:-3]}\n{extra}---\n\nBody\n"


def publish_into(
    repository: GitProjectRepository,
    *,
    add: dict[str, tuple[str, UUID]] | None = None,
    moved: dict[UUID, UUID] | None = None,
) -> str:
    """Publish straight into a project repository, as another host or a crashed move did."""
    base = repository.snapshot()
    files, ids = dict(base.files), dict(base.manifest.note_ids)
    for path, (content, note_id) in (add or {}).items():
        files[path], ids[path] = content, note_id
    prepared = repository.prepare(
        transaction_id=uuid4(),
        base_commit=base.commit_id,
        files=files,
        note_ids=ids,
        moved_notes=moved,
    )
    return repository.publish(prepared).commit_id


async def write(client, project: str, title: str) -> UUID:
    response = await client.post(
        f"/v2/projects/{project}/knowledge/write",
        json={"note": {"title": title, "directory": "", "content": f"About {title}\n"}},
    )
    assert response.status_code == 200, response.text
    return UUID(response.json()["entity"]["external_id"])


async def owners(config, note: UUID) -> list[tuple[int, str]]:
    _, factory = await db.get_or_create_db(config.database_path)
    async with factory() as session:
        rows = await session.execute(
            select(Entity.project_id, Entity.file_path).where(Entity.external_id == str(note))
        )
        return [tuple(row) for row in rows]


async def deferred_status(client, project: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The status once the worker's own pass recorded the wait (startup records it first)."""
    async with asyncio.timeout(15):
        while True:
            status = (await client.get(f"/v2/projects/{project}/git/status")).json()
            content = next(stage for stage in status["stages"] if stage["stage"] == "content")
            if content["deferred"] and (content["last_error"] or "").startswith("Deferred"):
                return status, content
            await asyncio.sleep(0.05)


def host():
    git_host = resolve_container().git_host
    assert git_host is not None and git_host.identities is not None
    return git_host


@pytest.mark.asyncio
async def test_a_moved_note_waits_at_startup_and_arrives_when_the_source_lets_go(
    host_setup, config_manager
):
    from basic_memory.api.app import app

    a = host_setup.project.external_id
    second, b_repo = await add_git_project(host_setup, config_manager, "second")
    b = second.external_id
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = await write(client, a, "Mover")
            await ready(client, a)
            await ready(client, b)
    # The target commit of a move whose source delete has not happened yet (P4 before I1:
    # this state stopped the whole host from starting).
    publish_into(
        b_repo,
        add={"moved/mover.md": (canonical("Mover", "moved/mover.md"), note)},
        moved={note: UUID(a)},
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            status, content = await deferred_status(client, b)
            assert status["caught_up"] is False
            assert content["deferred"] == [
                {
                    "external_id": str(note),
                    "owner_project_id": a,
                    "owner_project": "test-project",
                    "owner_last_error": None,
                }
            ]
            assert content["last_error"].startswith("Deferred: waiting for project 'test-project'")
            # A write into the waiting project is a clean 409 (not 503 "SQL pending").
            refused = await client.post(
                f"/v2/projects/{b}/knowledge/write",
                json={"note": {"title": "While Waiting", "directory": "", "content": "x"}},
            )
            assert refused.status_code == 409, refused.text
            detail = refused.json()["detail"]
            assert detail["kind"] == "deferred"
            assert detail["waiting_note_ids"] == str(note)
            assert detail["owner_projects"] == "test-project"
            assert detail["owner_project_ids"] == a
            assert detail["owner_last_error"] is None
            assert "Nothing was changed" in detail["message"]
            # Other projects are untouched by the wait.
            assert (await ready(client, a))["caught_up"]
            # The source lets go. Its completed content stage wakes the waiting project:
            # no notify to it here, and its own poll would wait 30 s.
            deleted = await client.delete(f"/v2/projects/{a}/knowledge/entities/{note}")
            assert deleted.status_code in {200, 202}, deleted.text
            worker = host().runtime_for(b).worker
            async with asyncio.timeout(15):
                while not (await worker.status()).caught_up:
                    await asyncio.sleep(0.05)
            assert await owners(host_setup.config, note) == [(second.id, "moved/mover.md")]
            status = (await client.get(f"/v2/projects/{b}/git/status")).json()
            assert all(stage["deferred"] is None for stage in status["stages"])
            read = await client.get(f"/v2/projects/{b}/knowledge/entities/{note}")
            assert read.status_code == 200, read.text


@pytest.mark.asyncio
async def test_a_real_duplicate_still_refuses_the_host_start_loudly(host_setup, config_manager):
    from basic_memory.api.app import app

    a = host_setup.project.external_id
    _, b_repo = await add_git_project(host_setup, config_manager, "second")
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = await write(client, a, "Twin")
            await ready(client, a)
    # No move proof, and the owner still holds the note: a duplicate, not a move.
    publish_into(b_repo, add={"twin.md": (canonical("Twin", "twin.md"), note)})
    with pytest.raises(BaseException) as refused:
        async with app.router.lifespan_context(app):
            pytest.fail("A host with a cross-project duplicate started")
    text = repr(refused.value)
    for nested in getattr(refused.value, "exceptions", ()):
        text += repr(nested) + "".join(repr(inner) for inner in getattr(nested, "exceptions", ()))
    assert "already owned by another project" in text
    assert str(note) in text and "test-project" in text


@pytest.mark.asyncio
async def test_a_wait_names_the_owners_own_failure(host_setup, config_manager, monkeypatch):
    """An owner whose projection fails never releases: its error is shown with the wait."""
    from fastmcp import Client

    from basic_memory.api.app import app
    from basic_memory.index.git_projection import GitProjectionPipeline
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    a = host_setup.project.external_id
    second, b_repo = await add_git_project(host_setup, config_manager, "second")
    b = second.external_id
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = await write(client, a, "Stuck")
            await ready(client, a)
    publish_into(
        b_repo,
        add={"moved/stuck.md": (canonical("Stuck", "moved/stuck.md"), note)},
        moved={note: UUID(a)},
    )
    apply = GitProjectionPipeline.apply

    async def broken(self, stage, commit_id, *, graph_base_commit=None):
        if self.project_external_id == a and stage == "content":
            raise RuntimeError("Owner probe failed")
        await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)

    monkeypatch.setattr(GitProjectionPipeline, "apply", broken)
    async with Client(mcp) as mcp_client:
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            # A new head in the owner makes its (failing) content stage run.
            publish_into(
                host_setup.repository, add={"other.md": (canonical("Other", "other.md"), uuid4())}
            )
            owner = host().runtime_for(a).worker
            async with asyncio.timeout(15):
                while not any(row.last_error for row in (await owner.status()).stages):
                    owner.notify()
                    await asyncio.sleep(0.05)
            _, content = await deferred_status(client, b)
            assert content["deferred"][0]["owner_last_error"] == "Owner probe failed"
            refused = await client.post(
                f"/v2/projects/{b}/knowledge/write",
                json={"note": {"title": "Blocked", "directory": "", "content": "x"}},
            )
            assert refused.status_code == 409
            assert refused.json()["detail"]["owner_last_error"] == (
                "test-project: Owner probe failed"
            )
            status = await mcp_client.call_tool(
                "bm_project", {"command": "status", "project": "second"}
            )
            assert "test-project fails itself: Owner probe failed" in str(status.content)
            # The MCP write refusal reads "not now", naming the owner to sync.
            written = await mcp_client.call_tool(
                "bm_write",
                {"project": "second", "title": "Blocked", "directory": "", "content": "x"},
                raise_on_error=False,
            )
            assert written.is_error
            assert "UNAVAILABLE" in str(written.content)
            assert "waiting for project 'test-project'" in str(written.content)
