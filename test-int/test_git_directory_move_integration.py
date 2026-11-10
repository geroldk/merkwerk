"""A folder moves inside one Git project as one commit, all or nothing (V1).

These run the normal API lifespan against real SQLite and a real bare repository. Notes
are seeded straight into Git (prepare/publish), as a peer or restore would, so a large
folder does not cost one bm_write per note.
"""

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastmcp import Client
from httpx import ASGITransport, AsyncClient
from mcp.types import TextContent
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import BasicMemoryConfig, ProjectEntry
from basic_memory.index.git_projection import GitProjectionPipeline
from basic_memory.models import Entity
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.project_repository import ProjectRepository

from tests.git_host_support import HostSetup, host_setup as host_setup, ready

CREATED = "2025-01-02T03:04:05+00:00"
MODIFIED = "2025-02-03T04:05:06+00:00"


def note(path: str, body: str = "Body", *, times: bool = True) -> str:
    """Canonical BM bytes for a note at `path`, with or without created/modified."""
    stem = path.removesuffix(".md")
    title = stem.rpartition("/")[2]
    lines = [
        "---",
        f"title: {title}",
        "type: note",
        f"permalink: test-project/{stem.lower().replace(' ', '-')}",
    ]
    if times:
        lines += [f"created: '{CREATED}'", f"modified: '{MODIFIED}'"]
    return "\n".join([*lines, "---", "", body, ""])


def seed(repository: GitProjectRepository, files: dict[str, str]) -> dict[str, UUID]:
    """Publish notes straight into Git, keeping what is already there."""
    base = repository.snapshot()
    ids = {path: uuid4() for path in files}
    repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=base.commit_id,
            files={**base.files, **files},
            note_ids={**base.manifest.note_ids, **ids},
        )
    )
    return ids


def candidate_refs(repository: GitProjectRepository) -> int:
    return len(repository._git("for-each-ref", "refs/bm/candidates").splitlines())


def commits(repository: GitProjectRepository) -> int:
    return int(repository._git("rev-list", "--count", "main").strip())


async def files_written(project_id: str, timeout: float = 30) -> None:
    """Wait until the files stage wrote the current head; graph/vectors may still run."""
    host = resolve_container().git_host
    assert host is not None
    worker = host.runtime_for(project_id).worker
    async with asyncio.timeout(timeout):
        while True:
            status = await worker.status()
            row = next((row for row in status.stages if row.stage == "files"), None)
            if (
                row is not None
                and row.completed_commit == status.source_commit
                and row.running_commit is None
                and row.last_error is None
            ):
                return
            worker.notify()
            await asyncio.sleep(0.05)


def runtime(project_id: str):
    host = resolve_container().git_host
    assert host is not None
    return host.runtime_for(project_id)


async def move(client: AsyncClient, project_id: str, source: str, destination: str):
    return await client.post(
        f"/v2/projects/{project_id}/knowledge/git-move-directory",
        json={"source_directory": source, "destination_directory": destination},
    )


async def entities(config: BasicMemoryConfig) -> dict[str, tuple[str, str | None, int]]:
    """external_id -> (file_path, permalink, id) as SQL holds it now."""
    _, factory = await db.get_or_create_db(config.database_path)
    async with factory() as session:
        rows = await session.execute(
            select(Entity.external_id, Entity.file_path, Entity.permalink, Entity.id)
        )
        return {external_id: (path, permalink, id_) for external_id, path, permalink, id_ in rows}


async def entity_times(config: BasicMemoryConfig) -> dict[str, tuple[Any, Any]]:
    _, factory = await db.get_or_create_db(config.database_path)
    async with factory() as session:
        rows = await session.execute(
            select(Entity.external_id, Entity.created_at, Entity.updated_at)
        )
        return {external_id: (created, updated) for external_id, created, updated in rows}


@pytest.fixture
def git_host(host_setup, config_manager) -> HostSetup:
    """The Git host as deployed: moved notes keep their permalinks (the shipped default).

    The shared test configuration turns update_permalinks_on_move on, which the folder
    move refuses by design.
    """
    host_setup.config.update_permalinks_on_move = False
    config_manager.save_config(host_setup.config)
    return host_setup


# --- One commit, every note ---


@pytest.mark.asyncio
async def test_a_folder_moves_as_one_commit_and_the_answer_lists_every_note(git_host):
    host_setup = git_host
    from basic_memory.api.app import app

    repository = host_setup.repository
    project_id = host_setup.project.external_id
    ids = seed(
        repository,
        {
            "drafts/a.md": note("drafts/a.md"),
            "drafts/sub/b.md": note("drafts/sub/b.md"),
            # Predates BM-010: it gets its indexed times, not the commit time.
            "drafts/old.md": note("drafts/old.md", times=False),
            "keep/c.md": note("keep/c.md"),
        },
    )
    before = repository.snapshot()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            sql_before = await entities(host_setup.config)
            times_before = await entity_times(host_setup.config)
            refs_before = candidate_refs(repository)
            commits_before = commits(repository)

            response = await move(client, project_id, "drafts", "archive/drafts")

            assert response.status_code == 200, response.text
            answer = response.json()
            after = repository.snapshot()
            # Exactly one new commit on top of the basis, with a message a person can read.
            assert commits(repository) == commits_before + 1
            assert (
                repository._git("rev-list", "--count", f"{before.commit_id}..{after.commit_id}")
                == b"1\n"
            )
            assert repository._git("log", "-1", "--format=%s").decode().strip() == (
                "Move drafts/ to archive/drafts/ (3 notes)"
            )
            assert answer["commit_id"] == after.commit_id
            assert answer["source"] == "drafts" and answer["destination"] == "archive/drafts"
            assert answer["total_notes"] == 3
            assert [entry["previous_file_path"] for entry in answer["moved_notes"]] == [
                "drafts/a.md",
                "drafts/old.md",
                "drafts/sub/b.md",
            ]
            for entry in answer["moved_notes"]:
                old = entry["previous_file_path"]
                assert entry["file_path"] == "archive/" + old
                assert entry["external_id"] == str(ids[old])
                assert entry["permalink"] == "test-project/" + old.removesuffix(".md")
            # The UUID travels with the note; notes outside the folder are untouched.
            assert after.manifest.note_ids == {
                **{"archive/" + old: ids[old] for old in ids if old.startswith("drafts/")},
                "keep/c.md": ids["keep/c.md"],
            }
            assert after.files["archive/drafts/a.md"] == before.files["drafts/a.md"]
            assert after.files["archive/drafts/sub/b.md"] == before.files["drafts/sub/b.md"]
            assert after.files["keep/c.md"] == before.files["keep/c.md"]
            stamped = after.files["archive/drafts/old.md"]
            assert "created: '" in stamped and "modified: '" in stamped
            # SQL follows before the answer: new paths, same ids, permalinks and times.
            sql_after = await entities(host_setup.config)
            for old, external_id in ids.items():
                path, permalink, entity_id = sql_after[str(external_id)]
                expected = "archive/" + old if old.startswith("drafts/") else old
                assert path == expected
                assert (permalink, entity_id) == sql_before[str(external_id)][1:]
            # Frontmatter times have whole seconds; the untimed note's row had more.
            assert await entity_times(host_setup.config) == {
                key: tuple(value.replace(microsecond=0) for value in values)
                for key, values in times_before.items()
            }
            read = await client.get(
                f"/v2/projects/{project_id}/knowledge/entities/{ids['drafts/a.md']}"
            )
            assert read.json()["file_path"] == "archive/drafts/a.md"
            # The private candidate ref of the move is gone again.
            assert candidate_refs(repository) == refs_before

            await ready(client, project_id)
            root = Path(host_setup.project.path)
            assert (root / "archive/drafts/sub/b.md").read_text() == before.files["drafts/sub/b.md"]
            assert not (root / "drafts/a.md").exists()
            # No empty-folder cleanup exists anywhere: the old folders stay on disk.
            assert (root / "drafts/sub").is_dir()
            found = await client.request(
                "QUERY",
                f"/v2/projects/{project_id}/search/",
                json={"text": "Body", "retrieval_mode": "fts"},
            )
            paths = {row["file_path"] for row in found.json()["results"]}
            assert "archive/drafts/a.md" in paths and "drafts/a.md" not in paths

    # A restart reads the same paths from Git and SQL.
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            assert repository.snapshot().manifest.note_ids == after.manifest.note_ids
            read = await client.get(
                f"/v2/projects/{project_id}/knowledge/entities/{ids['drafts/sub/b.md']}"
            )
            assert read.json()["file_path"] == "archive/drafts/sub/b.md"
            assert (await entities(host_setup.config)) == sql_after
            # The moved note keeps its permalink, so a new note at the old path gets another.
            written = await client.post(
                f"/v2/projects/{project_id}/knowledge/write",
                json={
                    "note": {"title": "a", "directory": "drafts", "content": "New"},
                    "allow_duplicate_title": True,
                },
            )
            assert written.status_code == 200, written.text
            assert written.json()["entity"]["file_path"] == "drafts/a.md"
            assert written.json()["entity"]["permalink"] == "test-project/drafts/a-1"


@pytest.mark.asyncio
async def test_the_upstream_folder_move_stays_refused_on_a_git_host(host_setup):
    from basic_memory.api.app import app

    seed(host_setup.repository, {"drafts/a.md": note("drafts/a.md")})
    head = host_setup.repository.head()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            response = await client.post(
                f"/v2/projects/{host_setup.project.external_id}/knowledge/move-directory",
                json={"source_directory": "drafts", "destination_directory": "archive"},
            )
            assert response.status_code == 403, response.text
    assert host_setup.repository.head() == head


# --- Refusals: nothing changes ---


async def state(
    setup: HostSetup,
) -> tuple[str, dict[str, tuple[str, str | None, int]], int, set[str]]:
    """Head, SQL rows, candidate refs and the files on disk: what a refusal must not touch."""
    root = Path(setup.project.path)
    on_disk = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".basic-memory" not in path.parts
    }
    return (
        setup.repository.head(),
        await entities(setup.config),
        candidate_refs(setup.repository),
        on_disk,
    )


@pytest.mark.asyncio
async def test_every_refusal_names_its_offenders_and_changes_nothing(git_host):
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    root = Path(git_host.project.path)
    seed(
        git_host.repository,
        {
            path: note(path)
            for path in (
                "src/a.md",
                "src/b.md",
                "src/sub/c.md",
                "dst/a.md",
                "dst2/B.md",
                "src3/Foo Bar.md",
                "dst3/foo-bar.md",
                "Drafts/x.md",
                "drafts/y.md",
                "archive/z.md",
            )
        },
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            before = await state(git_host)

            async def refused(source, destination, status, reason, conflicts=None):
                response = await move(client, project_id, source, destination)
                assert response.status_code == status, response.text
                detail = response.json()["detail"]
                assert detail["reason"] == reason, detail
                if conflicts is not None:
                    assert [
                        (conflict["path"], conflict["reason"], conflict["other"])
                        for conflict in detail["conflicts"]
                    ] == conflicts
                assert await state(git_host) == before
                return detail

            # A note outside the folder holds a target, exactly or by an equivalent name.
            await refused("src", "dst", 409, "occupied", [("dst/a.md", "occupied", "dst/a.md")])
            await refused(
                "src", "dst2", 409, "occupied", [("dst2/b.md", "equivalent", "dst2/B.md")]
            )
            await refused(
                "src3",
                "dst3",
                409,
                "occupied",
                [("dst3/Foo Bar.md", "equivalent", "dst3/foo-bar.md")],
            )
            await refused("src", "src/inner", 422, "inside_source")
            await refused("src", "src", 422, "same_path")
            await refused("/", "x", 422, "project_root")
            await refused("src", "/", 422, "project_root")
            await refused("missing", "x", 404, "not_found")
            await refused("src", "a/../b", 422, "invalid_path")
            await refused("src", ".git", 422, "invalid_path")
            await refused("archive", "Archive", 422, "case_only")
            await refused("DRAFTS", "x", 409, "ambiguous_source")

            # On disk: an attachment, an editor change and a symlink in the source.
            (root / "src/img.png").write_bytes(b"png")
            edited = (root / "src/b.md").read_text()
            (root / "src/b.md").write_text(edited + "Changed in an editor\n")
            (root / "src/link.md").symlink_to(root / "src/a.md")
            before = await state(git_host)
            detail = await refused(
                "src",
                "fresh",
                422,
                "unpublished_file",
                [
                    ("src/b.md", "edited_on_disk", None),
                    ("src/img.png", "unpublished_file", None),
                    ("src/link.md", "symlink", None),
                ],
            )
            assert "3 file(s) in the folder are not clean published notes" in detail["message"]
            (root / "src/img.png").unlink()
            (root / "src/link.md").unlink()
            (root / "src/b.md").write_text(edited)
            # A folder that holds only attachments is named, not "not found".
            (root / "attachments").mkdir()
            (root / "attachments/scan.pdf").write_bytes(b"pdf")
            before = await state(git_host)
            await refused(
                "attachments",
                "x",
                422,
                "unpublished_file",
                [("attachments/scan.pdf", "unpublished_file", None)],
            )
            # A file where the destination folder must go.
            (root / "blocked").write_text("not a folder")
            before = await state(git_host)
            await refused(
                "src",
                "blocked",
                409,
                "occupied",
                [
                    ("blocked/a.md", "file_folder_clash", "blocked"),
                    ("blocked/b.md", "file_folder_clash", "blocked"),
                    ("blocked/sub/c.md", "file_folder_clash", "blocked"),
                ],
            )
            (root / "blocked").unlink()
            # After all of that the same folder still moves, and the files stage is free.
            response = await move(client, project_id, "src", "fresh")
            assert response.status_code == 200, response.text
            await ready(client, project_id)
            assert (root / "fresh/sub/c.md").is_file()


@pytest.mark.asyncio
async def test_a_host_that_renames_permalinks_on_move_refuses_the_folder_move(host_setup):
    from basic_memory.api.app import app

    assert host_setup.config.update_permalinks_on_move
    seed(host_setup.repository, {"src/a.md": note("src/a.md")})
    head = host_setup.repository.head()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            response = await move(client, host_setup.project.external_id, "src", "dst")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["reason"] == "unsupported_config"
    assert host_setup.repository.head() == head


@pytest.mark.asyncio
async def test_a_defect_outside_the_folder_does_not_block_it_but_one_inside_does(git_host):
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    blank_title = "---\ntitle: '   '\ntype: note\npermalink: test-project/{name}\n---\n\nBody\n"
    seed(
        git_host.repository,
        {
            "src/a.md": note("src/a.md"),
            "elsewhere/blank.md": blank_title.format(name="elsewhere/blank"),
            "bad/blank.md": blank_title.format(name="bad/blank"),
        },
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            # A blank title blocks the graph stage for good; the files stage is enough here.
            await files_written(project_id)
            head = git_host.repository.head()
            response = await move(client, project_id, "bad", "worse")
            assert response.status_code == 422, response.text
            detail = response.json()["detail"]
            assert detail["reason"] == "invalid_draft"
            assert "worse/blank.md" in detail["message"]
            assert git_host.repository.head() == head
            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 200, response.text


# --- All or nothing ---


@pytest.mark.asyncio
async def test_a_failure_before_or_at_publication_leaves_git_sql_and_disk_unchanged(
    git_host, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.repository.git_project_repository import GitPublicationConflict

    project_id = git_host.project.external_id
    seed(git_host.repository, {"src/a.md": note("src/a.md"), "src/b.md": note("src/b.md")})
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await ready(client, project_id)
            repository = runtime(project_id).pipeline.repository
            before = await state(git_host)
            publish = GitProjectRepository.publish
            prepare = repository.prepare

            def broken_publish(self, prepared):
                raise RuntimeError("Injected failure at publication")

            monkeypatch.setattr(GitProjectRepository, "publish", broken_publish)
            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 500
            assert await state(git_host) == before

            def conflicting_publish(self, prepared):
                raise GitPublicationConflict(prepared.base_commit, "f" * 40)

            monkeypatch.setattr(GitProjectRepository, "publish", conflicting_publish)
            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 409, response.text
            assert response.json()["detail"]["reason"] == "base_advanced"
            assert await state(git_host) == before
            monkeypatch.setattr(GitProjectRepository, "publish", publish)

            def refused_prepare(**kwargs):
                raise ValueError("Injected identity refusal")

            monkeypatch.setattr(repository, "prepare", refused_prepare)
            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 422, response.text
            assert response.json()["detail"]["reason"] == "invalid_draft"
            assert await state(git_host) == before
            monkeypatch.setattr(repository, "prepare", prepare)

            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_a_commit_from_outside_bm_during_the_move_refuses_it_and_a_retry_moves(
    git_host, monkeypatch
):
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    seed(git_host.repository, {"src/a.md": note("src/a.md")})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            repository = runtime(project_id).pipeline.repository
            prepare = repository.prepare
            # Another handle on the same bare repository: a manual update-ref or a push.
            peer = GitProjectRepository(git_host.repository.path)
            peer_refs: list[str] = []

            def prepare_then_peer_commits(**kwargs):
                prepared = prepare(**kwargs)
                base = peer.snapshot()
                peer_prepared = peer.prepare(
                    transaction_id=uuid4(),
                    base_commit=base.commit_id,
                    files={**base.files, "peer/p.md": note("peer/p.md")},
                    note_ids={**base.manifest.note_ids, "peer/p.md": uuid4()},
                )
                peer.publish(peer_prepared)
                peer_refs.append(peer_prepared.candidate_ref)
                return prepared

            refs_before = set(
                git_host.repository._git("for-each-ref", "--format=%(refname)").split()
            )
            monkeypatch.setattr(repository, "prepare", prepare_then_peer_commits)
            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 409, response.text
            assert response.json()["detail"]["reason"] == "base_advanced"
            snapshot = git_host.repository.snapshot()
            assert "src/a.md" in snapshot.files and "dst/a.md" not in snapshot.files
            # The move's candidate ref is gone; only the peer's own is new.
            refs_after = set(
                git_host.repository._git("for-each-ref", "--format=%(refname)").split()
            )
            assert {
                ref.decode() for ref in refs_after - refs_before if b"candidates" in ref
            } == set(peer_refs)
            monkeypatch.setattr(repository, "prepare", prepare)

            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 200, response.text
            assert set(git_host.repository.snapshot().files) == {"dst/a.md", "peer/p.md"}


@pytest.mark.asyncio
async def test_sql_that_lags_after_publication_answers_503_and_the_worker_catches_up(
    git_host, monkeypatch
):
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    ids = seed(git_host.repository, {"src/a.md": note("src/a.md")})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            head = git_host.repository.head()
            recover = GitProjectionPipeline.recover_content

            async def lagging(self, commit_id, **kwargs):
                if kwargs.get("known_ancestor") is not None:
                    raise RuntimeError("Injected SQL failure after publication")
                await recover(self, commit_id, **kwargs)

            monkeypatch.setattr(GitProjectionPipeline, "recover_content", lagging)
            response = await move(client, project_id, "src", "dst")
            assert response.status_code == 503, response.text
            assert response.json()["detail"]["published_commit"] == git_host.repository.head()
            assert git_host.repository.head() != head
            await ready(client, project_id)
            assert (await entities(git_host.config))[str(ids["src/a.md"])][0] == "dst/a.md"


@pytest.mark.asyncio
async def test_a_stopped_runtime_refuses_the_move_as_not_accepting(git_host):
    from basic_memory.api.app import app
    from basic_memory.services.git_directory_moves import GitDirectoryMoveRefused

    seed(git_host.repository, {"src/a.md": note("src/a.md")})
    async with app.router.lifespan_context(app):
        stopped = runtime(git_host.project.external_id)
    with pytest.raises(GitDirectoryMoveRefused) as error:
        await stopped.move_directory("src", "dst")
    assert (error.value.status_code, error.value.reason) == (409, "not_accepting")
    assert "Nothing was moved" in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["content", "extra_note"])
async def test_sql_outside_the_published_basis_refuses_before_publication(git_host, drift):
    from basic_memory.api.app import app
    from sqlalchemy import update

    from basic_memory.models import NoteContent

    project_id = git_host.project.external_id
    seed(git_host.repository, {"src/a.md": note("src/a.md")})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            head = git_host.repository.head()
            _, factory = await db.get_or_create_db(git_host.config.database_path)
            async with db.scoped_session(factory) as session:
                if drift == "content":
                    await session.execute(update(NoteContent).values(markdown_content="Drifted"))
                else:
                    session.add(
                        Entity(
                            project_id=git_host.project.id,
                            title="Stray",
                            note_type="note",
                            content_type="text/markdown",
                            file_path="stray.md",
                        )
                    )
            response = await move(client, project_id, "src", "dst")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["reason"] == "not_accepting"
    assert "outside the captured Git basis" in response.json()["detail"]["message"]
    assert git_host.repository.head() == head


@pytest.mark.asyncio
async def test_an_open_transaction_conflicts_at_its_commit_and_keeps_its_draft(git_host):
    """Decision: open drafts do not block a folder move; they conflict, as after any write."""
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    seed(git_host.repository, {"src/a.md": note("src/a.md")})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            prefix = f"/v2/projects/{project_id}/git/transactions"
            started = await client.post(prefix, json={})
            assert started.status_code == 200, started.text
            transaction = started.json()
            keys = {"X-BM-Transaction-Key": transaction["transaction_key"]}
            written = await client.post(
                f"{prefix}/{transaction['transaction_id']}/write",
                headers=keys,
                json={"path": "draft.md", "content": note("draft.md"), "expected_revision": 0},
            )
            assert written.status_code == 200, written.text

            assert (await move(client, project_id, "src", "dst")).status_code == 200
            committed = await client.post(
                f"{prefix}/{transaction['transaction_id']}/commit",
                headers=keys,
                json={"expected_revision": 1},
            )

            assert committed.status_code == 409, committed.text
            status = await client.get(f"{prefix}/{transaction['transaction_id']}", headers=keys)
            assert status.json()["state"] == "conflicted" and status.json()["draft_available"]
            assert "draft.md" not in git_host.repository.snapshot().files


# --- Disk and paths that only a folder move meets ---


@pytest.mark.asyncio
async def test_a_folder_moving_up_into_its_parent_swaps_paths_on_disk(git_host):
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    files = {"a/b/x.md": note("a/b/x.md", "Upper"), "a/b/b/x.md": note("a/b/b/x.md", "Lower")}
    ids = seed(git_host.repository, files)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            response = await move(client, project_id, "a/b", "a")
            assert response.status_code == 200, response.text
            await ready(client, project_id)
    root = Path(git_host.project.path)
    assert git_host.repository.snapshot().manifest.note_ids == {
        "a/x.md": ids["a/b/x.md"],
        "a/b/x.md": ids["a/b/b/x.md"],
    }
    assert (root / "a/x.md").read_text() == files["a/b/x.md"]
    assert (root / "a/b/x.md").read_text() == files["a/b/b/x.md"]
    assert not (root / "a/b/b/x.md").exists()


@pytest.mark.asyncio
async def test_old_files_still_due_for_deletion_do_not_block_the_move(git_host, monkeypatch):
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    ids = seed(
        git_host.repository,
        {
            "src/a.md": note("src/a.md"),
            "src/gone.md": note("src/gone.md"),
            "dst/a.md": note("dst/a.md"),
        },
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            # Hold the files stage, so the old files of two single moves stay on disk.
            apply = GitProjectionPipeline.apply
            release = asyncio.Event()

            async def held(self, stage, commit_id, *, graph_base_commit=None):
                if stage == "files":
                    await release.wait()
                await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)

            monkeypatch.setattr(GitProjectionPipeline, "apply", held)
            base = f"/v2/projects/{project_id}/knowledge/entities"
            for old, new in (("src/gone.md", "else/gone.md"), ("dst/a.md", "else/a.md")):
                moved = await client.put(f"{base}/{ids[old]}/move", json={"destination_path": new})
                assert moved.status_code in {200, 202}, moved.text
            root = Path(git_host.project.path)
            assert (root / "src/gone.md").exists() and (root / "dst/a.md").exists()

            response = await move(client, project_id, "src", "dst")

            assert response.status_code == 200, response.text
            release.set()
            await ready(client, project_id)
    assert not (root / "src/gone.md").exists()
    assert (root / "dst/a.md").read_text() == git_host.repository.snapshot().files["dst/a.md"]
    assert (root / "else/gone.md").is_file()


@pytest.mark.asyncio
async def test_a_write_during_a_folder_move_waits_and_lands_on_the_new_basis(git_host, monkeypatch):
    from basic_memory.api.app import app
    from threading import Event

    project_id = git_host.project.external_id
    seed(git_host.repository, {"src/a.md": note("src/a.md")})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            repository = runtime(project_id).pipeline.repository
            prepare = repository.prepare
            entered, release = Event(), Event()

            def slow_prepare(**kwargs):
                entered.set()
                assert release.wait(10)
                return prepare(**kwargs)

            monkeypatch.setattr(repository, "prepare", slow_prepare)
            folder = asyncio.create_task(move(client, project_id, "src", "dst"))
            await asyncio.to_thread(entered.wait, 10)
            write = asyncio.create_task(
                client.post(
                    f"/v2/projects/{project_id}/knowledge/write",
                    json={"note": {"title": "During", "directory": "notes", "content": "x"}},
                )
            )
            await asyncio.sleep(0.2)
            assert not write.done()
            release.set()
            moved, written = await folder, await write
            assert moved.status_code == 200, moved.text
            assert written.status_code == 200, written.text
    assert set(git_host.repository.snapshot().files) == {"dst/a.md", "notes/During.md"}


@pytest.mark.asyncio
async def test_links_after_a_folder_move(git_host):
    """Links by permalink or title still resolve; links by file path do not (ENTWURF §V1).

    Bodies are never rewritten. A path link ('/x.md', '../x.md') names a file relative
    to the linking note and never falls back to the permalink.
    """
    from basic_memory.api.app import app

    project_id = git_host.project.external_id
    ids = seed(
        git_host.repository,
        {
            "keep/linker.md": note("keep/linker.md"),
            "drafts/inner.md": note("drafts/inner.md"),
            "drafts/target.md": note("drafts/target.md"),
        },
    )

    async def resolve(client, identifier: str, source_path: str) -> str | None:
        response = await client.request(
            "QUERY",
            f"/v2/projects/{project_id}/knowledge/links/resolve",
            json={"identifier": identifier, "source_path": source_path, "strict": True},
        )
        return response.json()["external_id"] if response.status_code == 200 else None

    async def links(client, inner: str) -> dict[str, str | None]:
        return {
            "by_permalink": await resolve(client, "test-project/drafts/target", "keep/linker.md"),
            "by_title": await resolve(client, "target", "keep/linker.md"),
            "by_path": await resolve(client, "/drafts/target.md", "keep/linker.md"),
            "relative": await resolve(client, "../keep/linker.md", inner),
        }

    target, linker = str(ids["drafts/target.md"]), str(ids["keep/linker.md"])
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
            assert await links(client, "drafts/inner.md") == {
                "by_permalink": target,
                "by_title": target,
                "by_path": target,
                "relative": linker,
            }
            # One level deeper, so the relative link climbs to a different folder.
            response = await move(client, project_id, "drafts", "archive/drafts")
            assert response.status_code == 200, response.text
            assert await links(client, "archive/drafts/inner.md") == {
                "by_permalink": target,
                "by_title": target,
                "by_path": None,
                "relative": None,
            }


@pytest.mark.asyncio
async def test_bm_move_on_a_git_host_publishes_one_commit_and_lists_every_note(git_host):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    ids = seed(
        git_host.repository,
        {"drafts/a.md": note("drafts/a.md"), "drafts/sub/b.md": note("drafts/sub/b.md")},
    )
    async with Client(mcp) as client:
        async with AsyncClient(transport=ASGITransport(_app()), base_url="http://test") as api:
            await ready(api, git_host.project.external_id)
        count = commits(git_host.repository)
        moved = await client.call_tool(
            "bm_move",
            {
                "path": "drafts",
                "destination": "archive/",
                "project": "test-project",
                "output_format": "json",
            },
        )
        assert not moved.is_error, moved.content
        block = moved.content[0]
        assert isinstance(block, TextContent)
        payload = json.loads(block.text)
        assert commits(git_host.repository) == count + 1
        assert payload["commit_id"] == git_host.repository.head()
        assert payload["moved_files"] == ["archive/drafts/a.md", "archive/drafts/sub/b.md"]
        assert [note["external_id"] for note in payload["moved_notes"]] == [
            str(ids["drafts/a.md"]),
            str(ids["drafts/sub/b.md"]),
        ]
        text = await client.call_tool(
            "bm_move",
            {"path": "archive/drafts", "destination": "drafts", "project": "test-project"},
        )
        block = text.content[0]
        assert isinstance(block, TextContent)
        lines = block.text.splitlines()
        assert lines[0].startswith("Moved directory archive/drafts to drafts (2 notes, one commit ")
        assert lines[1].startswith("- archive/drafts/a.md -> drafts/a.md (id ")
        refused = await client.call_tool(
            "bm_move",
            {"path": "drafts", "destination": "Drafts", "project": "test-project"},
            raise_on_error=False,
        )
        assert refused.is_error
        assert "INVALID_ARGUMENTS" in str(refused.content) and "two moves" in str(refused.content)
    assert commits(git_host.repository) == count + 2


def _app():
    from basic_memory.api.app import app

    return app


# --- Scale ---
# Gerold's decision: no upper limit on the number of notes. These measure what that costs
# under the gate. They are benchmarks, so the default integration run leaves them out.


async def _scale_move(setup: HostSetup, folder: str, count: int, monkeypatch) -> dict[str, Any]:
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401
    from basic_memory.repository.git_project_repository import (
        GitProjectRepository as Repository,
    )
    from basic_memory.services import git_project_projection

    repository = setup.repository
    name = "Notiz zur Analyse der Programmfamilie mit einem recht langen Namen"
    files = {
        f"{folder}/{index // 100:02}/{name} {index:04}.md": note(
            f"{folder}/{index // 100:02}/{name} {index:04}.md",
            "- [fact] Skalierungsprobe\n" + "Text der Notiz. " * 40,
            times=index % 10 != 0,
        )
        for index in range(count)
    }
    seed(repository, files)
    # Graph and vectors are blocked so the measurement holds only the move itself.
    apply = GitProjectionPipeline.apply
    release = asyncio.Event()

    async def blocked(self, stage, commit_id, *, graph_base_commit=None):
        if stage in {"graph", "vectors"}:
            await release.wait()
        await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)

    monkeypatch.setattr(GitProjectionPipeline, "apply", blocked)
    timings: dict[str, float] = {"projection": 0.0, "identity_proofs": 0.0}
    project_content = git_project_projection.project_git_content
    proof = Repository.contains_published_note_identity
    proofs = 0

    async def timed_projection(*args, **kwargs):
        started = time.perf_counter()
        try:
            return await project_content(*args, **kwargs)
        finally:
            timings["projection"] += time.perf_counter() - started

    def timed_proof(self, *args, **kwargs):
        nonlocal proofs
        started = time.perf_counter()
        try:
            return proof(self, *args, **kwargs)
        finally:
            proofs += 1
            timings["identity_proofs"] += time.perf_counter() - started

    monkeypatch.setattr("basic_memory.index.git_projection.project_git_content", timed_projection)
    monkeypatch.setattr(Repository, "contains_published_note_identity", timed_proof)
    async with Client(mcp) as mcp_client:
        await files_written(setup.project.external_id, timeout=300)
        timings["projection"] = timings["identity_proofs"] = 0.0
        proofs = 0
        head = repository.head()
        started = time.perf_counter()
        moved = await mcp_client.call_tool(
            "bm_move",
            {
                "path": folder,
                "destination": f"{folder}-archiv",
                "project": "test-project",
                "output_format": "json",
            },
        )
        json_seconds = time.perf_counter() - started
        assert not moved.is_error, moved.content
        block = moved.content[0]
        assert isinstance(block, TextContent)
        payload = json.loads(block.text)
        assert payload["total_files"] == count and len(payload["moved_notes"]) == count
        assert repository._git("rev-list", "--count", f"{head}..main").strip() == b"1"
        json_bytes = len(block.text.encode())
        measured = dict(timings, proofs=proofs)
        timings["projection"] = timings["identity_proofs"] = 0.0
        proofs = 0
        started = time.perf_counter()
        back = await mcp_client.call_tool(
            "bm_move",
            {"path": f"{folder}-archiv", "destination": folder, "project": "test-project"},
        )
        text_seconds = time.perf_counter() - started
        assert not back.is_error, back.content
        text = back.content[0]
        assert isinstance(text, TextContent)
        assert len(text.text.splitlines()) == count + 1
        release.set()
    return {
        "notes_in_project": len(repository.snapshot().files),
        "commits": commits(repository),
        "moved": count,
        "json_seconds": round(json_seconds, 2),
        "text_seconds": round(text_seconds, 2),
        "json_bytes": json_bytes,
        "text_bytes": len(text.text.encode()),
        "first": {key: round(value, 2) for key, value in measured.items()},
        "second": {
            "projection": round(timings["projection"], 2),
            "identity_proofs": round(timings["identity_proofs"], 2),
            "proofs": proofs,
        },
    }


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.asyncio
async def test_six_hundred_notes_move_as_one_commit(git_host, monkeypatch):
    result = await _scale_move(git_host, "archiv/2026/analysen", 600, monkeypatch)
    print("V1 scale (fresh repository):", json.dumps(result))


@pytest.fixture
async def real_history_host(app_config, config_manager, tmp_path, monkeypatch):
    """A copy of a real project repository with long history, as its own SQL project.

    Opt in with V1_SCALE_GIT_REPO=<bare repository>. The source is only read (the
    sandbox mounts it read-only); the copy lives in this test's tmp_path.
    """
    source = os.environ.get("V1_SCALE_GIT_REPO")
    if not source:
        pytest.skip("Set V1_SCALE_GIT_REPO to a bare project repository to run this benchmark")
    copy = tmp_path / "project.git"
    # The copy gets a new owner: the source's root binding and lock stay behind.
    shutil.copytree(
        source,
        copy,
        symlinks=True,
        ignore=shutil.ignore_patterns("bm-local-binding.json", "bm-projection.lock"),
    )
    repository = GitProjectRepository(copy)
    project_uuid = repository.snapshot().manifest.project_external_id
    await db.shutdown_db()
    root = tmp_path / "notes"
    root.mkdir()
    app_config.projects = {"test-project": ProjectEntry(path=str(root))}
    app_config.index_changes = True
    app_config.update_permalinks_on_move = False
    monkeypatch.setattr(BasicMemoryConfig, "is_test_env", property(lambda self: False))
    _, factory = await db.get_or_create_db(app_config.database_path)
    async with db.scoped_session(factory) as session:
        project = await ProjectRepository().create(
            session,
            {
                "name": "test-project",
                "path": str(root),
                "is_active": True,
                "is_default": True,
                "external_id": str(project_uuid),
            },
        )
    app_config.git_repositories = {project_uuid: repository.path}
    config_manager.save_config(app_config)
    try:
        yield HostSetup(app_config, project, repository)
    finally:
        await db.shutdown_db()


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.asyncio
async def test_six_hundred_notes_move_on_a_repository_with_long_history(
    real_history_host, monkeypatch
):
    result = await _scale_move(real_history_host, "v1-skalierung/analysen", 600, monkeypatch)
    print("V1 scale (real history):", json.dumps(result))
