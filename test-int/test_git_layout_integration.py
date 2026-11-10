"""Published moves/deletes, recovery and private-file protection through real BM."""

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, update

from basic_memory.file_utils import parse_frontmatter
from basic_memory.index.git_projection import GitProjectionPipeline, build_git_projection_pipeline
from basic_memory.index.git_projection_worker import GitProjectionWorker
from basic_memory.index.note_content_materialization import LocalNoteContentStorage
from basic_memory.models import Entity, GitFileRetirement, NoteContent, Project, Relation
from basic_memory.repository.git_project_repository import GitProjectRepository, GitSnapshot
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.repository.git_layout_repository import GitLayoutRepository
from basic_memory.services.git_project_projection import project_git_content


@pytest.fixture
def pipeline(tmp_path_factory, test_project, engine_factory, app_config):
    repository = GitProjectRepository.create(
        tmp_path_factory.mktemp("git-layout") / "project.git",
        project_external_id=UUID(test_project.external_id),
    )
    return build_git_projection_pipeline(
        repository, project=test_project, session_maker=engine_factory[1], app_config=app_config
    )


def initial(pipeline: GitProjectionPipeline) -> str:
    files = {
        "notes/Source.md": "---\ntitle: Source\ntype: note\npermalink: test-project/notes/source\n---\n\n- [fact] Layoutsourceprobe\n\n- relates_to [[Target]]\n",
        "notes/Target.md": "---\ntitle: Target\ntype: note\npermalink: test-project/notes/target\n---\n\nRetiredtargetprobe\n",
    }
    return publish(pipeline, files, {path: uuid4() for path in files})


def publish(pipeline: GitProjectionPipeline, files: dict[str, str], ids: dict[str, UUID]) -> str:
    repository = pipeline.repository
    return repository.publish(
        repository.prepare(
            transaction_id=uuid4(), base_commit=repository.head(), files=files, note_ids=ids
        )
    ).commit_id


async def ready(worker: GitProjectionWorker) -> None:
    async with asyncio.timeout(15):
        while not (await worker.status()).caught_up:
            await asyncio.sleep(0.02)


async def settle(pipeline: GitProjectionPipeline) -> None:
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await ready(worker)


@dataclass(frozen=True, slots=True)
class DiskEvent:
    action: str
    path: str
    # Notes with neither their old nor their new bytes at their old or new path.
    absent: frozenset[UUID]
    # Notes on disk twice (old and new copy).
    doubled: frozenset[UUID]


def watch_disk(
    monkeypatch, pipeline: GitProjectionPipeline, old: GitSnapshot, new: GitSnapshot
) -> tuple[list[DiskEvent], set[str]]:
    """After every file write and guarded delete, look at which notes are on disk.

    Scoped to the worker alone: no concurrent accept() runs during these tests.
    """
    root = pipeline.file_service.base_path
    old_paths = {external_id: path for path, external_id in old.manifest.note_ids.items()}
    wanted: dict[UUID, tuple[set[str], set[bytes]]] = {}
    for path, external_id in new.manifest.note_ids.items():
        # A note new in this commit was never on disk, so it cannot go missing.
        if (previous := old_paths.get(external_id)) is not None:
            wanted[external_id] = (
                {path, previous},
                {new.files[path].encode(), old.files[previous].encode()},
            )
    events: list[DiskEvent] = []
    statuses: set[str] = set()

    async def look(action: str, path: str) -> None:
        absent, doubled = set(), set()
        for external_id, (paths, texts) in wanted.items():
            held = sum(
                (root / candidate).is_file() and (root / candidate).read_bytes() in texts
                for candidate in paths
            )
            if held == 0:
                absent.add(external_id)
            elif held > 1:
                doubled.add(external_id)
        events.append(DiskEvent(action, path, frozenset(absent), frozenset(doubled)))
        async with pipeline.session_maker() as session:
            statuses.update(await session.scalars(select(NoteContent.file_write_status)))

    write = LocalNoteContentStorage.write_file
    delete = LocalNoteContentStorage.delete_file_if_matches

    async def watched_write(self, path, content, *, metadata=None):
        checksum = await write(self, path, content, metadata=metadata)
        await look("write", str(path))
        return checksum

    async def watched_delete(self, path, *, expected_checksum):
        outcome = await delete(self, path, expected_checksum=expected_checksum)
        await look("delete", str(path))
        return outcome

    monkeypatch.setattr(LocalNoteContentStorage, "write_file", watched_write)
    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", watched_delete)
    return events, statuses


LAYOUT_OPERATIONS = [
    "move",
    "swap",
    "delete_target",
    "delete_all",
    "reuse_path",
    "coalesce_moves",
    "move_onto_freed",
    "move_onto_freed_coalesced",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", LAYOUT_OPERATIONS)
async def test_published_layout_change_converges_content_files_graph_and_vectors(
    pipeline, operation, monkeypatch
):
    initial(pipeline)
    await settle(pipeline)
    old = pipeline.repository.snapshot()
    async with pipeline.session_maker() as session:
        before = {
            UUID(row.external_id): (row.id, row.file_path)
            for row in await session.scalars(select(Entity))
        }
    files, ids = dict(old.files), dict(old.manifest.note_ids)
    if operation in {"move", "reuse_path", "coalesce_moves"} or operation.startswith(
        "move_onto_freed"
    ):
        files["moved/Source.md"] = files.pop("notes/Source.md")
        ids["moved/Source.md"] = ids.pop("notes/Source.md")
        if operation == "move_onto_freed_coalesced":
            publish(pipeline, files, ids)
        if operation.startswith("move_onto_freed"):
            # Target moves onto the path Source just left.
            files["notes/Source.md"] = files.pop("notes/Target.md")
            ids["notes/Source.md"] = ids.pop("notes/Target.md")
    elif operation == "swap":
        files["notes/Source.md"], files["notes/Target.md"] = (
            files["notes/Target.md"],
            files["notes/Source.md"],
        )
        ids["notes/Source.md"], ids["notes/Target.md"] = (
            ids["notes/Target.md"],
            ids["notes/Source.md"],
        )
    elif operation == "delete_target":
        files.pop("notes/Target.md")
        ids.pop("notes/Target.md")
    else:
        files, ids = {}, {}
    publish(pipeline, files, ids)
    if operation == "reuse_path":
        files["notes/Source.md"] = (
            "---\ntitle: Replacement\ntype: note\npermalink: test-project/notes/replacement\n---\n\nReplacementprobe\n"
        )
        ids["notes/Source.md"] = uuid4()
        publish(pipeline, files, ids)
    elif operation == "coalesce_moves":
        files["final/Source.md"] = files.pop("moved/Source.md")
        ids["final/Source.md"] = ids.pop("moved/Source.md")
        publish(pipeline, files, ids)
    events, statuses = watch_disk(monkeypatch, pipeline, old, pipeline.repository.snapshot())
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()

    # --- No moment without the file (F2) ---
    assert events
    if operation == "swap":
        # A cycle cannot avoid a gap without a temporary path: one note is absent from
        # the first delete until its own write, which is the last event (C10).
        assert all(len(event.absent) <= 1 for event in events)
        gaps = [index for index, event in enumerate(events) if event.absent]
        assert gaps and events[gaps[0]].action == "delete" and gaps[-1] < len(events) - 1
        assert events[gaps[-1] + 1].action == "write"
    else:
        assert all(not event.absent for event in events), events
    # At most one note is on disk twice at any moment (C11).
    assert all(len(event.doubled) <= 1 for event in events)
    if operation in {"move", "coalesce_moves"}:
        new_path = "moved/Source.md" if operation == "move" else "final/Source.md"
        actions = [(event.action, event.path) for event in events]
        assert actions.index(("write", new_path)) < actions.index(("delete", "notes/Source.md"))

    async with pipeline.session_maker() as session:
        entities = {UUID(row.external_id): row for row in await session.scalars(select(Entity))}
        assert set(entities) == set(ids.values())
        assert len(entities) == len(snapshot.manifest.notes)
        assert await session.scalar(select(GitFileRetirement.project_id)) is None
        statuses.update(await session.scalars(select(NoteContent.file_write_status)))
        assert "external_change_detected" not in statuses
        for path, external_id in ids.items():
            entity = entities[external_id]
            if external_id in before:
                assert entity.id == before[external_id][0]
                assert entity.external_id == str(external_id)
            assert entity.file_path == path
            # The permalink is the published frontmatter's, never a "-1" twin.
            assert entity.permalink == parse_frontmatter(snapshot.files[path])["permalink"]
            assert entity.permalink is not None and not entity.permalink.endswith("-1")
            content = await session.get(NoteContent, entity.id)
            assert content is not None and content.file_write_status == "synced"
            assert content.file_checksum == content.db_checksum
            assert content.markdown_content == snapshot.files[path]
            assert (pipeline.file_service.base_path / path).read_bytes() == snapshot.files[
                path
            ].encode()
        live_paths = set(files)
        assert not set(await session.scalars(select(Entity.file_path))) - live_paths
        for external_id, (_, old_path) in before.items():
            if old_path not in files:
                assert not (pipeline.file_service.base_path / old_path).exists()
            if external_id not in ids.values():
                assert (
                    await pipeline.search_service.repository.get_entity_chunk_manifest(
                        before[external_id][0]
                    )
                    == []
                )
                assert (
                    await pipeline.search_service.repository.get_entity_physical_chunk_keys(
                        before[external_id][0]
                    )
                    == set()
                )
        if operation == "delete_target":
            source_id = before[old.manifest.note_ids["notes/Source.md"]][0]
            relation = await session.scalar(select(Relation).where(Relation.from_id == source_id))
            assert relation is not None and relation.to_id is None and relation.to_name == "Target"
    for external_id, entity in entities.items():
        rows = await pipeline.search_service.repository.get_entity_search_rows(entity.id)
        assert rows and all(row.file_path == entity.file_path for row in rows)
        assert all(row.file_path in live_paths for row in rows)
        manifest = await pipeline.search_service.repository.get_entity_chunk_manifest(entity.id)
        assert manifest and all(row.embedding_status == "ready" for row in manifest)
    if operation == "delete_all":
        # Upstream dropped the exact count; an unfiltered page that is empty says the same.
        assert await pipeline.search_service.repository.search(limit=1) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("delete", [False, True])
async def test_private_old_path_is_preserved_and_retirement_survives_restart(pipeline, delete):
    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    files, ids = dict(snapshot.files), dict(snapshot.manifest.note_ids)
    old_bytes = files.pop("notes/Source.md")
    note_id = ids.pop("notes/Source.md")
    if not delete:
        files["moved/Source.md"], ids["moved/Source.md"] = old_bytes, note_id
    publish(pipeline, files, ids)
    target = pipeline.file_service.base_path / "notes/Source.md"
    target.write_bytes(b"Private editor bytes")
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        async with asyncio.timeout(10):
            while not next(
                (row.last_error for row in (await worker.status()).stages if row.stage == "files"),
                None,
            ):
                await asyncio.sleep(0.02)
    assert target.read_bytes() == b"Private editor bytes"
    if not delete:
        # The editor's file blocks only the old path: the note's new file is written.
        assert (pipeline.file_service.base_path / "moved/Source.md").read_bytes() == (
            old_bytes.encode()
        )
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(GitFileRetirement.project_id)) == pipeline.project_id
    target.write_bytes(old_bytes.encode())
    await settle(pipeline)
    assert not target.exists()


async def child_worker(pipeline, app_config, engine_factory, tmp_path, stage):
    config = tmp_path / "child-config.json"
    config.write_text(app_config.model_dump_json())
    return await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "test-int/fixtures/git_projection_worker_crash.py",
            str(engine_factory[0].url.database),
            str(pipeline.repository.path),
            str(pipeline.project_id),
            str(config),
            stage,
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
        },
    )


CRASH_EXIT = {"retirement": 76, "after_writes": 77, "after_write_before_publish": 78}


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("BASIC_MEMORY_TEST_POSTGRES") == "1", reason="SQLite process proof")
@pytest.mark.parametrize("operation", ["move", "delete"])
@pytest.mark.parametrize("stage", ["content", "retirement", "files", "after_writes"])
async def test_layout_process_death_resumes_without_lost_old_file_permission(
    pipeline, app_config, engine_factory, tmp_path, operation, stage
):
    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    files, ids = dict(snapshot.files), dict(snapshot.manifest.note_ids)
    content, note_id = files.pop("notes/Source.md"), ids.pop("notes/Source.md")
    if operation == "move":
        files["moved/Source.md"], ids["moved/Source.md"] = content, note_id
    commit = publish(pipeline, files, ids)
    async with pipeline.session_maker() as session:
        entity_id = await session.scalar(
            select(Entity.id).where(Entity.external_id == str(note_id))
        )
    result = await child_worker(pipeline, app_config, engine_factory, tmp_path, stage)
    assert result.returncode == CRASH_EXIT.get(stage, 73), result.stderr
    root = pipeline.file_service.base_path
    if stage == "after_writes":
        # Died at the first old file to go, right after its note's new file was written
        # and published; the old file and its permission are still there.
        assert (root / "notes/Source.md").read_bytes() == content.encode()
        async with pipeline.session_maker() as session:
            assert await session.scalar(select(GitFileRetirement.file_path)) == "notes/Source.md"
            if operation == "move":
                assert (root / "moved/Source.md").read_bytes() == content.encode()
                row = await session.get(NoteContent, entity_id)
                assert row is not None and row.file_write_status == "synced"
    await settle(replace(pipeline, repository=GitProjectRepository(pipeline.repository.path)))
    assert pipeline.repository.head() == commit
    assert not (root / "notes/Source.md").exists()
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(GitFileRetirement.project_id)) is None
        if operation == "move":
            assert (
                await session.scalar(select(Entity.id).where(Entity.external_id == str(note_id)))
                == entity_id
            )


def reuse_layout(
    operation: str, files: dict[str, str], ids: dict[str, UUID]
) -> list[tuple[dict[str, str], dict[str, UUID]]]:
    """Commits that reuse a path another note still occupies on disk."""
    files, ids = dict(files), dict(ids)
    if operation == "swap":
        files["notes/Source.md"], files["notes/Target.md"] = (
            files["notes/Target.md"],
            files["notes/Source.md"],
        )
        ids["notes/Source.md"], ids["notes/Target.md"] = (
            ids["notes/Target.md"],
            ids["notes/Source.md"],
        )
        return [(files, ids)]
    files["moved/Source.md"] = files.pop("notes/Source.md")
    ids["moved/Source.md"] = ids.pop("notes/Source.md")
    commits = [(dict(files), dict(ids))]
    if operation == "reuse_path":
        files["notes/Source.md"] = (
            "---\ntitle: Replacement\ntype: note\npermalink: test-project/notes/replacement\n"
            "---\n\nReplacementprobe\n"
        )
        ids["notes/Source.md"] = uuid4()
    else:
        files["notes/Source.md"] = files.pop("notes/Target.md")
        ids["notes/Source.md"] = ids.pop("notes/Target.md")
    return [*commits, (files, ids)]


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("BASIC_MEMORY_TEST_POSTGRES") == "1", reason="SQLite process proof")
@pytest.mark.parametrize("operation", ["swap", "reuse_path", "move_onto_freed"])
async def test_death_between_a_reused_paths_delete_and_its_write_converges(
    pipeline, app_config, engine_factory, tmp_path, operation
):
    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    for files, ids in reuse_layout(operation, dict(snapshot.files), snapshot.manifest.note_ids):
        publish(pipeline, files, ids)
    result = await child_worker(pipeline, app_config, engine_factory, tmp_path, "reused_delete")
    assert result.returncode == 79, result.stderr
    root = pipeline.file_service.base_path
    if operation == "swap":
        # Died after the one delete that breaks the cycle: exactly one of the two paths
        # was cleared, and the other note's old file is still there.
        present = [
            path for path in ("notes/Source.md", "notes/Target.md") if (root / path).exists()
        ]
        assert len(present) == 1, present
        assert (root / present[0]).read_bytes() == snapshot.files[present[0]].encode()
    else:
        assert not (root / "notes/Source.md").exists()
    await settle(replace(pipeline, repository=GitProjectRepository(pipeline.repository.path)))
    head = pipeline.repository.snapshot()
    for path, text in head.files.items():
        assert (root / path).read_bytes() == text.encode()
    async with pipeline.session_maker() as session:
        # A free write at a retired path completes its row (C3).
        assert await session.scalar(select(GitFileRetirement.project_id)) is None
    # The note at the reused path takes a later edit at the same path as its own.
    files, ids = dict(head.files), dict(head.manifest.note_ids)
    files["notes/Source.md"] += "\nA later edit\n"
    publish(pipeline, files, ids)
    await settle(pipeline)
    assert (root / "notes/Source.md").read_bytes() == files["notes/Source.md"].encode()
    async with pipeline.session_maker() as session:
        statuses = set(await session.scalars(select(NoteContent.file_write_status)))
        assert statuses == {"synced"}


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("BASIC_MEMORY_TEST_POSTGRES") == "1", reason="SQLite process proof")
@pytest.mark.parametrize("then", ["edit", "edit_then_move"])
async def test_death_after_a_write_before_its_publish_is_not_an_editor_change(
    pipeline, app_config, engine_factory, tmp_path, then, monkeypatch
):
    """C7: the bytes left on disk were published by Git; later commits must converge."""
    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    files, ids = dict(snapshot.files), dict(snapshot.manifest.note_ids)
    first_edit = files["notes/Source.md"] + "\nFirst published edit\n"
    files["notes/Source.md"] = first_edit
    publish(pipeline, files, ids)
    result = await child_worker(
        pipeline, app_config, engine_factory, tmp_path, "after_write_before_publish"
    )
    assert result.returncode == 78, result.stderr
    root = pipeline.file_service.base_path
    assert (root / "notes/Source.md").read_bytes() == first_edit.encode()
    async with pipeline.session_maker() as session:
        row = await session.scalar(
            select(NoteContent).where(NoteContent.file_path == "notes/Source.md")
        )
        assert row is not None and row.file_write_status == "writing"
        assert row.file_checksum != row.db_checksum
    # A second edit reaches SQL before any files pass; the row keeps the old file checksum.
    files["notes/Source.md"] = first_edit + "Second published edit\n"
    publish(pipeline, files, ids)
    await project_git_content(
        pipeline.repository, project_id=pipeline.project_id, session_maker=pipeline.session_maker
    )
    if then == "edit_then_move":
        files["moved/Source.md"] = files.pop("notes/Source.md")
        ids["moved/Source.md"] = ids.pop("notes/Source.md")
        publish(pipeline, files, ids)
    events, statuses = watch_disk(monkeypatch, pipeline, snapshot, pipeline.repository.snapshot())
    await settle(replace(pipeline, repository=GitProjectRepository(pipeline.repository.path)))
    head = pipeline.repository.snapshot()
    for path, text in head.files.items():
        assert (root / path).read_bytes() == text.encode()
    if then == "edit_then_move":
        assert not (root / "notes/Source.md").exists()
    if then == "edit_then_move":
        # The recovery wrote the moved note before it removed anything: no gap.
        assert events and all(not event.absent for event in events), events
    else:
        # In place, the bytes Git proved are cleared and the note written right after.
        assert [(event.action, event.path, bool(event.absent)) for event in events] == [
            ("delete", "notes/Source.md", True),
            ("write", "notes/Source.md", False),
        ]
    async with pipeline.session_maker() as session:
        statuses.update(await session.scalars(select(NoteContent.file_write_status)))
        assert "external_change_detected" not in statuses
        assert await session.scalar(select(GitFileRetirement.project_id)) is None


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("BASIC_MEMORY_TEST_POSTGRES") == "1", reason="SQLite process proof")
async def test_second_process_owner_is_refused_while_first_keeps_running(
    pipeline, app_config, engine_factory, tmp_path
):
    initial(pipeline)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await ready(worker)
        result = await child_worker(pipeline, app_config, engine_factory, tmp_path, "content")
        assert result.returncode != 73 and "already running" in result.stderr
        assert (await worker.status()).caught_up


@pytest.mark.asyncio
async def test_unknown_checksum_cannot_authorize_deleting_an_editor_file(pipeline):
    from basic_memory.file_utils import compute_checksum

    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    files, ids = dict(snapshot.files), dict(snapshot.manifest.note_ids)
    files["moved/Source.md"] = files.pop("notes/Source.md")
    ids["moved/Source.md"] = ids.pop("notes/Source.md")
    publish(pipeline, files, ids)
    private = b"Private editor bytes"
    target = pipeline.file_service.base_path / "notes/Source.md"
    target.write_bytes(private)
    async with pipeline.session_maker() as session, session.begin():
        await session.execute(
            update(NoteContent)
            .where(NoteContent.file_path == "notes/Source.md")
            .values(db_checksum=await compute_checksum(private))
        )
    with pytest.raises(GitProjectionConflict, match="proven published ancestry"):
        await project_git_content(
            pipeline.repository,
            project_id=pipeline.project_id,
            session_maker=pipeline.session_maker,
        )
    assert target.read_bytes() == private
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(GitFileRetirement.project_id)) is None


@pytest.mark.asyncio
async def test_rapid_round_trips_preserve_old_permissions_and_current_file(pipeline):
    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    files, ids = dict(snapshot.files), dict(snapshot.manifest.note_ids)
    original = files["notes/Source.md"]
    layout = GitLayoutRepository(pipeline.project_id)
    old_permission = None
    path = "notes/Source.md"
    for step, destination in enumerate(
        ["moved/Source.md", "notes/Source.md", "moved/Source.md", "notes/Source.md"]
    ):
        files[destination] = original if step == 3 else files[path] + f"\nPublished step {step}\n"
        files.pop(path)
        ids[destination] = ids.pop(path)
        publish(pipeline, files, ids)
        result = await project_git_content(
            pipeline.repository,
            project_id=pipeline.project_id,
            session_maker=pipeline.session_maker,
        )
        assert result.notes_moved == 1
        async with pipeline.session_maker() as session:
            pending = await layout.retirements(session)
        if old_permission is None:
            old_permission = pending[0]
        path = destination
    assert old_permission is not None
    async with pipeline.session_maker() as session, session.begin():
        await layout.complete_retirement(session, old_permission)
    async with pipeline.session_maker() as session:
        pending = await layout.retirements(session)
        current = next(row for row in pending if row.file_path == "notes/Source.md")
        assert current.retirement_id != old_permission.retirement_id
        assert len(current.expected_checksums) >= 2
    await settle(pipeline)
    assert (pipeline.file_service.base_path / path).read_bytes() == original.encode()


@pytest.mark.asyncio
async def test_editor_change_during_old_file_cleanup_blocks_completion(pipeline, monkeypatch):
    from basic_memory.index.note_content_materialization import LocalNoteContentStorage

    initial(pipeline)
    await settle(pipeline)
    snapshot = pipeline.repository.snapshot()
    files, ids = dict(snapshot.files), dict(snapshot.manifest.note_ids)
    content = files.pop("notes/Source.md")
    ids.pop("notes/Source.md")
    publish(pipeline, files, ids)
    target = pipeline.file_service.base_path / "notes/Source.md"
    delete = LocalNoteContentStorage.delete_file_if_matches

    async def changed_before_delete(self, path, *, expected_checksum):
        target.write_bytes(b"New private editor bytes")
        return await delete(self, path, expected_checksum=expected_checksum)

    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", changed_before_delete)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        async with asyncio.timeout(10):
            while not next(
                (row.last_error for row in (await worker.status()).stages if row.stage == "files"),
                None,
            ):
                await asyncio.sleep(0.02)
    assert target.read_bytes() == b"New private editor bytes"
    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", delete)
    target.write_bytes(content.encode())
    await settle(pipeline)


@pytest.mark.asyncio
async def test_git_advance_after_layout_does_not_finish_the_stale_target(pipeline, monkeypatch):
    initial(pipeline)
    apply = GitLayoutRepository.apply
    advanced = False

    async def advance_after_layout(self, session, *, expected, target):
        nonlocal advanced
        result = await apply(self, session, expected=expected, target=target)
        if not advanced:
            advanced = True
            snapshot = pipeline.repository.snapshot()
            files = dict(snapshot.files)
            files["notes/Source.md"] += "\nNew published version\n"
            publish(pipeline, files, snapshot.manifest.note_ids)
        return result

    monkeypatch.setattr(GitLayoutRepository, "apply", advance_after_layout)
    with pytest.raises(GitProjectionConflict, match="advanced during projection"):
        await project_git_content(
            pipeline.repository,
            project_id=pipeline.project_id,
            session_maker=pipeline.session_maker,
        )
    await settle(pipeline)


@pytest.mark.asyncio
async def test_running_worker_refuses_changed_root_registration(pipeline):
    initial(pipeline)
    await settle(pipeline)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        try:
            async with pipeline.session_maker() as session, session.begin():
                await session.execute(
                    update(Project)
                    .where(Project.id == pipeline.project_id)
                    .values(path=str(pipeline.file_service.base_path / "unapproved-root"))
                )
            with pytest.raises(GitProjectionConflict, match="differs from SQL registration"):
                await worker.status()
            with pytest.raises(GitProjectionConflict, match="differs from SQL registration"):
                await pipeline.apply("files", pipeline.repository.head())
        finally:
            async with pipeline.session_maker() as session, session.begin():
                await session.execute(
                    update(Project)
                    .where(Project.id == pipeline.project_id)
                    .values(path=str(pipeline.file_service.base_path))
                )
        worker.notify()
        await ready(worker)
    assert not (pipeline.file_service.base_path / "unapproved-root").exists()
