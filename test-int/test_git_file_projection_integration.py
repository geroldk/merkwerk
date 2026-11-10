"""The Git files stage writes each new file before the old one goes (F2), through real BM.

Disk is a delayed projection of Git. These tests watch disk while the worker catches up:
a note never vanishes because it moves, folders a move empties go away, a newer commit
during a pass is not a failure, and bytes Basic Memory wrote are never mistaken for an
editor's.
"""

import asyncio
import hashlib
import os
from uuid import UUID, uuid4

import pytest
from loguru import logger
from sqlalchemy import select, update

from basic_memory.index import git_projection
from basic_memory.index.git_file_projection_plan import BaseFile, DiskView
from basic_memory.index.git_projection import GitProjectionPipeline, build_git_projection_pipeline
from basic_memory.index.git_projection_worker import GitProjectionWorker
from basic_memory.index.local_dependencies import LocalMarkdownFileIndexer
from basic_memory.index.note_content_materialization import LocalNoteContentStorage
from basic_memory.indexing.note_materialization_runner import (
    RepositoryNoteMaterializationPublisher,
)
from basic_memory.models import GitFileRetirement, GitProjectionProgress, NoteContent
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.runtime.note_content import (
    RuntimeNoteMaterializationResult,
    RuntimeNoteMaterializationStatus,
)
from basic_memory.services.git_project_projection import project_git_content


@pytest.fixture
def pipeline(tmp_path_factory, test_project, engine_factory, app_config):
    repository = GitProjectRepository.create(
        tmp_path_factory.mktemp("git-files") / "project.git",
        project_external_id=UUID(test_project.external_id),
    )
    return build_git_projection_pipeline(
        repository, project=test_project, session_maker=engine_factory[1], app_config=app_config
    )


def note(path: str, body: str = "Body") -> str:
    stem = path.removesuffix(".md")
    return (
        f"---\ntitle: {stem.rpartition('/')[2]}\ntype: note\n"
        f"permalink: test-project/{stem.lower()}\n---\n\n{body}\n"
    )


def publish(pipeline: GitProjectionPipeline, files: dict[str, str], ids: dict[str, UUID]) -> str:
    repository = pipeline.repository
    return repository.publish(
        repository.prepare(
            transaction_id=uuid4(), base_commit=repository.head(), files=files, note_ids=ids
        )
    ).commit_id


def seed(pipeline: GitProjectionPipeline, files: dict[str, str]) -> dict[str, UUID]:
    ids = {path: uuid4() for path in files}
    publish(pipeline, files, ids)
    return ids


async def ready(worker: GitProjectionWorker, timeout: float = 30) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not (await worker.status()).caught_up:
                await asyncio.sleep(0.02)
    except TimeoutError:
        stages = [(row.stage, row.last_error) for row in (await worker.status()).stages]
        raise AssertionError(f"worker did not catch up: {stages}") from None


async def settle(pipeline: GitProjectionPipeline, timeout: float = 30) -> None:
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await ready(worker, timeout)


async def files_error(worker: GitProjectionWorker) -> str:
    async with asyncio.timeout(15):
        while True:
            row = next(
                (row for row in (await worker.status()).stages if row.stage == "files"), None
            )
            if row is not None and row.last_error and row.last_error != "superseded":
                return row.last_error
            await asyncio.sleep(0.02)


async def statuses(pipeline: GitProjectionPipeline) -> set[str]:
    async with pipeline.session_maker() as session:
        return set(await session.scalars(select(NoteContent.file_write_status)))


async def retirement_rows(pipeline: GitProjectionPipeline) -> int:
    async with pipeline.session_maker() as session:
        return len(list(await session.scalars(select(GitFileRetirement.file_path))))


def on_disk(pipeline: GitProjectionPipeline, files: dict[str, str]) -> None:
    root = pipeline.file_service.base_path
    for path, text in files.items():
        assert (root / path).read_bytes() == text.encode(), path


def watch_events(monkeypatch, check) -> list[str]:
    """Call `check(action, path)` after every file write and guarded delete.

    The hooks run inside the worker, which turns an exception into a retried stage, so
    a check records what it finds and the test asserts on that afterwards.
    """
    events: list[str] = []
    write = LocalNoteContentStorage.write_file
    delete = LocalNoteContentStorage.delete_file_if_matches

    async def watched_write(self, path, content, *, metadata=None):
        checksum = await write(self, path, content, metadata=metadata)
        events.append(f"write {path}")
        check("write", str(path))
        return checksum

    async def watched_delete(self, path, *, expected_checksum):
        outcome = await delete(self, path, expected_checksum=expected_checksum)
        events.append(f"delete {path}")
        check("delete", str(path))
        return outcome

    monkeypatch.setattr(LocalNoteContentStorage, "write_file", watched_write)
    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", watched_delete)
    return events


# --- A newer commit during a pass: superseded, not failed ---


@pytest.mark.asyncio
async def test_new_commit_during_files_pass_continues(pipeline, monkeypatch):
    count = 20
    files = {f"notes/N{index:02}.md": note(f"notes/N{index:02}.md") for index in range(count)}
    ids = seed(pipeline, files)
    await settle(pipeline)
    for path in files:
        files[path] += "C1 edit\n"
    publish(pipeline, files, ids)

    materialize = git_projection.materialize_published_note
    calls = 0
    second: str | None = None

    async def counted(request, **kwargs):
        nonlocal calls, second
        calls += 1
        result = await materialize(request, **kwargs)
        if calls == 10:
            changed = dict(files)
            changed["notes/N15.md"] += "C2 edit\n"
            second = publish(pipeline, changed, ids)
        return result

    monkeypatch.setattr(git_projection, "materialize_published_note", counted)
    warnings: list[str] = []
    handler = logger.add(
        lambda message: warnings.append(
            f"{message.record['message']}: {message.record['extra'].get('error')}"
        ),
        level="WARNING",
    )
    worker = GitProjectionWorker(pipeline, poll_interval=2)
    loop = asyncio.get_running_loop()
    errors: list[str | None] = []
    moments: dict[str, float] = {}
    finish, begin = worker.progress.finish, worker.progress.begin

    async def recorded_finish(stage, commit_id, run_id, *, error=None):
        if stage == "files":
            errors.append(error)
            if error == "superseded":
                moments["superseded"] = loop.time()
        return await finish(stage, commit_id, run_id, error=error)

    async def recorded_begin(stage, commit_id, *, reusable_commit=None):
        if stage == "content" and commit_id == second:
            moments.setdefault("next", loop.time())
        return await begin(stage, commit_id, reusable_commit=reusable_commit)

    monkeypatch.setattr(worker.progress, "finish", recorded_finish)
    monkeypatch.setattr(worker.progress, "begin", recorded_begin)
    try:
        async with worker.run():
            await ready(worker, 60)
    finally:
        logger.remove(handler)
    # 20 writes for C1, one for C2's change (one more may be in flight): the pass for C1
    # ran to its end instead of starting over, and nothing was re-materialized.
    assert calls <= count + 1 + 1
    assert not any("pending after failure" in message for message in warnings)
    assert "superseded" in errors and set(errors) <= {None, "superseded"}
    # No backoff: the next pass started well within one poll interval.
    assert moments["next"] - moments["superseded"] < worker.poll_interval
    on_disk(pipeline, pipeline.repository.snapshot().files)


@pytest.mark.asyncio
async def test_a_commit_during_a_swap_never_ends_a_pass_with_a_note_missing(pipeline, monkeypatch):
    """C5: no HEAD check between a guarded delete and its write, or inside a cycle."""
    files = {"notes/a.md": note("notes/a.md"), "notes/b.md": note("notes/b.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    swapped = {"notes/a.md": files["notes/b.md"], "notes/b.md": files["notes/a.md"]}
    swapped_ids = {"notes/a.md": ids["notes/b.md"], "notes/b.md": ids["notes/a.md"]}
    publish(pipeline, swapped, swapped_ids)
    root = pipeline.file_service.base_path
    delete = LocalNoteContentStorage.delete_file_if_matches
    advanced = False

    async def advance_after_first_delete(self, path, *, expected_checksum):
        nonlocal advanced
        outcome = await delete(self, path, expected_checksum=expected_checksum)
        if not advanced:
            advanced = True
            publish(
                pipeline,
                {**swapped, "notes/c.md": note("notes/c.md")},
                {**swapped_ids, "notes/c.md": uuid4()},
            )
        return outcome

    monkeypatch.setattr(
        LocalNoteContentStorage, "delete_file_if_matches", advance_after_first_delete
    )
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    finish = worker.progress.finish
    at_finish: list[tuple[str | None, set[bytes]]] = []

    async def recorded_finish(stage, commit_id, run_id, *, error=None):
        if stage == "files":
            held = {(root / path).read_bytes() for path in swapped if (root / path).is_file()}
            at_finish.append((error, held))
        return await finish(stage, commit_id, run_id, error=error)

    monkeypatch.setattr(worker.progress, "finish", recorded_finish)
    async with worker.run():
        await ready(worker)
    assert advanced and at_finish[0][0] == "superseded"
    # Both notes were on disk at the end of every pass, also the superseded one.
    assert all(held == {text.encode() for text in files.values()} for _, held in at_finish)
    on_disk(pipeline, pipeline.repository.snapshot().files)


@pytest.mark.asyncio
async def test_a_concurrent_move_during_a_status_publish_never_removes_the_only_copy(
    pipeline, monkeypatch
):
    """C13 and C6: accept() moves a note while the stage publishes its status.

    The note moved away and back with only SQL following (KEEP), so the stage settles it
    without writing. Before that publish lands, the note moves again and a new note
    takes its path. The materializer's orphan cleanup would delete the only copy; on the
    Git host the retirement row owns that file instead.
    """
    files = {"notes/Source.md": note("notes/Source.md"), "notes/Target.md": note("notes/Target.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    source_bytes = files["notes/Source.md"].encode()
    away = {
        "moved/Source.md": files["notes/Source.md"],
        "notes/Target.md": files["notes/Target.md"],
    }
    away_ids = {
        "moved/Source.md": ids["notes/Source.md"],
        "notes/Target.md": ids["notes/Target.md"],
    }
    publish(pipeline, away, away_ids)
    await project_git_content(
        pipeline.repository, project_id=pipeline.project_id, session_maker=pipeline.session_maker
    )
    back = {**files, "notes/Target.md": files["notes/Target.md"] + "Second body\n"}
    publish(pipeline, back, ids)
    # Then the note moves away again, and a later commit gives its path to a new note.
    third = {**away, "notes/Target.md": back["notes/Target.md"]}
    # The moved note keeps its permalink, so the new note gets its own.
    fourth = {**third, "notes/Source.md": note("notes/Replacement.md", "A new note, old path")}
    fourth_ids = {**away_ids, "notes/Source.md": uuid4()}
    root = pipeline.file_service.base_path
    missing: list[tuple[str, str]] = []

    def source_on_disk(action: str, path: str) -> None:
        if not any(
            (root / candidate).is_file() and (root / candidate).read_bytes() == source_bytes
            for candidate in ("notes/Source.md", "moved/Source.md")
        ):
            missing.append((action, path))

    events = watch_events(monkeypatch, source_on_disk)
    publish_state = RepositoryNoteMaterializationPublisher.publish_written_file_state
    moved = False

    async def move_before_publish(self, request, prepared_write, written_file):
        nonlocal moved
        if not moved and prepared_write.file_path == "notes/Source.md":
            moved = True
            # What accept() does: publish, then commit SQL; the files stage is not gated.
            publish(pipeline, third, away_ids)
            await project_git_content(
                pipeline.repository,
                project_id=pipeline.project_id,
                session_maker=pipeline.session_maker,
            )
            publish(pipeline, fourth, fourth_ids)
        result = await publish_state(self, request, prepared_write, written_file)
        source_on_disk("publish", prepared_write.file_path)
        return result

    monkeypatch.setattr(
        RepositoryNoteMaterializationPublisher, "publish_written_file_state", move_before_publish
    )
    warnings: list[str] = []
    handler = logger.add(
        lambda message: warnings.append(
            f"{message.record['message']}: {message.record['extra'].get('error')}"
        ),
        level="WARNING",
    )
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    finish = worker.progress.finish
    target_at_superseded: list[bytes] = []

    async def recorded_finish(stage, commit_id, run_id, *, error=None):
        if stage == "files" and error == "superseded":
            target_at_superseded.append((root / "notes/Target.md").read_bytes())
        return await finish(stage, commit_id, run_id, error=error)

    monkeypatch.setattr(worker.progress, "finish", recorded_finish)
    try:
        async with worker.run():
            await ready(worker)
    finally:
        logger.remove(handler)
    assert not any("pending after failure" in message for message in warnings), warnings
    assert moved and events
    assert missing == []
    # The other note's write in the superseded pass was not abandoned (C6).
    assert target_at_superseded == [back["notes/Target.md"].encode()]
    on_disk(pipeline, fourth)
    assert await retirement_rows(pipeline) == 0
    assert await statuses(pipeline) == {"synced"}


# --- A note moved back with the same bytes (C2) ---


@pytest.mark.asyncio
async def test_a_note_moved_back_unchanged_is_settled_and_takes_a_later_edit(pipeline, monkeypatch):
    files = {"notes/Source.md": note("notes/Source.md"), "notes/Target.md": note("notes/Target.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    away_ids = {
        "moved/Source.md": ids["notes/Source.md"],
        "notes/Target.md": ids["notes/Target.md"],
    }
    publish(
        pipeline,
        {"moved/Source.md": files["notes/Source.md"], "notes/Target.md": files["notes/Target.md"]},
        away_ids,
    )
    await project_git_content(
        pipeline.repository, project_id=pipeline.project_id, session_maker=pipeline.session_maker
    )
    publish(pipeline, files, ids)
    await settle(pipeline)
    async with pipeline.session_maker() as session:
        row = await session.scalar(
            select(NoteContent).where(NoteContent.file_path == "notes/Source.md")
        )
        # The KEEP pass published the row: a later edit finds its own file.
        assert row is not None and row.file_write_status == "synced"
        assert row.file_checksum is not None and row.file_checksum == row.db_checksum
    assert await retirement_rows(pipeline) == 0

    def no_proof(*args, **kwargs):
        raise AssertionError("an edit of a settled note needs no Git history proof")

    monkeypatch.setattr(GitProjectRepository, "contains_published_note_content", no_proof)
    files["notes/Source.md"] += "Later edit\n"
    publish(pipeline, files, ids)
    await settle(pipeline)
    on_disk(pipeline, files)
    assert await statuses(pipeline) == {"synced"}


# --- An editor's file at one end of a swap (C4) ---


@pytest.mark.asyncio
async def test_a_swap_with_an_editor_file_at_one_end_deletes_nothing(pipeline):
    files = {"notes/a.md": note("notes/a.md"), "notes/b.md": note("notes/b.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    publish(
        pipeline,
        {"notes/a.md": files["notes/b.md"], "notes/b.md": files["notes/a.md"]},
        {"notes/a.md": ids["notes/b.md"], "notes/b.md": ids["notes/a.md"]},
    )
    root = pipeline.file_service.base_path
    # An editor replaced a's old file: b's new path. b's only copy is its old file.
    (root / "notes/a.md").write_bytes(b"Private editor bytes")
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        error = await files_error(worker)
    assert "unpublished editor change" in error
    assert (root / "notes/a.md").read_bytes() == b"Private editor bytes"
    assert (root / "notes/b.md").read_bytes() == files["notes/b.md"].encode()
    assert await retirement_rows(pipeline) == 2


# --- Case- and Unicode-only renames (C1), simulated with a hard link ---


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_a_case_only_rename_through_an_alias_keeps_the_note(pipeline, changed):
    files = {"notes/Source.md": note("notes/Source.md"), "notes/Target.md": note("notes/Target.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    renamed = {
        "notes/source.md": files["notes/Source.md"]
        + ("Changed with the rename\n" if changed else ""),
        "notes/Target.md": files["notes/Target.md"],
    }
    publish(
        pipeline,
        renamed,
        {"notes/source.md": ids["notes/Source.md"], "notes/Target.md": ids["notes/Target.md"]},
    )
    root = pipeline.file_service.base_path
    # On macOS or Windows the new spelling reaches the old file; a hard link does on Linux.
    os.link(root / "notes/Source.md", root / "notes/source.md")
    await settle(pipeline)
    on_disk(pipeline, renamed)
    assert await retirement_rows(pipeline) == 0
    assert await statuses(pipeline) == {"synced"}
    old = root / "notes/Source.md"
    if changed:
        # The new bytes went to a new file; the old one held only the old bytes.
        assert not old.exists()
    else:
        # The old spelling is the live file itself: renamed in place, never deleted.
        assert not old.exists() or old.samefile(root / "notes/source.md")


# --- Files and folders trading places (C8) ---


@pytest.mark.asyncio
@pytest.mark.parametrize("same_note", [False, True])
async def test_a_file_becomes_a_folder_and_back(pipeline, same_note):
    files = {"a.md": note("a.md"), "other.md": note("other.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    root = pipeline.file_service.base_path
    inside = note("a.md/b.md", "Inside")
    if same_note:
        folder = {"a.md/b.md": files["a.md"], "other.md": files["other.md"]}
        folder_ids = {"a.md/b.md": ids["a.md"], "other.md": ids["other.md"]}
    else:
        folder = {"a.md/b.md": inside, "other.md": files["other.md"]}
        folder_ids = {"a.md/b.md": uuid4(), "other.md": ids["other.md"]}
    publish(pipeline, folder, folder_ids)
    await settle(pipeline)
    assert (root / "a.md").is_dir()
    on_disk(pipeline, folder)

    if same_note:
        back, back_ids = files, ids
    else:
        back = {"a.md": note("a.md", "A file again"), "other.md": files["other.md"]}
        back_ids = {"a.md": uuid4(), "other.md": ids["other.md"]}
    publish(pipeline, back, back_ids)
    await settle(pipeline)
    assert (root / "a.md").is_file()
    on_disk(pipeline, back)
    assert await retirement_rows(pipeline) == 0
    assert await statuses(pipeline) == {"synced"}


@pytest.mark.asyncio
async def test_a_folder_that_holds_a_foreign_file_blocks_the_path_and_keeps_it(pipeline):
    files = {"a.md/b.md": note("a.md/b.md"), "other.md": note("other.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    root = pipeline.file_service.base_path
    (root / "a.md/picture.png").write_bytes(b"png")
    publish(
        pipeline,
        {"a.md": note("a.md"), "other.md": files["other.md"]},
        {
            "a.md": uuid4(),
            "other.md": ids["other.md"],
        },
    )
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        error = await files_error(worker)
    assert "does not own blocks the published path" in error
    assert (root / "a.md/picture.png").read_bytes() == b"png"


# --- A large folder move ---


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_folder_move_of_over_a_hundred_notes_has_no_gap_and_leaves_no_empty_folders(
    pipeline, monkeypatch
):
    count = 110
    files = {
        f"old/{'x' if index % 2 else 'y'}/N{index:03}.md": note(
            f"old/{'x' if index % 2 else 'y'}/N{index:03}.md", f"Body {index}"
        )
        for index in range(count)
    }
    ids = seed(pipeline, files)
    await settle(pipeline, 240)
    moved = {"new/" + path.removeprefix("old/"): text for path, text in files.items()}
    moved_ids = {"new/" + path.removeprefix("old/"): note_id for path, note_id in ids.items()}
    publish(pipeline, moved, moved_ids)
    root = pipeline.file_service.base_path
    pairs = [
        (path, "new/" + path.removeprefix("old/"), text.encode()) for path, text in files.items()
    ]
    worst_double = 0
    missing: list[tuple[str, str, str]] = []

    def every_note_on_disk(action: str, path: str) -> None:
        nonlocal worst_double
        doubled = 0
        for old_path, new_path, data in pairs:
            held = sum(
                (root / candidate).is_file() and (root / candidate).read_bytes() == data
                for candidate in (old_path, new_path)
            )
            if not held:
                missing.append((action, path, old_path))
            doubled += held == 2
        worst_double = max(worst_double, doubled)

    events = watch_events(monkeypatch, every_note_on_disk)
    indexed: list[str] = []
    index = LocalMarkdownFileIndexer.index_file

    async def recorded_index(self, file_path, *, source):
        indexed.append(str(file_path))
        return await index(self, file_path, source=source)

    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", recorded_index)
    await settle(pipeline, 240)
    assert missing == []
    assert len(events) == 2 * count
    # Each old file goes right after its new one: one note at a time is on disk twice.
    assert worst_double <= 1
    assert not (root / "old").exists()
    on_disk(pipeline, moved)
    assert indexed and not any(path.startswith("old/") for path in indexed)
    assert await retirement_rows(pipeline) == 0


# --- What the stage reports instead of guessing ---


@pytest.mark.asyncio
async def test_an_editor_change_at_a_reused_path_during_cleanup_keeps_the_editors_file(
    pipeline, monkeypatch
):
    files = {"notes/Source.md": note("notes/Source.md"), "notes/Target.md": note("notes/Target.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    publish(
        pipeline,
        {"moved/Source.md": files["notes/Source.md"], "notes/Target.md": files["notes/Target.md"]},
        {"moved/Source.md": ids["notes/Source.md"], "notes/Target.md": ids["notes/Target.md"]},
    )
    publish(
        pipeline,
        {
            "moved/Source.md": files["notes/Source.md"],
            "notes/Target.md": files["notes/Target.md"],
            "notes/Source.md": note("notes/Replacement.md"),
        },
        {
            "moved/Source.md": ids["notes/Source.md"],
            "notes/Target.md": ids["notes/Target.md"],
            "notes/Source.md": uuid4(),
        },
    )
    root = pipeline.file_service.base_path
    delete = LocalNoteContentStorage.delete_file_if_matches

    async def editor_first(self, path, *, expected_checksum):
        if path == "notes/Source.md":
            (root / path).write_bytes(b"Saved by an editor just now")
        return await delete(self, path, expected_checksum=expected_checksum)

    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", editor_first)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        error = await files_error(worker)
    assert "changed during guarded Git cleanup" in error
    assert (root / "notes/Source.md").read_bytes() == b"Saved by an editor just now"
    assert (root / "moved/Source.md").read_bytes() == files["notes/Source.md"].encode()


@pytest.mark.asyncio
async def test_a_hidden_folder_left_inside_blocks_a_file_where_the_folder_was(pipeline):
    files = {"a.md/b.md": note("a.md/b.md"), "other.md": note("other.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    root = pipeline.file_service.base_path
    (root / "a.md/.cache").mkdir()
    publish(
        pipeline,
        {"a.md": note("a.md"), "other.md": files["other.md"]},
        {"a.md": uuid4(), "other.md": ids["other.md"]},
    )
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        error = await files_error(worker)
    # The old note went; the hidden folder stays, so the new file cannot take the path.
    assert "does not own blocks the published path" in error
    assert not (root / "a.md/b.md").exists() and (root / "a.md/.cache").is_dir()


@pytest.mark.asyncio
async def test_a_symlink_inside_a_folder_where_a_file_must_be_blocks_the_path(pipeline):
    files = {"a.md/b.md": note("a.md/b.md"), "other.md": note("other.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    root = pipeline.file_service.base_path
    (root / "a.md/link").symlink_to(root / "other.md")
    publish(
        pipeline,
        {"a.md": note("a.md"), "other.md": files["other.md"]},
        {"a.md": uuid4(), "other.md": ids["other.md"]},
    )
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        error = await files_error(worker)
    assert "does not own blocks the published path" in error
    # The deleted note's file went; the symlink stays, so the new file cannot take the path.
    assert (root / "a.md/link").is_symlink() and not (root / "a.md/b.md").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["conflict", "wrong_bytes"])
async def test_a_write_the_materializer_did_not_finish_is_reported(pipeline, monkeypatch, outcome):
    files = {"notes/a.md": note("notes/a.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    publish(pipeline, {"moved/a.md": files["notes/a.md"]}, {"moved/a.md": ids["notes/a.md"]})
    root = pipeline.file_service.base_path

    async def unfinished(request, **kwargs):
        if outcome == "conflict":
            # An editor saved the file between the stage's look and the write.
            return RuntimeNoteMaterializationResult(
                entity_id=request.entity_id,
                status=RuntimeNoteMaterializationStatus.conflict,
                reason="external change",
            )
        (root / "moved").mkdir(exist_ok=True)
        (root / "moved/a.md").write_bytes(b"Something else")
        return RuntimeNoteMaterializationResult(
            entity_id=request.entity_id,
            status=RuntimeNoteMaterializationStatus.written,
            reason="written",
        )

    monkeypatch.setattr(git_projection, "materialize_published_note", unfinished)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        error = await files_error(worker)
    expected = (
        "File materialization: conflict" if outcome == "conflict" else "differs from published"
    )
    assert expected in error
    # The new copy is not verified, so the old one stays.
    assert (root / "notes/a.md").read_bytes() == files["notes/a.md"].encode()
    assert await retirement_rows(pipeline) == 1


@pytest.mark.asyncio
async def test_bytes_that_are_not_utf8_or_changed_since_the_look_are_never_proven(pipeline):
    files = {"notes/a.md": note("notes/a.md")}
    commit = publish(pipeline, files, {"notes/a.md": uuid4()})
    root = pipeline.file_service.base_path
    (root / "notes").mkdir()
    (root / "notes/a.md").write_bytes(b"\xff\xfe not text")
    (root / "notes/b.md").write_bytes(files["notes/a.md"].encode())
    binary = hashlib.sha256(b"\xff\xfe not text").hexdigest()
    disk = DiskView(files={"notes/a.md": binary, "notes/b.md": "seen before a later change"})
    assert (
        await pipeline._published_bytes(commit, ["notes/a.md", "notes/b.md"], disk) == frozenset()
    )


@pytest.mark.asyncio
async def test_the_last_written_commit_is_used_only_inside_the_targets_history(pipeline):
    files = {"notes/a.md": note("notes/a.md")}
    ids = seed(pipeline, files)
    snapshot = pipeline.repository.snapshot()
    # Nothing written yet: no basis.
    assert await pipeline._base_files(snapshot) is None
    await settle(pipeline)
    assert await pipeline._base_files(snapshot) == {
        "notes/a.md": BaseFile(
            hashlib.sha256(files["notes/a.md"].encode()).hexdigest(), "notes/a.md"
        )
    }
    # A commit outside the target's history (a private candidate) is never trusted.
    candidate = pipeline.repository.prepare(
        transaction_id=uuid4(),
        base_commit=snapshot.commit_id,
        files={**files, "notes/b.md": note("notes/b.md")},
        note_ids={**ids, "notes/b.md": uuid4()},
    )
    async with pipeline.session_maker() as session, session.begin():
        await session.execute(
            update(GitProjectionProgress)
            .where(GitProjectionProgress.stage == "files")
            .values(completed_commit=candidate.commit_id)
        )
    assert await pipeline._base_files(snapshot) is None


@pytest.mark.asyncio
async def test_the_final_check_covers_notes_that_needed_nothing(pipeline, monkeypatch):
    """C12: a delete after classification must never take a live file away unnoticed."""
    files = {f"notes/N{index}.md": note(f"notes/N{index}.md") for index in range(3)}
    ids = seed(pipeline, files)
    await settle(pipeline)
    renamed = {
        ("zz/N0.md" if path == "notes/N0.md" else path): text for path, text in files.items()
    }
    renamed_ids = {("zz/N0.md" if path == "notes/N0.md" else path): i for path, i in ids.items()}
    publish(pipeline, renamed, renamed_ids)
    root = pipeline.file_service.base_path
    delete = LocalNoteContentStorage.delete_file_if_matches

    async def damage_a_finished_note(self, path, *, expected_checksum):
        outcome = await delete(self, path, expected_checksum=expected_checksum)
        # As if a classifier bug had touched a note that needed nothing.
        (root / "notes/N1.md").write_bytes(b"Damaged after classification")
        return outcome

    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", damage_a_finished_note)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    finish = worker.progress.finish
    errors: list[str | None] = []

    async def recorded_finish(stage, commit_id, run_id, *, error=None):
        if stage == "files":
            errors.append(error)
        return await finish(stage, commit_id, run_id, error=error)

    monkeypatch.setattr(worker.progress, "finish", recorded_finish)
    async with worker.run():
        await files_error(worker)
    # The very pass that damaged it noticed it, in its final check of every file.
    assert errors[0] is not None and "differs from published Git bytes" in errors[0]
    monkeypatch.setattr(LocalNoteContentStorage, "delete_file_if_matches", delete)
    (root / "notes/N1.md").write_bytes(files["notes/N1.md"].encode())
    await settle(pipeline)
    on_disk(pipeline, renamed)


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_old_bytes_of_an_unknown_owner_go_only_after_a_clean_pass(pipeline, conflict):
    """C4: with no basis to name the owner, an old file waits for every other note."""
    files = {"notes/a.md": note("notes/a.md"), "notes/b.md": note("notes/b.md")}
    ids = seed(pipeline, files)
    await settle(pipeline)
    # Moved with an edit, and no record of the last written commit: nothing names the
    # note the old bytes belong to.
    moved = {
        "moved/a.md": files["notes/a.md"] + "Edit\n",
        "moved/b.md": files["notes/b.md"] + "Edit\n",
    }
    moved_ids = {"moved/a.md": ids["notes/a.md"], "moved/b.md": ids["notes/b.md"]}
    if conflict:
        moved["notes/c.md"] = note("notes/c.md")
        moved_ids["notes/c.md"] = uuid4()
    publish(pipeline, moved, moved_ids)
    async with pipeline.session_maker() as session, session.begin():
        await session.execute(
            update(GitProjectionProgress)
            .where(GitProjectionProgress.stage == "files")
            .values(completed_commit=None)
        )
    root = pipeline.file_service.base_path
    if conflict:
        (root / "notes/c.md").write_bytes(b"An editor got there first")
        worker = GitProjectionWorker(pipeline, poll_interval=0.02)
        async with worker.run():
            error = await files_error(worker)
        assert "differs from published Git bytes" in error
        # Both notes are written, but their old files wait for a clean pass.
        on_disk(pipeline, {path: moved[path] for path in ("moved/a.md", "moved/b.md")})
        on_disk(pipeline, files)
        assert await retirement_rows(pipeline) == 2
        return
    await settle(pipeline)
    on_disk(pipeline, moved)
    assert not (root / "notes").exists()
    assert await retirement_rows(pipeline) == 0
