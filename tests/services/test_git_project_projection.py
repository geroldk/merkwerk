"""Real SQL/Git tests for identity restoration and conservative adoption."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from basic_memory.index.git_projection import build_git_projection_pipeline
from basic_memory.index.note_content_materialization import recover_stuck_materializations
from basic_memory.models import Entity, NoteContent, Project
from basic_memory.read_cache import ReadCacheChange, ReadCacheInvalidationStatus
from basic_memory.repository.entity_repository import AcceptedPendingEntityWrite
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_projection_repository import (
    GitProjectionConflict,
    GitProjectionRepository,
    GitProjectionSuperseded,
    register_git_project,
)
from basic_memory.repository.note_content_repository import NoteContentVersionConflict
from basic_memory.schemas.git_project import GitProjectManifest
from basic_memory.services.file_service import FileService
from basic_memory.services.git_project_projection import project_git_content

from tests.git_host_support import local_time_with_offset as local_time_with_offset


@pytest.fixture
def repository(
    test_project: Project, tmp_path_factory: pytest.TempPathFactory
) -> GitProjectRepository:
    return GitProjectRepository.create(
        tmp_path_factory.mktemp("git-projection") / "project.git",
        project_external_id=UUID(test_project.external_id),
    )


@pytest.fixture
def markdown(test_project: Project) -> str:
    return f"---\ntitle: A\ntype: note\npermalink: {test_project.name}/notes/a\n---\n\nOriginal\n"


def publish(
    repository: GitProjectRepository, content: str, note_id: UUID, path: str = "notes/a.md"
) -> str:
    return repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=repository.head(),
            files={path: content},
            note_ids={path: note_id},
        )
    ).commit_id


def local_entity(project_id: int, external_id: UUID, path: str = "notes/a.md") -> Entity:
    return Entity(
        project_id=project_id,
        external_id=str(external_id),
        title="A",
        note_type="note",
        content_type="text/markdown",
        file_path=path,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_entity", [False, True])
async def test_retry_keeps_uuid_and_version_and_restores_a_missing_file(
    repository, markdown, test_project, session_maker, app_config, seed_entity
):
    note_id = uuid4()
    publish(repository, markdown, note_id)
    if seed_entity:
        async with session_maker() as session, session.begin():
            session.add(local_entity(test_project.id, note_id))
    first = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker
    )
    assert first.notes_changed == 1
    storage = FileService(Path(test_project.path))
    assert (
        await recover_stuck_materializations(
            session_maker=session_maker, file_service=storage, project_id=test_project.id
        )
    ).written == 1
    target = Path(test_project.path) / "notes/a.md"
    assert target.read_bytes() == markdown.encode()
    target.unlink()
    retry = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker
    )
    assert retry.notes_changed == 0
    # An unchanged note stays as it is in SQL (F2): no pending flip at every commit.
    # The Git files stage checks every file on disk itself and writes the lost one.
    async with session_maker() as session:
        content = await session.scalar(
            select(NoteContent).where(NoteContent.project_id == test_project.id)
        )
        assert content is not None and content.file_write_status == "synced"
    pipeline = build_git_projection_pipeline(
        repository, project=test_project, session_maker=session_maker, app_config=app_config
    )
    await pipeline.apply("files", repository.head())
    assert target.read_bytes() == markdown.encode()
    async with session_maker() as session:
        entity = await session.scalar(select(Entity).where(Entity.external_id == str(note_id)))
        assert entity is not None
        content = await session.get(NoteContent, entity.id)
        assert content is not None
        assert content.external_id == entity.external_id == str(note_id)
        assert content.db_version == 1


@pytest.mark.asyncio
async def test_published_update_advances_sql_but_older_target_cannot_revert_it(
    repository, markdown, test_project, session_maker
):
    note_id = uuid4()
    old_commit = publish(repository, markdown, note_id)
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    storage = FileService(Path(test_project.path))
    await recover_stuck_materializations(
        session_maker=session_maker, file_service=storage, project_id=test_project.id
    )
    updated = markdown + "\nNew published content\n"
    publish(repository, updated, note_id)
    result = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker
    )
    assert result.notes_changed == 1
    await recover_stuck_materializations(
        session_maker=session_maker, file_service=storage, project_id=test_project.id
    )
    with pytest.raises(GitProjectionConflict, match="current published Git head"):
        await project_git_content(
            repository,
            project_id=test_project.id,
            session_maker=session_maker,
            commit_id=old_commit,
        )
    async with session_maker() as session:
        content = await session.scalar(
            select(NoteContent).where(NoteContent.project_id == test_project.id)
        )
        assert content is not None
        assert content.markdown_content == updated
        assert content.db_version == 2
    assert (Path(test_project.path) / "notes/a.md").read_bytes() == updated.encode()


@pytest.mark.asyncio
async def test_unexpected_editor_file_is_not_overwritten(
    repository, markdown, test_project, session_maker
):
    publish(repository, markdown, uuid4())
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    target = Path(test_project.path) / "notes/a.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"Unpublished editor draft")
    result = await recover_stuck_materializations(
        session_maker=session_maker,
        file_service=FileService(Path(test_project.path)),
        project_id=test_project.id,
    )
    assert result.written == 0
    assert target.read_bytes() == b"Unpublished editor draft"
    async with session_maker() as session:
        content = await session.scalar(
            select(NoteContent).where(NoteContent.project_id == test_project.id)
        )
        assert content is not None
        assert content.markdown_content == markdown
        assert content.file_write_status == "external_change_detected"


@pytest.mark.asyncio
async def test_wrong_or_missing_project_is_rejected(
    repository, markdown, test_project, session_maker
):
    publish(repository, markdown, uuid4())
    with pytest.raises(GitProjectionConflict, match="project UUID"):
        await project_git_content(
            repository, project_id=test_project.id + 999, session_maker=session_maker
        )
    foreign = GitProjectRepository.create(repository.path.parent / "foreign.git")
    with pytest.raises(GitProjectionConflict, match="project UUID"):
        await project_git_content(foreign, project_id=test_project.id, session_maker=session_maker)


@pytest.mark.asyncio
async def test_existing_uuid_does_not_move_between_projects(
    repository, markdown, test_project, session_maker
):
    note_id = uuid4()
    publish(repository, markdown, note_id)
    async with session_maker() as session, session.begin():
        other = Project(name="other", path=str(repository.path.parent / "other"), is_active=True)
        session.add(other)
        await session.flush()
        session.add(local_entity(other.id, note_id))
    with pytest.raises(GitProjectionConflict, match="another project or path"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )


@pytest.mark.asyncio
async def test_existing_path_cannot_acquire_another_uuid(
    repository, markdown, test_project, session_maker
):
    desired, occupied = uuid4(), uuid4()
    repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=repository.head(),
            files={"notes/a.md": markdown, "notes/b.md": markdown},
            note_ids={"notes/a.md": desired, "notes/b.md": occupied},
        )
    )
    async with session_maker() as session, session.begin():
        session.add(local_entity(test_project.id, occupied))
    with pytest.raises(GitProjectionConflict, match="path is already owned"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )


@pytest.mark.asyncio
async def test_unrelated_local_notes_prevent_snapshot_adoption(
    repository, markdown, test_project, session_maker
):
    publish(repository, markdown, uuid4())
    async with session_maker() as session, session.begin():
        session.add(local_entity(test_project.id, uuid4(), "unrelated.md"))
    with pytest.raises(GitProjectionConflict, match="outside this snapshot"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )


@pytest.mark.asyncio
async def test_inconsistent_note_content_identity_is_rejected(
    repository, markdown, test_project, session_maker
):
    publish(repository, markdown, uuid4())
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    async with session_maker() as session, session.begin():
        await session.execute(
            update(NoteContent)
            .where(NoteContent.project_id == test_project.id)
            .values(external_id=str(uuid4()))
        )
    with pytest.raises(GitProjectionConflict, match="content identity"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["inside", "contains"])
async def test_git_control_files_must_be_outside_the_project_root(
    repository, markdown, test_project, session_maker, location
):
    publish(repository, markdown, uuid4())
    root = repository.path / "notes" if location == "inside" else repository.path.parent
    async with session_maker() as session, session.begin():
        project = await session.get(Project, test_project.id)
        assert project is not None
        project.path = str(root)
    with pytest.raises(GitProjectionConflict, match="outside the indexed"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Plain text",
        "---\npermalink: notes/a\n---\n# A",
        "---\ntitle: A\n---\nBody",
        "---\ntitle: A\npermalink: notes/a\n---\nBody",
        "---\n[bad: yaml\n---\nBody",
    ],
)
async def test_noncanonical_import_needs_preparation_before_publication(
    repository, test_project, session_maker, content
):
    publish(repository, content, uuid4())
    with pytest.raises(GitProjectionConflict, match="canonical BM"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )
    async with session_maker() as session:
        assert (
            await session.scalar(select(Entity.id).where(Entity.project_id == test_project.id))
            is None
        )


@pytest.mark.asyncio
async def test_project_registration_preserves_uuid_and_rejects_collisions(
    repository, test_project, session_maker
):
    manifest = repository.snapshot().manifest
    async with session_maker() as session, session.begin():
        existing = await register_git_project(
            session, manifest, name=test_project.name, path=Path(test_project.path)
        )
        assert existing.id == test_project.id
        with pytest.raises(GitProjectionConflict, match="already bound"):
            await register_git_project(
                session, manifest, name="renamed", path=Path(test_project.path)
            )
        with pytest.raises(GitProjectionConflict, match="already owned"):
            await register_git_project(
                session,
                GitProjectManifest(project_external_id=uuid4()),
                name=test_project.name,
                path=Path(test_project.path),
            )


@pytest.mark.asyncio
async def test_sql_revision_is_compared_again_before_projection(
    repository, markdown, test_project, session_maker, monkeypatch
):
    note_id = uuid4()
    publish(repository, markdown, note_id)
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    projection = GitProjectionRepository(test_project.id)
    observed = repository.committed_at(repository.head())
    fields = AcceptedPendingEntityWrite(
        title="A",
        note_type="note",
        entity_metadata=None,
        content_type="text/markdown",
        permalink="notes/a",
        file_path="notes/a.md",
        created_at=observed,
        updated_at=observed,
        created_by=None,
        last_updated_by=None,
        external_id=str(note_id),
    )
    async with session_maker() as session, session.begin():
        basis = await projection.read_basis(session, note_id, "notes/a.md")
        assert basis is not None
        with pytest.raises(GitProjectionConflict, match="Prepared fields"):
            await projection.restore_note(
                session,
                external_id=note_id,
                fields=replace(fields, external_id=str(uuid4())),
                markdown_content=markdown,
                checksum=basis.checksum,
                expected=basis,
            )
        with pytest.raises(NoteContentVersionConflict, match="advanced"):
            await projection.restore_note(
                session,
                external_id=note_id,
                fields=fields,
                markdown_content=markdown,
                checksum=basis.checksum,
                expected=None,
            )
        # An unchanged note is not touched: no state field update, no new revision.
        monkeypatch.setattr(
            projection.contents,
            "update_state_fields",
            AsyncMock(side_effect=AssertionError("unchanged note must keep its file state")),
        )
        assert not await projection.restore_note(
            session,
            external_id=note_id,
            fields=fields,
            markdown_content=markdown,
            checksum=basis.checksum,
            expected=basis,
        )


class RecordingCache:
    def __init__(self) -> None:
        self.projects: list[str] = []

    async def invalidate(
        self, project_id: str, change: ReadCacheChange
    ) -> ReadCacheInvalidationStatus:
        self.projects.append(project_id)
        return ReadCacheInvalidationStatus.invalidated

    async def invalidate_project(self, project_id: str) -> ReadCacheInvalidationStatus:
        self.projects.append(project_id)
        return ReadCacheInvalidationStatus.invalidated


@pytest.mark.asyncio
async def test_projection_retires_old_cached_project_reads(
    repository, markdown, test_project, session_maker
):
    publish(repository, markdown, uuid4())
    cache = RecordingCache()
    await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker, read_cache=cache
    )
    assert cache.projects == [test_project.external_id]


@pytest.mark.asyncio
async def test_sql_bytes_from_a_private_candidate_cannot_be_adopted(
    repository, markdown, test_project, session_maker
):
    note_id = uuid4()
    publish(repository, markdown, note_id)
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    private = markdown + "Private append"
    repository.prepare(
        transaction_id=uuid4(),
        base_commit=repository.head(),
        files={"notes/a.md": private},
        note_ids={"notes/a.md": note_id},
    )
    async with session_maker() as session, session.begin():
        await session.execute(
            update(NoteContent)
            .where(NoteContent.project_id == test_project.id)
            .values(markdown_content=private)
        )
    with pytest.raises(GitProjectionConflict, match="outside the target"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )


@pytest.mark.asyncio
async def test_published_head_change_during_preparation_stops_projection(
    repository, markdown, test_project, session_maker, monkeypatch
):
    note_id = uuid4()
    publish(repository, markdown, note_id)
    from basic_memory.services import git_project_projection as service

    original_checksum = service.compute_checksum

    async def publish_before_write(content: str) -> str:
        checksum = await original_checksum(content)
        publish(repository, markdown + "Next published version", note_id)
        return checksum

    monkeypatch.setattr(service, "compute_checksum", publish_before_write)
    # Superseded, not a failure: the worker starts the next pass at once (F2).
    with pytest.raises(GitProjectionSuperseded, match="advanced during projection"):
        await project_git_content(
            repository, project_id=test_project.id, session_maker=session_maker
        )
    async with session_maker() as session:
        assert (
            await session.scalar(select(Entity.id).where(Entity.project_id == test_project.id))
            is None
        )


@pytest.mark.asyncio
async def test_content_appearing_after_an_absence_read_is_not_overwritten(
    repository, markdown, test_project, session_maker, monkeypatch
):
    note_id = uuid4()
    publish(repository, markdown, note_id)
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    projection = GitProjectionRepository(test_project.id)
    observed = repository.committed_at(repository.head())
    fields = AcceptedPendingEntityWrite(
        title="A",
        note_type="note",
        entity_metadata=None,
        content_type="text/markdown",
        permalink="notes/a",
        file_path="notes/a.md",
        created_at=observed,
        updated_at=observed,
        created_by=None,
        last_updated_by=None,
        external_id=str(note_id),
    )
    # Emulate the earlier absence read; the row is already visible by the time
    # persistence runs. Its unique key must win, never an unconditional v1 upsert.
    monkeypatch.setattr(projection, "read_basis", AsyncMock(return_value=None))
    with pytest.raises(IntegrityError):
        async with session_maker() as session, session.begin():
            await projection.restore_note(
                session,
                external_id=note_id,
                fields=fields,
                markdown_content="stale candidate",
                checksum="different",
                expected=None,
            )
    async with session_maker() as session:
        content = await session.scalar(
            select(NoteContent).where(NoteContent.project_id == test_project.id)
        )
        assert content is not None
        assert content.markdown_content == markdown
        assert content.db_version == 1


@pytest.mark.asyncio
async def test_a_published_move_keeps_times_and_repoints_conflict_keys(
    repository, test_project, session_maker, local_time_with_offset
):
    """The layout UPDATE skips Entity's validators; it must do what they would."""
    note_id = uuid4()
    timed = (
        f"---\ntitle: A\ntype: note\npermalink: {test_project.name}/notes/a\n"
        "created: '2025-01-02T03:04:05+00:00'\nmodified: '2025-02-03T04:05:06+00:00'\n"
        "---\n\nOriginal\n"
    )
    publish(repository, timed, note_id)
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    publish(repository, timed, note_id, path="archive/Moved Note.md")

    moved = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker
    )

    assert moved.notes_moved == 1
    async with session_maker() as session:
        entity = await session.scalar(select(Entity).where(Entity.external_id == str(note_id)))
        assert entity is not None
        assert (entity.file_path_key, entity.file_path_permalink) == (
            "archive/moved note.md",
            "archive/moved-note",
        )
        # The same instants, not the UTC wall time read back as local time.
        assert entity.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert entity.updated_at == datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)


@pytest.mark.asyncio
async def test_a_known_published_basis_proves_moved_notes_without_history(
    repository, markdown, test_project, session_maker, monkeypatch
):
    """A folder move publishes on top of a basis it read; that basis is the proof (V1)."""
    note_id = uuid4()
    publish(repository, markdown, note_id)
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    basis = repository.snapshot()
    publish(repository, markdown, note_id, path="archive/a.md")

    def no_history(*args, **kwargs):
        pytest.fail("A note held by the known basis needs no search through history")

    monkeypatch.setattr(GitProjectRepository, "contains_published_note_identity", no_history)
    moved = await project_git_content(
        repository,
        project_id=test_project.id,
        session_maker=session_maker,
        known_ancestor=basis,
    )

    assert moved.notes_moved == 1
    async with session_maker() as session:
        entity = await session.scalar(select(Entity).where(Entity.external_id == str(note_id)))
        assert entity is not None and entity.file_path == "archive/a.md"


@pytest.mark.asyncio
async def test_a_known_basis_outside_the_targets_history_is_refused(
    repository, markdown, test_project, session_maker
):
    # A private candidate exists in the repository but was never published.
    private = repository.candidate_snapshot(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=repository.head(),
            files={"notes/private.md": markdown},
            note_ids={"notes/private.md": uuid4()},
        )
    )
    publish(repository, markdown, uuid4())

    with pytest.raises(GitProjectionConflict, match="not published ancestry"):
        await project_git_content(
            repository,
            project_id=test_project.id,
            session_maker=session_maker,
            known_ancestor=private,
        )


# --- I1: a note whose UUID another project still owns waits; the rest is projected ---


def canonical(test_project: Project, path: str) -> str:
    stem = path.removesuffix(".md")
    return f"---\ntitle: {stem}\ntype: note\npermalink: {test_project.name}/{stem}\n---\n\nBody\n"


async def owner_project(test_project, session_maker, repository, note_id: UUID):
    """Project `other`, whose SQL still holds `note_id`, and the host identities."""
    from basic_memory.repository.git_host_identities import GitHostIdentities, GitHostProject

    async with session_maker() as session, session.begin():
        other = Project(name="other", path=str(repository.path.parent / "other"), is_active=True)
        session.add(other)
        await session.flush()
        session.add(local_entity(other.id, note_id, "owned.md"))
        other_id, other_uuid = other.id, UUID(other.external_id)
    other_repository = GitProjectRepository.create(
        repository.path.parent / "other.git", project_external_id=other_uuid
    )
    identities = GitHostIdentities(
        [
            GitHostProject(
                test_project.id, UUID(test_project.external_id), test_project.name, repository
            ),
            GitHostProject(other_id, other_uuid, "other", other_repository),
        ]
    )
    return other_id, other_repository, identities


async def graph_progress(session_maker, project_id: int, commit: str) -> None:
    from basic_memory.models import GitProjectionProgress

    async with session_maker() as session, session.begin():
        for stage in ("graph", "vectors"):
            row = await session.get(GitProjectionProgress, (project_id, stage))
            if row is None:
                row = GitProjectionProgress(
                    project_id=project_id, stage=stage, projection_key="1", target_commit=commit
                )
                session.add(row)
            row.completed_commit = row.reusable_commit = commit


async def graph_completed(session_maker, project_id: int) -> list[str | None]:
    from basic_memory.models import GitProjectionProgress

    async with session_maker() as session:
        rows = await session.scalars(
            select(GitProjectionProgress)
            .where(GitProjectionProgress.project_id == project_id)
            .order_by(GitProjectionProgress.stage)
        )
        return [row.completed_commit for row in rows]


@pytest.mark.asyncio
async def test_a_waiting_note_is_skipped_while_the_rest_is_projected(
    repository, test_project, session_maker, monkeypatch
):
    from basic_memory.repository.git_projection_repository import GitProjectionDeferred
    from basic_memory.services import git_project_projection as module

    kept, removed, waiting = uuid4(), uuid4(), uuid4()
    repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=repository.head(),
            files={p: canonical(test_project, p) for p in ("kept.md", "removed.md")},
            note_ids={"kept.md": kept, "removed.md": removed},
        )
    )
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    other_id, _, identities = await owner_project(test_project, session_maker, repository, waiting)
    # Two commits SQL has not seen (as from a peer): `kept` moves and `removed` goes, then
    # `waiting` takes the very path `kept` left. `other` released it in Git (its head
    # never held it), so only its SQL lags (rule a).
    for files, note_ids in (
        ({"moved/kept.md": canonical(test_project, "kept.md")}, {"moved/kept.md": kept}),
        (
            {
                "moved/kept.md": canonical(test_project, "kept.md"),
                "kept.md": canonical(test_project, "waiting.md"),
            },
            {"moved/kept.md": kept, "kept.md": waiting},
        ),
    ):
        repository.publish(
            repository.prepare(
                transaction_id=uuid4(),
                base_commit=repository.head(),
                files=files,
                note_ids=note_ids,
            )
        )
    await graph_progress(session_maker, other_id, repository.head())
    invalidations = 0
    invalidate = module.invalidate_git_graphs

    async def counted(session):
        nonlocal invalidations
        invalidations += 1
        await invalidate(session)

    monkeypatch.setattr(module, "invalidate_git_graphs", counted)
    with pytest.raises(GitProjectionDeferred) as deferred:
        await project_git_content(
            repository,
            project_id=test_project.id,
            session_maker=session_maker,
            identities=identities,
        )
    assert [note.external_id for note in deferred.value.waiting] == [waiting]
    assert deferred.value.owner_project_ids == {other_id}
    assert invalidations == 1  # the structural change of this pass, once
    async with session_maker() as session:
        rows = dict(
            (
                await session.execute(
                    select(Entity.external_id, Entity.file_path).where(
                        Entity.project_id == test_project.id
                    )
                )
            ).all()
        )
    assert rows == {str(kept): "moved/kept.md"}
    # Later passes wait again without touching the graph progress of any project.
    await graph_progress(session_maker, other_id, repository.head())
    for _ in range(3):
        with pytest.raises(GitProjectionDeferred):
            await project_git_content(
                repository,
                project_id=test_project.id,
                session_maker=session_maker,
                identities=identities,
            )
    assert invalidations == 1
    assert await graph_completed(session_maker, other_id) == [repository.head()] * 2
    # The owner releases the UUID (its own projection drops the row): the note arrives.
    async with session_maker() as session, session.begin():
        entity = await session.scalar(select(Entity).where(Entity.external_id == str(waiting)))
        await session.delete(entity)
    result = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker, identities=identities
    )
    assert result.notes_total == 2
    async with session_maker() as session:
        entity = await session.scalar(select(Entity).where(Entity.external_id == str(waiting)))
        assert entity is not None
        assert (entity.project_id, entity.file_path) == (test_project.id, "kept.md")


@pytest.mark.asyncio
async def test_a_real_duplicate_still_refuses_the_snapshot_with_names(
    repository, test_project, session_maker
):
    note = uuid4()
    _, other_repository, identities = await owner_project(
        test_project, session_maker, repository, note
    )
    other_repository.publish(
        other_repository.prepare(
            transaction_id=uuid4(),
            base_commit=other_repository.head(),
            files={"owned.md": "owned"},
            note_ids={"owned.md": note},
        )
    )
    publish(repository, canonical(test_project, "notes/a.md"), note)
    with pytest.raises(GitProjectionConflict, match="another project or path") as refused:
        await project_git_content(
            repository,
            project_id=test_project.id,
            session_maker=session_maker,
            identities=identities,
        )
    assert f"{note} (owned by project 'other'" in str(refused.value)
    assert type(refused.value) is GitProjectionConflict


@pytest.mark.asyncio
async def test_an_unchanged_note_is_not_restored_again_and_removals_still_run(
    repository, markdown, test_project, session_maker, monkeypatch
):
    """Only changed notes cost a restore; every pass used to touch the whole project."""
    first, second = uuid4(), uuid4()
    other = markdown.replace("notes/a", "notes/b").replace("title: A", "title: B")

    def publish_tree(files: dict[str, str]) -> None:
        ids = {"notes/a.md": first, "notes/b.md": second}
        repository.publish(
            repository.prepare(
                transaction_id=uuid4(),
                base_commit=repository.head(),
                files=files,
                note_ids={path: ids[path] for path in files},
            )
        )

    publish_tree({"notes/a.md": markdown, "notes/b.md": other})
    await project_git_content(repository, project_id=test_project.id, session_maker=session_maker)
    restored: list[UUID] = []
    restore = GitProjectionRepository.restore_note

    async def recording(self, session, **arguments):
        restored.append(arguments["external_id"])
        return await restore(self, session, **arguments)

    monkeypatch.setattr(GitProjectionRepository, "restore_note", recording)
    publish_tree({"notes/a.md": markdown, "notes/b.md": other + "Neu\n"})
    changed = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker
    )
    assert restored == [second] and changed.notes_changed == 1
    publish_tree({"notes/a.md": markdown})
    removed = await project_git_content(
        repository, project_id=test_project.id, session_maker=session_maker
    )
    assert restored == [second] and removed.notes_removed == 1
    async with session_maker() as session:
        rows = await session.execute(
            select(Entity.file_path, NoteContent.markdown_content)
            .join(NoteContent, NoteContent.entity_id == Entity.id)
            .where(Entity.project_id == test_project.id)
        )
        assert [tuple(row) for row in rows] == [("notes/a.md", markdown)]
