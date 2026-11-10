"""Real API acceptance with opt-in lifecycle, fixed Git basis and deferred indexes."""

import asyncio
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text, update

import basic_memory.services.git_project_runtime as git_runtime_module

from basic_memory.deps.services import get_note_content_mutation_service
from basic_memory.index.git_projection import GitProjectionPipeline, build_git_projection_pipeline
from basic_memory.index.local_notes import LocalAcceptedNotePreparerFactory
from basic_memory.indexing.accepted_note_mutation_runner import (
    AcceptedNoteCreateMutation,
    AcceptedNoteMutationActor,
    AcceptedNoteMutationDependencies,
    AcceptedNoteMutationMovePolicy,
)
from basic_memory.models import Entity, NoteContent, Observation, Project, Relation
from basic_memory.repository.accepted_note_repositories import AcceptedNoteRepositories
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.repository.project_repository import ProjectRepository
from basic_memory.repository.search_repository import create_search_repository
from basic_memory.schemas.base import Entity as EntitySchema
from basic_memory.services.git_project_runtime import GitProjectRuntime
from basic_memory.services.note_content_writes import (
    NoteContentMutationService,
    NoteContentMutationServiceError,
)


@pytest.fixture
async def runtime(test_project, engine_factory, app_config, app, tmp_path_factory):
    root = Path(test_project.path) / "git-notes"
    root.mkdir()
    test_project.path = str(root)
    app_config.projects[test_project.name].path = str(root)
    async with engine_factory[1]() as session, session.begin():
        await session.execute(
            update(Project).where(Project.id == test_project.id).values(path=str(root))
        )
    repository = GitProjectRepository.create(
        tmp_path_factory.mktemp("git-runtime") / "project.git",
        project_external_id=UUID(test_project.external_id),
    )
    pipeline = build_git_projection_pipeline(
        repository, project=test_project, session_maker=engine_factory[1], app_config=app_config
    )
    repositories = AcceptedNoteRepositories(
        external_vector_cleaner_factory=lambda project_id: create_search_repository(
            engine_factory[1], project_id, app_config
        )
    )
    dependencies = AcceptedNoteMutationDependencies(
        project_repository=ProjectRepository(),
        lookup_repositories=repositories,
        write_repositories=repositories,
        preparer_factory=LocalAcceptedNotePreparerFactory(
            session_maker=engine_factory[1], app_config=app_config
        ),
        move_policy=AcceptedNoteMutationMovePolicy(
            update_permalinks_on_move=app_config.update_permalinks_on_move
        ),
        verify_storage_absent_on_create=True,
    )
    runtime = GitProjectRuntime(pipeline, dependencies, poll_interval=0.02)

    async def git_mutator():
        return NoteContentMutationService(
            session_maker=engine_factory[1],
            mutation_dependencies=dependencies,
            mutation_executor=runtime,
        )

    app.dependency_overrides[get_note_content_mutation_service] = git_mutator
    try:
        yield runtime
    finally:
        app.dependency_overrides.pop(get_note_content_mutation_service, None)


def create_request(runtime, title="Runtime"):
    return AcceptedNoteCreateMutation(
        project_external_id=runtime.pipeline.project_external_id,
        data=EntitySchema(title=title, directory="notes", content="- [fact] Runtimeprobe\n"),
        actor=AcceptedNoteMutationActor(None),
        source="git-runtime-test",
    )


async def wait_ready(runtime):
    async with asyncio.timeout(15):
        while not (await runtime.worker.status()).caught_up:
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_api_returns_during_blocked_graph_and_vectors(runtime, client, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    apply = GitProjectionPipeline.apply

    async def slow_indexes(self, stage, commit_id, *, graph_base_commit=None):
        if stage in {"graph", "vectors"}:
            entered.set()
            await release.wait()
        await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)

    async def forbidden_legacy_finish(self, result, cache_change):
        pytest.fail("Legacy graph/search work must not run on a Git response path")

    monkeypatch.setattr(GitProjectionPipeline, "apply", slow_indexes)
    monkeypatch.setattr(NoteContentMutationService, "_finish_mutation", forbidden_legacy_finish)
    async with runtime.run():
        await asyncio.wait_for(entered.wait(), 10)
        response = await asyncio.wait_for(
            client.post(
                f"/v2/projects/{runtime.pipeline.project_external_id}/knowledge/write",
                json={
                    "note": {
                        "title": "Runtime",
                        "directory": "notes",
                        "content": "- [fact] Runtimeprobe\n\n- relates_to [[Target]]\n",
                    }
                },
            ),
            10,
        )
        assert response.status_code == 200, response.text
        note = response.json()["entity"]
        assert not release.is_set()
        assert runtime.pipeline.repository.snapshot().files[note["file_path"]] == note["content"]
        async with runtime.pipeline.session_maker() as session:
            assert await session.scalar(select(Observation.id)) is None
            assert await session.scalar(select(Relation.id)) is None
        assert not await runtime.pipeline.search_service.repository.get_entity_search_rows(
            note["id"]
        )
        assert not await runtime.pipeline.search_service.repository.get_entity_chunk_manifest(
            note["id"]
        )
        assert not (await runtime.worker.status()).caught_up
        release.set()
        await wait_ready(runtime)


@pytest.mark.asyncio
async def test_api_update_edit_move_delete_publish_and_keep_uuid(runtime, client):
    base = f"/v2/projects/{runtime.pipeline.project_external_id}/knowledge"
    async with runtime.run():
        response = await client.post(
            base + "/write",
            json={"note": {"title": "Runtime", "directory": "notes", "content": "Original"}},
        )
        assert response.status_code == 200, response.text
        note = response.json()["entity"]
        note_id = note["external_id"]
        commits = [runtime.pipeline.repository.head()]
        response = await client.put(
            base + f"/entities/{note_id}",
            json={"title": "Runtime", "directory": "notes", "content": "Updated"},
        )
        assert response.status_code in {200, 202}, response.text
        commits.append(runtime.pipeline.repository.head())
        response = await client.patch(
            base + f"/entities/{note_id}",
            json={"operation": "append", "content": "\n- [fact] Appendedprobe\n"},
        )
        assert response.status_code == 202, response.text
        commits.append(runtime.pipeline.repository.head())
        await wait_ready(runtime)
        response = await client.put(
            base + f"/entities/{note_id}/move", json={"destination_path": "notes/Moved.md"}
        )
        assert response.status_code in {200, 202}, response.text
        commits.append(runtime.pipeline.repository.head())
        await wait_ready(runtime)
        snapshot = runtime.pipeline.repository.snapshot()
        assert snapshot.manifest.note_ids == {"notes/Moved.md": UUID(note_id)}
        assert not (runtime.pipeline.file_service.base_path / note["file_path"]).exists()
        assert (runtime.pipeline.file_service.base_path / "notes/Moved.md").exists()
        response = await client.delete(base + f"/entities/{note_id}")
        assert response.status_code in {200, 202}, response.text
        commits.append(runtime.pipeline.repository.head())
        await wait_ready(runtime)
        assert runtime.pipeline.repository.snapshot().files == {}
        assert not (runtime.pipeline.file_service.base_path / "notes/Moved.md").exists()
        assert len(set(commits)) == len(commits)
        # Missing delete is a no-op and must not add another Git commit.
        response = await client.delete(base + f"/entities/{note_id}")
        assert response.status_code in {200, 202}
        assert runtime.pipeline.repository.head() == commits[-1]


@pytest.mark.asyncio
async def test_startup_restores_sql_before_yielding_and_before_new_preparation(runtime):
    repository = runtime.pipeline.repository
    identity = uuid4()
    text = "---\ntitle: Existing\ntype: note\npermalink: test-project/notes/existing\n---\n\nExisting\n"
    first = repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=repository.head(),
            files={"notes/Existing.md": text},
            note_ids={"notes/Existing.md": identity},
        )
    )
    async with runtime.run():
        async with runtime.pipeline.session_maker() as session:
            content = await session.scalar(
                select(NoteContent).where(NoteContent.external_id == str(identity))
            )
            assert content is not None and content.markdown_content == text
        # Simulate a peer publication while this installation's SQL is still old.
        updated = text + "Peerpublicationprobe\n"
        second = repository.publish(
            repository.prepare(
                transaction_id=uuid4(),
                base_commit=first.commit_id,
                files={"notes/Existing.md": updated},
                note_ids={"notes/Existing.md": identity},
            )
        )
        await runtime.accept(create_request(runtime))
        snapshot = repository.snapshot()
        assert snapshot.files["notes/Existing.md"] == updated
        assert repository.contains_published_note_content(
            snapshot.commit_id, "notes/Existing.md", updated
        )
        assert snapshot.commit_id != second.commit_id
        await wait_ready(runtime)


@pytest.mark.asyncio
async def test_mutation_admission_requires_running_and_matching_project(runtime):
    with pytest.raises(GitProjectionConflict, match="not accepting"):
        await runtime.accept(create_request(runtime))
    async with runtime.run():
        with pytest.raises(GitProjectionConflict, match="different"):
            await runtime.accept(replace(create_request(runtime), project_external_id=str(uuid4())))


@pytest.mark.asyncio
async def test_two_api_writes_are_serialized_without_losing_either(runtime, client):
    async with runtime.run():

        async def write(title):
            return await client.post(
                f"/v2/projects/{runtime.pipeline.project_external_id}/knowledge/write",
                json={"note": {"title": title, "directory": "notes", "content": title}},
            )

        responses = await asyncio.gather(write("First"), write("Second"))
        assert [response.status_code for response in responses] == [200, 200]
        assert set(runtime.pipeline.repository.snapshot().files) == {
            "notes/First.md",
            "notes/Second.md",
        }
        await wait_ready(runtime)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_publication_thread_and_recovers_sql(
    runtime, monkeypatch
):
    publish = GitProjectRepository.publish
    entered, release = threading.Event(), threading.Event()

    def slow_publish(self, prepared):
        entered.set()
        assert release.wait(10)
        return publish(self, prepared)

    monkeypatch.setattr(GitProjectRepository, "publish", slow_publish)
    async with runtime.run():
        task = asyncio.create_task(runtime.accept(create_request(runtime)))
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done() and runtime.gate.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "notes/Runtime.md" in runtime.pipeline.repository.snapshot().files
        await wait_ready(runtime)
        async with runtime.pipeline.session_maker() as session:
            assert (
                await session.scalar(
                    select(Entity.id).where(Entity.file_path == "notes/Runtime.md")
                )
                is not None
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["git_before", "git_after", "sql_before", "sql_after"])
async def test_whole_asgi_process_loss_recovers_the_git_outcome(
    runtime, engine_factory, test_project, app_config, tmp_path, fault
):
    config_path = tmp_path / "child-config.json"
    config_path.write_text(app_config.model_dump_json())
    initial = runtime.pipeline.repository.head()

    def run_child() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                sys.executable,
                "test-int/fixtures/git_runtime_crash.py",
                str(engine_factory[0].url.database),
                str(runtime.pipeline.repository.path),
                str(test_project.id),
                str(config_path),
                fault,
            ],
            env={**os.environ, "PYTHONPATH": "src:tests"},
            capture_output=True,
            timeout=30,
        )

    child = await asyncio.to_thread(run_child)
    assert child.returncode == 79, child.stderr.decode()
    expected = fault != "git_before"
    snapshot = runtime.pipeline.repository.snapshot()
    assert ("notes/Crash.md" in snapshot.files) == expected
    assert (snapshot.commit_id != initial) == expected
    async with runtime.run():
        # The yield itself guarantees SQL content, without waiting for indexes.
        async with engine_factory[1]() as session:
            content = await session.scalar(
                select(NoteContent).where(NoteContent.file_path == "notes/Crash.md")
            )
            assert (content is not None) == expected
            if content is not None:
                assert content.markdown_content == snapshot.files["notes/Crash.md"]
                assert content.external_id == str(snapshot.manifest.note_ids["notes/Crash.md"])
        await wait_ready(runtime)
    assert runtime.pipeline.repository.head() == snapshot.commit_id


@pytest.mark.asyncio
async def test_peer_publication_during_preparation_returns_conflict_and_keeps_peer(
    runtime, client, monkeypatch
):
    run_create = git_runtime_module.run_accepted_note_create
    repository = runtime.pipeline.repository
    peer_text = "---\ntitle: Peer\ntype: note\npermalink: test-project/notes/peer\n---\nPeer\n"
    peer_id = uuid4()

    async def publish_peer_first(session, *, request, dependencies):
        base = repository.snapshot()
        repository.publish(
            repository.prepare(
                transaction_id=uuid4(),
                base_commit=base.commit_id,
                files={**base.files, "notes/Peer.md": peer_text},
                note_ids={**base.manifest.note_ids, "notes/Peer.md": peer_id},
            )
        )
        return await run_create(session, request=request, dependencies=dependencies)

    monkeypatch.setattr(git_runtime_module, "run_accepted_note_create", publish_peer_first)
    async with runtime.run():
        response = await client.post(
            f"/v2/projects/{runtime.pipeline.project_external_id}/knowledge/write",
            json={
                "note": {"title": "Rejected", "directory": "notes", "content": "Must stay private"}
            },
        )
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["message"] == "Git project advanced during preparation"
        assert repository.snapshot().files == {"notes/Peer.md": peer_text}
        await wait_ready(runtime)
        async with runtime.pipeline.session_maker() as session:
            assert await session.scalar(select(Entity.id).where(Entity.title == "Rejected")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["bytes", "checksum", "identity"])
async def test_private_sql_state_cannot_become_a_new_git_basis(runtime, change):
    async with runtime.run():
        await runtime.accept(create_request(runtime))
        await wait_ready(runtime)
        initial = runtime.pipeline.repository.head()
        async with runtime.pipeline.session_maker() as session, session.begin():
            if change == "identity":
                session.add(
                    Entity(
                        project_id=runtime.pipeline.project_id,
                        title="Private",
                        file_path="private.md",
                        content_type="text/markdown",
                        note_type="note",
                    )
                )
            else:
                await session.execute(
                    update(NoteContent).values(
                        **(
                            {"markdown_content": "Unpublished SQL"}
                            if change == "bytes"
                            else {"db_checksum": "forged"}
                        )
                    )
                )
        with pytest.raises(GitProjectionConflict, match="outside"):
            await runtime.accept(create_request(runtime, "Another"))
        assert runtime.pipeline.repository.head() == initial


@pytest.mark.asyncio
async def test_queued_mutation_is_refused_when_shutdown_closes_admission(runtime):
    context = runtime.run()
    await context.__aenter__()
    await runtime.gate.acquire()
    task = asyncio.create_task(runtime.accept(create_request(runtime)))
    await asyncio.sleep(0)
    closing = asyncio.create_task(context.__aexit__(None, None, None))
    await asyncio.sleep(0)
    runtime.gate.release()
    with pytest.raises(GitProjectionConflict, match="stopped"):
        await task
    await closing
    assert runtime.pipeline.repository.snapshot().files == {}


@pytest.mark.asyncio
async def test_deletion_unresolves_incoming_links_on_a_fresh_sqlite_connection(
    runtime, engine_factory
):
    repository = runtime.pipeline.repository
    files = {
        "Source.md": "---\ntitle: Source\ntype: note\npermalink: test-project/source\n---\n\n- relates_to [[Target]]\n",
        "Target.md": "---\ntitle: Target\ntype: note\npermalink: test-project/target\n---\nTarget\n",
    }
    ids = {path: uuid4() for path in files}
    repository.publish(
        repository.prepare(
            transaction_id=uuid4(), base_commit=repository.head(), files=files, note_ids=ids
        )
    )
    async with runtime.run():
        await wait_ready(runtime)
    # No inherited scoped-session PRAGMA from earlier graph indexing may help us.
    await engine_factory[0].dispose()
    async with engine_factory[1]() as session:
        assert await session.scalar(text("PRAGMA foreign_keys")) == 0
    del files["Target.md"]
    del ids["Target.md"]
    repository.publish(
        repository.prepare(
            transaction_id=uuid4(), base_commit=repository.head(), files=files, note_ids=ids
        )
    )
    async with runtime.run():
        async with engine_factory[1]() as session:
            relation = await session.scalar(select(Relation))
            assert relation is not None and relation.to_id is None and relation.to_name == "Target"
        await wait_ready(runtime)


@pytest.mark.asyncio
async def test_sql_commit_failure_reports_published_commit_and_recovers(
    runtime, monkeypatch, engine_factory
):
    dialect = engine_factory[0].sync_engine.dialect
    do_commit = dialect.do_commit
    armed = True

    def fail_commit(connection):
        nonlocal armed
        if armed and runtime.last_publication is not None:
            armed = False
            raise RuntimeError("SQL commit unavailable after Git publication")
        do_commit(connection)

    monkeypatch.setattr(dialect, "do_commit", fail_commit)
    async with runtime.run():
        with pytest.raises(NoteContentMutationServiceError) as raised:
            await runtime.accept(create_request(runtime))
        assert raised.value.status_code == 503
        assert isinstance(raised.value.detail, dict)
        assert raised.value.detail["published_commit"] == runtime.pipeline.repository.head()
        await wait_ready(runtime)
        async with engine_factory[1]() as session:
            content = await session.scalar(select(NoteContent))
            assert content is not None and "Runtimeprobe" in content.markdown_content


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["accepted_checksum", "source_path"])
async def test_invalid_core_acceptance_cannot_publish_or_authorize_cleanup(
    runtime, monkeypatch, fault
):
    async with runtime.run():
        if fault == "source_path":
            await runtime.accept(create_request(runtime))
            await wait_ready(runtime)
            run_move = git_runtime_module.run_accepted_note_move

            async def wrong_source(session, *, request, dependencies):
                result = await run_move(session, request=request, dependencies=dependencies)
                assert result.change.project_change is not None
                return replace(
                    result,
                    change=replace(
                        result.change,
                        project_change=replace(
                            result.change.project_change, previous_file_path="private/editor.md"
                        ),
                    ),
                )

            monkeypatch.setattr(git_runtime_module, "run_accepted_note_move", wrong_source)
            async with runtime.pipeline.session_maker() as session:
                note_id = await session.scalar(select(Entity.external_id))
            from basic_memory.indexing.accepted_note_mutation_runner import AcceptedNoteMoveMutation

            request = AcceptedNoteMoveMutation(
                project_external_id=runtime.pipeline.project_external_id,
                entity_external_id=note_id,
                destination_path="Moved.md",
                actor=AcceptedNoteMutationActor(None),
                source="test",
            )
        else:
            run_create = git_runtime_module.run_accepted_note_create

            async def invalid_checksum(session, *, request, dependencies):
                result = await run_create(session, request=request, dependencies=dependencies)
                assert result.change.project_change is not None
                content = await session.get(NoteContent, result.change.project_change.entity_id)
                assert content is not None
                content.db_checksum = "inconsistent-adapter-result"
                return result

            monkeypatch.setattr(git_runtime_module, "run_accepted_note_create", invalid_checksum)
            request = create_request(runtime)
        before = runtime.pipeline.repository.head()
        with pytest.raises(GitProjectionConflict):
            await runtime.accept(request)
        assert runtime.pipeline.repository.head() == before
        await wait_ready(runtime)


@pytest.mark.asyncio
async def test_clean_restart_preserves_synced_materialization_state(runtime):
    async with runtime.run():
        await runtime.accept(create_request(runtime))
        await wait_ready(runtime)
    async with runtime.run():
        async with runtime.pipeline.session_maker() as session:
            content = await session.scalar(select(NoteContent))
            assert content is not None and content.file_write_status == "synced"
            assert content.file_version == content.db_version
        assert (await runtime.worker.status()).caught_up


@pytest.mark.asyncio
async def test_real_api_append_reindexes_only_the_changed_note(runtime, client, monkeypatch):
    from basic_memory.index.local_dependencies import LocalMarkdownFileIndexer

    async with runtime.run():
        first = await runtime.accept(create_request(runtime, "First"))
        await runtime.accept(create_request(runtime, "Second"))
        await wait_ready(runtime)
        assert first.project_change is not None
        changed = first.project_change
        calls = []
        index = LocalMarkdownFileIndexer.index_file

        async def record(self, path, *, source):
            calls.append(path)
            return await index(self, path, source=source)

        monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", record)
        response = await client.patch(
            f"/v2/projects/{runtime.pipeline.project_external_id}/knowledge/entities/{changed.note_external_id}",
            json={"operation": "append", "content": "\n- [fact] Apideltaprobe\n"},
        )
        assert response.status_code == 202, response.text
        await wait_ready(runtime)
        assert calls == [changed.file_path]


@pytest.mark.asyncio
async def test_without_host_identities_the_commit_guard_still_checks_sql(
    runtime, engine_factory, tmp_path
):
    """A runtime built without the host view (tests, tools) still refuses a UUID that
    another project's SQL holds when a draft with a chosen UUID commits (I1)."""
    from basic_memory.repository.git_transaction_repository import GitTransactionError
    from basic_memory.schemas.git_transactions import (
        GitTransactionCommit,
        GitTransactionControl,
        GitTransactionStart,
    )

    note = uuid4()
    async with engine_factory[1]() as session, session.begin():
        other = Project(name="other", path=str(tmp_path / "other"), is_active=True)
        session.add(other)
        await session.flush()
        session.add(
            Entity(
                project_id=other.id,
                external_id=str(note),
                title="Other",
                note_type="note",
                content_type="text/markdown",
                file_path="other.md",
            )
        )
    assert runtime.pipeline.identities is None
    async with runtime.run():
        transactions = runtime.transactions
        started = await transactions.start(GitTransactionStart())
        await transactions.write_identified(
            started.transaction_id,
            started.transaction_key,
            path="kept.md",
            content="---\ntitle: Kept\ntype: note\npermalink: kept\n---\n\nBody\n",
            note_id=note,
            expected_revision=0,
        )
        with pytest.raises(GitTransactionError, match=f"Note UUID {note} is still owned"):
            await transactions.commit(
                started.transaction_id,
                started.transaction_key,
                GitTransactionCommit(expected_revision=1),
            )
        rolled_back = await transactions.rollback_admitted(
            started.transaction_id,
            started.transaction_key,
            GitTransactionControl(expected_revision=1),
        )
        assert rolled_back.state == "rolled_back"
