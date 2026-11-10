"""Durable worker recovery with actual Git, SQLite, BM graph/FTS and vectors."""

import asyncio
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from basic_memory.index.git_projection import (
    GitProjectionPipeline,
    PublishedGitIndexWriter,
    build_git_projection_pipeline,
)
from basic_memory.index.git_projection_worker import GitPassOutcome, GitProjectionWorker
from basic_memory.indexing.models import IndexFrontmatterUpdate
from basic_memory.models import Entity, NoteContent, Observation, Relation
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.repository.semantic_chunking import build_vector_chunk_records
from basic_memory.runtime.vector_sync import VectorSyncBatchResult
from basic_memory.services.search_service import SearchService


@pytest.fixture
def pipeline(tmp_path_factory, test_project, engine_factory, app_config) -> GitProjectionPipeline:
    repository = GitProjectRepository.create(
        tmp_path_factory.mktemp("git-worker") / "project.git",
        project_external_id=UUID(test_project.external_id),
    )
    return build_git_projection_pipeline(
        repository, project=test_project, session_maker=engine_factory[1], app_config=app_config
    )


def publish(pipeline: GitProjectionPipeline, text: str = "Workerprobe") -> str:
    repository = pipeline.repository
    snapshot = repository.snapshot()
    files = {
        "notes/Source.md": (
            "---\ntitle: Source\ntype: note\npermalink: test-project/notes/source\n---\n\n"
            f"# Source\n\n- [fact] {text}\n\n- relates_to [[Target]]\n"
        ),
        "notes/Target.md": (
            "---\ntitle: Target\ntype: note\npermalink: test-project/notes/target\n---\n\nTarget\n"
        ),
    }
    return repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=snapshot.commit_id,
            files=files,
            note_ids={path: snapshot.manifest.note_ids.get(path, uuid4()) for path in files},
        )
    ).commit_id


async def wait_ready(worker: GitProjectionWorker) -> None:
    async with asyncio.timeout(15):
        while not (await worker.status()).caught_up:
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_worker_catches_up_real_read_graph_full_text_and_vectors(
    pipeline, client: AsyncClient
):
    commit = publish(pipeline)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await wait_ready(worker)
    status = await worker.status()
    assert status.source_commit == commit
    assert all(row.completed_commit == commit for row in status.stages)
    snapshot = pipeline.repository.snapshot()
    ids = {}
    async with pipeline.session_maker() as session:
        for note in snapshot.manifest.notes:
            entity = await session.scalar(
                select(Entity).where(Entity.external_id == str(note.external_id))
            )
            assert entity is not None
            ids[note.path] = entity.id
            content = await session.get(NoteContent, entity.id)
            assert content is not None and content.file_write_status == "synced"
            assert content.markdown_content == snapshot.files[note.path]
            read = await client.get(
                f"/v2/projects/{pipeline.project_external_id}/knowledge/entities/{note.external_id}"
            )
            assert read.status_code == 200, read.text
            assert read.json()["content"] == snapshot.files[note.path]
            assert (
                Path(pipeline.file_service.base_path) / note.path
            ).read_bytes() == snapshot.files[note.path].encode()
        assert (
            await session.scalar(
                select(Observation.content).where(Observation.entity_id == ids["notes/Source.md"])
            )
            == "Workerprobe"
        )
        assert (
            await session.scalar(
                select(Relation.to_id).where(Relation.from_id == ids["notes/Source.md"])
            )
            == ids["notes/Target.md"]
        )
    search = pipeline.search_service.repository
    rows = await search.get_entity_search_rows(ids["notes/Source.md"])
    assert Counter(row.type for row in rows) == {"entity": 1, "observation": 1, "relation": 1}
    expected = {record["chunk_key"] for record in build_vector_chunk_records(rows).records}
    manifest = await search.get_entity_chunk_manifest(ids["notes/Source.md"])
    assert {row.chunk_key for row in manifest} == expected
    assert all(row.embedding_status == "ready" for row in manifest)
    assert await search.get_entity_physical_chunk_keys(ids["notes/Source.md"]) == expected
    found = await client.request(
        "QUERY",
        f"/v2/projects/{pipeline.project_external_id}/search/",
        json={"text": "Workerprobe", "retrieval_mode": "fts"},
    )
    assert found.status_code == 200 and "Workerprobe" in found.text
    assert pipeline.repository.head() == commit


@pytest.mark.asyncio
async def test_vector_failure_does_not_delay_newer_content_and_restart_resumes(
    pipeline, monkeypatch: pytest.MonkeyPatch
):
    first = publish(pipeline, "Firstprobe")
    entered, release = asyncio.Event(), asyncio.Event()
    apply = GitProjectionPipeline.apply

    async def slow_vectors(self, stage, commit_id, *, graph_base_commit=None):
        if stage == "vectors":
            entered.set()
            await release.wait()
            raise RuntimeError("Embedding provider temporarily unavailable")
        await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)

    monkeypatch.setattr(GitProjectionPipeline, "apply", slow_vectors)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await asyncio.wait_for(entered.wait(), 10)
        second = publish(pipeline, "Secondprobe")
        third = publish(pipeline, "Thirdprobe")
        assert first != second != third
        # No notify(): a durable Git publication alone must suffice for catch-up.
        async with asyncio.timeout(10):
            while True:
                status = await worker.status()
                if (
                    next(row for row in status.stages if row.stage == "graph").completed_commit
                    == third
                ):
                    break
                await asyncio.sleep(0.02)
        assert not status.caught_up
        assert next(row for row in status.stages if row.stage == "vectors").completed_commit is None
        release.set()
        async with asyncio.timeout(10):
            while not next(
                row for row in (await worker.status()).stages if row.stage == "vectors"
            ).last_error:
                await asyncio.sleep(0.02)
    # Reconstruct the owner/repository rather than retaining an in-memory queue.
    monkeypatch.setattr(GitProjectionPipeline, "apply", apply)
    reopened = replace(pipeline, repository=GitProjectRepository(pipeline.repository.path))
    restarted = GitProjectionWorker(reopened, poll_interval=0.02)
    async with restarted.run():
        await wait_ready(restarted)
    status = await restarted.status()
    assert status.caught_up and status.source_commit == third
    assert next(row for row in status.stages if row.stage == "vectors").attempts >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["content", "files", "graph", "vectors"])
@pytest.mark.parametrize("after_work", [False, True])
async def test_cancelled_stage_is_retried_without_false_completion(
    pipeline, monkeypatch, stage, after_work
):
    commit = publish(pipeline)
    entered = asyncio.Event()
    apply = GitProjectionPipeline.apply

    async def interrupted(self, current_stage, commit_id, *, graph_base_commit=None):
        if current_stage == stage:
            if after_work:
                await apply(self, current_stage, commit_id, graph_base_commit=graph_base_commit)
            entered.set()
            await asyncio.Future()
        await apply(self, current_stage, commit_id, graph_base_commit=graph_base_commit)

    monkeypatch.setattr(GitProjectionPipeline, "apply", interrupted)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await asyncio.wait_for(entered.wait(), 10)
    row = next(row for row in (await worker.status()).stages if row.stage == stage)
    assert row.running_commit == commit and row.completed_commit is None
    monkeypatch.setattr(GitProjectionPipeline, "apply", apply)
    restarted = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with restarted.run():
        await wait_ready(restarted)
    assert (
        next(row for row in (await restarted.status()).stages if row.stage == stage).attempts == 2
    )
    async with pipeline.session_maker() as session:
        assert set(await session.scalars(select(NoteContent.db_version))) == {1}


@pytest.mark.asyncio
async def test_editor_conflict_keeps_graph_and_vectors_pending(pipeline):
    publish(pipeline)
    target = pipeline.file_service.base_path / "notes/Source.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"Private editor draft")
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        async with asyncio.timeout(10):
            while not next(
                (row.last_error for row in (await worker.status()).stages if row.stage == "files"),
                None,
            ):
                await asyncio.sleep(0.02)
        status = await worker.status()
    rows = {row.stage: row for row in status.stages}
    assert rows["content"].completed_commit == status.source_commit
    assert all(rows[stage].completed_commit is None for stage in ("files", "graph", "vectors"))
    assert target.read_bytes() == b"Private editor draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "deferred", "unavailable"])
async def test_vector_partial_results_are_pending_not_ready(pipeline, monkeypatch, outcome):
    publish(pipeline)
    if outcome == "unavailable":

        async def disabled(self):
            return False

        monkeypatch.setattr(SearchService, "semantic_effectively_enabled", disabled)
    else:

        async def partial(self, entity_id):
            return VectorSyncBatchResult(
                entities_total=1,
                entities_synced=0,
                entities_failed=int(outcome == "failed"),
                entities_deferred=int(outcome == "deferred"),
                sample_errors=(outcome,),
            )

        monkeypatch.setattr(SearchService, "sync_entity_vectors", partial)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        async with asyncio.timeout(10):
            while not next(
                (
                    row.last_error
                    for row in (await worker.status()).stages
                    if row.stage == "vectors"
                ),
                None,
            ):
                await asyncio.sleep(0.02)
        status = await worker.status()
    rows = {row.stage: row for row in status.stages}
    assert rows["graph"].completed_commit == status.source_commit
    assert rows["vectors"].completed_commit is None
    assert not status.caught_up


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("BASIC_MEMORY_TEST_POSTGRES") == "1", reason="SQLite process proof")
@pytest.mark.parametrize("stage", ["content", "files", "graph", "vectors"])
async def test_process_death_after_work_before_checkpoint_recovers(
    pipeline, app_config, engine_factory, tmp_path, stage
):
    commit = publish(pipeline)
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(app_config.model_dump_json())
    database = engine_factory[0].url.database
    assert database is not None
    result = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "test-int/fixtures/git_projection_worker_crash.py",
            database,
            str(pipeline.repository.path),
            str(pipeline.project_id),
            str(config_path),
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
    assert result.returncode == 73, result.stderr
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    status = await worker.status()
    row = next(row for row in status.stages if row.stage == stage)
    assert row.running_commit == commit and row.completed_commit is None
    async with worker.run():
        await wait_ready(worker)
    assert (await worker.status()).caught_up


@pytest.mark.asyncio
async def test_worker_owns_its_lifecycle_and_retries_checkpoint_failure(pipeline, monkeypatch):
    publish(pipeline)
    with pytest.raises(ValueError, match="positive"):
        GitProjectionWorker(pipeline, poll_interval=0)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    assert not (await worker.status()).caught_up
    assert await worker._pass(vectors=True) is GitPassOutcome.done
    target = worker.progress.target
    failures = 0

    async def unavailable_once(commit_id, *, vector_key, graph_key="1"):
        nonlocal failures
        if failures == 0:
            failures += 1
            raise OSError("Transient checkpoint storage failure")
        await target(commit_id, vector_key=vector_key, graph_key=graph_key)

    monkeypatch.setattr(worker.progress, "target", unavailable_once)
    async with worker.run():
        with pytest.raises(RuntimeError, match="already running"):
            async with worker.run():
                pytest.fail("A second lifecycle must not start")
        worker.notify()
        await wait_ready(worker)
    assert failures == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["begin", "finish"])
async def test_target_advancing_at_checkpoint_boundary_is_coalesced(
    pipeline, monkeypatch, boundary
):
    first = publish(pipeline)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    begin, finish = worker.progress.begin, worker.progress.finish
    advanced = False

    async def advance_target():
        nonlocal advanced
        if not advanced:
            advanced = True
            commit_id = publish(pipeline, "Coalescedprobe")
            await worker.progress.target(
                commit_id, vector_key=pipeline.vector_key, graph_key=pipeline.graph_key
            )

    async def advance_before_begin(stage, commit_id, *, reusable_commit=None):
        await advance_target()
        return await begin(stage, commit_id, reusable_commit=reusable_commit)

    async def advance_before_finish(stage, commit_id, run_id, *, error=None):
        await advance_target()
        return await finish(stage, commit_id, run_id, error=error)

    monkeypatch.setattr(
        worker.progress,
        boundary,
        advance_before_begin if boundary == "begin" else advance_before_finish,
    )
    async with worker.run():
        await wait_ready(worker)
    assert (await worker.status()).source_commit != first
    assert advanced


@pytest.mark.asyncio
async def test_pipeline_rejects_missing_sql_private_file_and_head_change(pipeline):
    commit = publish(pipeline)
    with pytest.raises(GitProjectionConflict, match="SQL content"):
        await pipeline.apply("files", commit)
    with pytest.raises(GitProjectionConflict, match="SQL content"):
        await pipeline.apply("graph", commit)
    with pytest.raises(GitProjectionConflict, match="head advanced"):
        await pipeline.require_head("0" * 40)
    target = pipeline.file_service.base_path / "notes/Source.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"Unpublished editor bytes")
    with pytest.raises(GitProjectionConflict, match="differs from published"):
        await pipeline.require_file(pipeline.repository.snapshot(), "notes/Source.md")
    with pytest.raises(GitProjectionConflict, match="would rewrite"):
        await PublishedGitIndexWriter().write_frontmatter(
            IndexFrontmatterUpdate(path="notes/Source.md", metadata={"permalink": "other"})
        )
    assert target.read_bytes() == b"Unpublished editor bytes"


@pytest.mark.asyncio
async def test_superseded_index_generation_keeps_graph_pending_and_cache_is_retired(
    pipeline, monkeypatch
):
    from basic_memory.index.local_dependencies import LocalMarkdownFileIndexer
    from basic_memory.read_cache import ReadCacheChange, ReadCacheInvalidationStatus

    class Cache:
        def __init__(self):
            self.invalidated = []

        async def invalidate(self, project_id: str, change: ReadCacheChange):
            self.invalidated.append(project_id)
            return ReadCacheInvalidationStatus.invalidated

        async def invalidate_project(self, project_id: str):
            self.invalidated.append(project_id)
            return ReadCacheInvalidationStatus.invalidated

    publish(pipeline)
    index = LocalMarkdownFileIndexer.index_file
    cache = Cache()
    pipeline = replace(pipeline, read_cache=cache)

    async def superseded(self, file_path, *, source):
        return replace(await index(self, file_path, source=source), content_superseded=True)

    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", superseded)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        async with asyncio.timeout(10):
            while not next(
                (row.last_error for row in (await worker.status()).stages if row.stage == "graph"),
                None,
            ):
                await asyncio.sleep(0.02)
        assert not (await worker.status()).caught_up
    assert cache.invalidated and set(cache.invalidated) == {pipeline.project_external_id}
    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", index)
    restarted = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with restarted.run():
        await wait_ready(restarted)


@pytest.mark.asyncio
async def test_vector_configuration_change_retries_only_vector_checkpoint(pipeline):
    publish(pipeline)
    worker = GitProjectionWorker(pipeline, poll_interval=0.02)
    async with worker.run():
        await wait_ready(worker)
    before = {row.stage: row for row in (await worker.status()).stages}
    changed = replace(pipeline, vector_key=pipeline.vector_key + ":changed-policy")
    restarted = GitProjectionWorker(changed, poll_interval=0.02)
    async with restarted.run():
        await wait_ready(restarted)
        # Wait for the new owner to discover the changed policy even when the old
        # checkpoint was already complete at context entry.
        async with asyncio.timeout(10):
            while (
                next(
                    row for row in (await restarted.status()).stages if row.stage == "vectors"
                ).projection_key
                != changed.vector_key
            ):
                await asyncio.sleep(0.02)
        await wait_ready(restarted)
    after = {row.stage: row for row in (await restarted.status()).stages}
    assert all(
        after[stage].attempts == before[stage].attempts for stage in ("content", "files", "graph")
    )
    assert after["vectors"].projection_key == changed.vector_key
