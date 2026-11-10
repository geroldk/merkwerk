"""Real graph/FTS convergence when unchanged notes are skipped between Git heads."""

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from basic_memory.index.git_projection import build_git_projection_pipeline
from basic_memory.index.git_projection_worker import GitPassOutcome, GitProjectionWorker
from basic_memory.index.local_dependencies import LocalMarkdownFileIndexer
from basic_memory.models import Entity, Observation, Relation
from basic_memory.repository.git_project_repository import GitProjectRepository


@pytest.fixture
def pipeline(tmp_path_factory, test_project, engine_factory, app_config):
    repository = GitProjectRepository.create(
        tmp_path_factory.mktemp("git-delta") / "project.git",
        project_external_id=UUID(test_project.external_id),
    )
    return build_git_projection_pipeline(
        repository, project=test_project, session_maker=engine_factory[1], app_config=app_config
    )


def publish(pipeline, files):
    repository = pipeline.repository
    base = repository.snapshot()
    return repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=base.commit_id,
            files=files,
            note_ids={path: base.manifest.note_ids.get(path, uuid4()) for path in files},
        )
    ).commit_id


async def initialize(pipeline):
    files = {
        "Source.md": "---\ntitle: Source\ntype: note\npermalink: test-project/source\n---\n\n- [fact] Firstprobe\n\n- relates_to [[Target]]\n",
        "Target.md": "---\ntitle: Target\ntype: note\npermalink: test-project/target\n---\n\nTarget\n",
    }
    publish(pipeline, files)
    worker = GitProjectionWorker(pipeline)
    assert await worker._pass(vectors=False) is GitPassOutcome.done
    assert await worker._pass(vectors=True) is GitPassOutcome.done
    assert (await worker.status()).caught_up
    return worker, files


@pytest.mark.asyncio
async def test_body_only_update_indexes_one_note_and_preserves_untouched_graph(
    pipeline, monkeypatch
):
    worker, files = await initialize(pipeline)
    calls = []
    index = LocalMarkdownFileIndexer.index_file

    async def record(self, path, *, source):
        calls.append(path)
        return await index(self, path, source=source)

    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", record)
    files["Source.md"] = files["Source.md"].replace("Firstprobe", "Deltaprobe")
    commit = publish(pipeline, files)
    assert await worker._pass(vectors=False) is GitPassOutcome.done
    assert calls == ["Source.md"]
    assert await worker._pass(vectors=True) is GitPassOutcome.done
    assert (await worker.status()).caught_up
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(Observation.content)) == "Deltaprobe"
        relation = await session.scalar(select(Relation))
        assert relation is not None and relation.to_id is not None
        source_id = relation.from_id
    rows = await pipeline.search_service.repository.get_entity_search_rows(source_id)
    assert any("Deltaprobe" in row.content_stems for row in rows if row.type == "observation")
    assert all(
        row.embedding_status == "ready"
        for row in await pipeline.search_service.repository.get_entity_chunk_manifest(source_id)
    )
    restarted = GitProjectionWorker(pipeline)
    assert await restarted._pass(vectors=False) is GitPassOutcome.done
    assert calls == ["Source.md"]
    assert (
        next(
            row for row in (await restarted.status()).stages if row.stage == "graph"
        ).reusable_commit
        == commit
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["error", "process_death"])
async def test_partial_intermediate_graph_is_repaired_even_when_latest_reverts_its_body(
    pipeline, monkeypatch, engine_factory, app_config, tmp_path, fault
):
    worker, first = await initialize(pipeline)
    files = {**first, "Source.md": first["Source.md"].replace("Firstprobe", "Partialprobe")}
    publish(pipeline, files)
    index = LocalMarkdownFileIndexer.index_file

    async def fail_after_source(self, path, *, source):
        await index(self, path, source=source)
        raise RuntimeError("Lost graph acknowledgement after physical writes")

    if fault == "error":
        monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", fail_after_source)
        assert await worker._pass(vectors=False) is GitPassOutcome.failed
    else:
        config_path = tmp_path / "worker-config.json"
        config_path.write_text(app_config.model_dump_json())
        child = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "test-int/fixtures/git_projection_worker_crash.py",
                str(engine_factory[0].url.database),
                str(pipeline.repository.path),
                str(pipeline.project_id),
                str(config_path),
                "graph",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
            },
        )
        assert child.returncode == 73, child.stderr
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(Observation.content)) == "Partialprobe"
    files = {**first, "Target.md": first["Target.md"] + "Later target body\n"}
    publish(pipeline, files)
    calls = []

    async def record(self, path, *, source):
        calls.append(path)
        return await index(self, path, source=source)

    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", record)
    restarted = GitProjectionWorker(pipeline)
    assert await restarted._pass(vectors=False) is GitPassOutcome.done
    assert calls == ["Source.md", "Target.md"]
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(Observation.content)) == "Firstprobe"
    assert await restarted._pass(vectors=True) is GitPassOutcome.done
    assert (await restarted.status()).caught_up


@pytest.mark.asyncio
async def test_delete_between_graph_runs_invalidates_reuse_before_the_target_is_recreated(
    pipeline, monkeypatch
):
    worker, first = await initialize(pipeline)
    original = pipeline.repository.snapshot()
    deleted = publish(pipeline, {"Source.md": first["Source.md"]})
    # Crash/restart may happen after SQL deletion but before any graph attempt.
    await pipeline.apply("content", deleted)
    async with pipeline.session_maker() as session:
        assert await session.scalar(select(Relation.to_id)) is None
    assert (
        next(row for row in (await worker.status()).stages if row.stage == "graph").reusable_commit
        is None
    )
    pipeline.repository.publish(
        pipeline.repository.prepare(
            transaction_id=uuid4(),
            base_commit=deleted,
            files=first,
            note_ids=original.manifest.note_ids,
        )
    )
    calls = []
    index = LocalMarkdownFileIndexer.index_file

    async def record(self, path, *, source):
        calls.append(path)
        return await index(self, path, source=source)

    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", record)
    assert await worker._pass(vectors=False) is GitPassOutcome.done
    assert calls == ["Source.md", "Target.md"]
    async with pipeline.session_maker() as session:
        target_id = await session.scalar(select(Entity.id).where(Entity.title == "Target"))
        assert target_id is not None
        assert await session.scalar(select(Relation.to_id)) == target_id
    assert await worker._pass(vectors=True) is GitPassOutcome.done
    assert (await worker.status()).caught_up


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["title", "policy"])
async def test_identity_or_graph_policy_change_indexes_unchanged_sources(
    pipeline, monkeypatch, change
):
    worker, files = await initialize(pipeline)
    calls = []
    index = LocalMarkdownFileIndexer.index_file

    async def record(self, path, *, source):
        calls.append(path)
        return await index(self, path, source=source)

    monkeypatch.setattr(LocalMarkdownFileIndexer, "index_file", record)
    if change == "title":
        files["Target.md"] = files["Target.md"].replace("title: Target", "title: Renamed")
        publish(pipeline, files)
    else:
        pipeline = replace(pipeline, graph_key=pipeline.graph_key + ":new-graph-policy")
        worker = GitProjectionWorker(pipeline)
        assert not (await worker.status()).caught_up
    assert await worker._pass(vectors=False) is GitPassOutcome.done
    assert calls == ["Source.md", "Target.md"]
    assert not (await worker.status()).caught_up
    assert await worker._pass(vectors=True) is GitPassOutcome.done
    assert (await worker.status()).caught_up
