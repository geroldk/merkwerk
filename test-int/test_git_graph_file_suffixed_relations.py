"""A link stored unresolved by the old resolver, on a Git host after an install.

A note whose permalink ends in a file suffix (`<project>/threads/fix-in-tax-costs.py`)
and lives in another folder could not be named by `[[threads/fix-in-tax-costs.py]]`, so
the relation was stored with no target. The Git host has no project-wide relation pass:
the graph stage resolves the links of the notes it indexes.
"""

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from basic_memory.index.git_projection import build_git_projection_pipeline
from basic_memory.index.git_projection_worker import GitPassOutcome, GitProjectionWorker
from basic_memory.models import Entity, Relation
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.services import bulk_link_resolver

LINK = "threads/fix-in-tax-costs.py"
TARGET = "archive/threads/Fix in tax_costs.py.md"


@pytest.fixture
def pipeline(tmp_path_factory, test_project, engine_factory, app_config):
    repository = GitProjectRepository.create(
        tmp_path_factory.mktemp("git-suffix") / "project.git",
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


def note(title, permalink, body):
    return f"---\ntitle: {title}\ntype: note\npermalink: {permalink}\n---\n\n{body}\n"


async def link_targets(pipeline):
    async with pipeline.session_maker() as session:
        target_id = await session.scalar(select(Entity.id).where(Entity.file_path == TARGET))
        targets = (
            await session.scalars(select(Relation.to_id).where(Relation.to_name == LINK))
        ).all()
    return target_id, targets


async def project_with_the_old_resolver(pipeline, monkeypatch, files) -> int:
    """Project `files` as the old resolver did: no routed permalink after the file paths.

    This simulates the stored state by switching the new candidate list off for one pass;
    it is not a graph written by the old release.
    """
    with monkeypatch.context() as old_resolver:
        old_resolver.setattr(
            bulk_link_resolver,
            "build_permalink_candidates_after_file_paths",
            lambda *args, **kwargs: [],
            raising=False,
        )
        publish(pipeline, files)
        assert await GitProjectionWorker(pipeline)._pass(vectors=False) is GitPassOutcome.done
    target_id, targets = await link_targets(pipeline)
    assert target_id is not None and targets == [None]
    return target_id


@pytest.mark.asyncio
async def test_a_stored_unresolved_link_resolves_on_the_next_full_graph_pass(pipeline, monkeypatch):
    files = {
        "Source.md": note("Source", "test-project/source", f"- relates_to [[{LINK}]]"),
        TARGET: note("Fix", f"test-project/{LINK}", "The fix"),
        "Other.md": note("Other", "test-project/other", "Firstprobe"),
    }
    target_id = await project_with_the_old_resolver(pipeline, monkeypatch, files)

    # A restart re-indexes nothing, and a body-only change re-indexes only its own note.
    restarted = GitProjectionWorker(pipeline)
    assert await restarted._pass(vectors=False) is GitPassOutcome.done
    files["Other.md"] = files["Other.md"].replace("Firstprobe", "Secondprobe")
    publish(pipeline, files)
    assert await restarted._pass(vectors=False) is GitPassOutcome.done
    assert (await link_targets(pipeline))[1] == [None]

    # A new note changes the identities, so the graph stage indexes every note again.
    files["New.md"] = note("New", "test-project/new", "New note")
    publish(pipeline, files)
    assert await restarted._pass(vectors=False) is GitPassOutcome.done
    assert (await link_targets(pipeline))[1] == [target_id]


@pytest.mark.asyncio
async def test_editing_the_linking_note_resolves_its_stored_link(pipeline, monkeypatch):
    files = {
        "Source.md": note("Source", "test-project/source", f"- relates_to [[{LINK}]]"),
        TARGET: note("Fix", f"test-project/{LINK}", "The fix"),
    }
    target_id = await project_with_the_old_resolver(pipeline, monkeypatch, files)

    # A body-only change re-indexes just this note, which resolves its own links again.
    files["Source.md"] += "\nEdited after the install\n"
    publish(pipeline, files)
    worker = GitProjectionWorker(pipeline)
    assert await worker._pass(vectors=False) is GitPassOutcome.done
    assert (await link_targets(pipeline))[1] == [target_id]
