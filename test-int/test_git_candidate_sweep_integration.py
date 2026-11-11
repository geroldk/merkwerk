"""A Git host's start drops the candidate refs nothing refers to (B4).

Before B4 every accepted write left its private candidate ref behind for good, one per
write; a crash between a candidate commit and its cleanup still can. The start sweeps
them, but never one an open move intent may need.
"""

import subprocess
import time
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

import basic_memory.services.git_project_runtime as runtime_module
from basic_memory.repository.git_move_intent_repository import MOVES_DIRECTORY
from basic_memory.repository.git_project_repository import GitProjectRepository

from tests.git_host_support import host_setup as host_setup, ready


def leftover_candidates(repository: GitProjectRepository, count: int) -> None:
    """What `count` accepted writes left before B4: a published commit, a ref each."""
    for number in range(count):
        base = repository.snapshot()
        path = f"notes/Left {number}.md"
        text = (
            f"---\ntitle: Left {number}\ntype: note\npermalink: test-project/notes/left-{number}"
            f"\n---\n\nLeft {number}\n"
        )
        repository.publish(
            repository.prepare(
                transaction_id=uuid4(),
                base_commit=base.commit_id,
                files={**base.files, path: text},
                note_ids={**base.manifest.note_ids, path: uuid4()},
            )
        )


def candidate_refs(repository: GitProjectRepository) -> list[bytes]:
    return repository._git("for-each-ref", "--format=%(refname)", "refs/bm/candidates").split()


@pytest.mark.asyncio
async def test_the_start_sweeps_published_candidates_and_writes_leave_none(host_setup):
    from basic_memory.api.app import app

    repository = host_setup.repository
    project_id = host_setup.project.external_id
    leftover_candidates(repository, 3)
    assert len(candidate_refs(repository)) == 3
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            assert candidate_refs(repository) == []
            await ready(client, project_id)
            response = await client.post(
                f"/v2/projects/{project_id}/knowledge/write",
                json={"note": {"title": "New", "directory": "notes", "content": "New"}},
            )
            assert response.status_code == 200, response.text
            assert "notes/New.md" in repository.snapshot().files
            assert candidate_refs(repository) == []


class RecordedLogger:
    """Stands in for the runtime's logger: the app's start replaces every loguru sink."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(message.format(*args, **kwargs))

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        pass


@pytest.mark.asyncio
async def test_an_unreadable_move_intent_keeps_every_candidate(host_setup, monkeypatch):
    from basic_memory.api.app import app

    repository = host_setup.repository
    leftover_candidates(repository, 2)
    intents = repository.path / MOVES_DIRECTORY
    intents.mkdir(mode=0o700)
    (intents / f"{uuid4()}.json").write_text("{ not an intent")
    recorded = RecordedLogger()
    monkeypatch.setattr(runtime_module, "logger", recorded)
    async with app.router.lifespan_context(app):
        assert len(candidate_refs(repository)) == 2
    assert (
        f"Git candidate sweep skipped in {repository.path.name}: 1 unreadable move intent file(s)"
        in recorded.warnings
    )


@pytest.mark.asyncio
async def test_a_sweep_whose_git_call_times_out_does_not_stop_the_start(host_setup, monkeypatch):
    """A slow Git (many old refs, a busy disk) must not turn the start into a restart loop."""
    from basic_memory.api.app import app

    repository = host_setup.repository
    project_id = host_setup.project.external_id
    leftover_candidates(repository, 1)

    def timed_out(self: GitProjectRepository) -> None:
        # What subprocess.run(timeout=30) raises for `git for-each-ref --merged=...`.
        raise subprocess.TimeoutExpired(["git", "for-each-ref"], 30)

    monkeypatch.setattr(GitProjectRepository, "candidate_refs", timed_out)
    recorded = RecordedLogger()
    monkeypatch.setattr(runtime_module, "logger", recorded)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, project_id)
    # The ref stays for the next start; the start went on.
    assert len(candidate_refs(repository)) == 1
    assert recorded.warnings == [
        f"Git candidate sweep failed in {repository.path.name}: "
        "Command '['git', 'for-each-ref']' timed out after 30 seconds"
    ]


@pytest.mark.asyncio
async def test_the_sweep_of_a_thousand_refs_keeps_the_start_short(host_setup):
    """Production hosts carry one ref per write since Git hosting began."""
    from basic_memory.api.app import app

    repository = host_setup.repository
    base = repository.head()
    # A thousand record-less candidate refs on the published head: one update-ref call.
    commands = "".join(f"update refs/bm/candidates/{uuid4()}/{base} {base}\n" for _ in range(1000))
    repository._git("update-ref", "--stdin", input_bytes=commands.encode())
    assert len(candidate_refs(repository)) == 1000
    started = time.monotonic()
    async with app.router.lifespan_context(app):
        elapsed = time.monotonic() - started
        assert candidate_refs(repository) == []
    # The whole start, migrations and projection included; the sweep is two Git listings
    # and one batched delete.
    assert elapsed < 30, elapsed
