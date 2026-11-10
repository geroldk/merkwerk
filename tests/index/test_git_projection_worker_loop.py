"""A superseded pass is not a failure: no failure count, no backoff (F2)."""

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from basic_memory.index import git_projection_worker as worker_module
from basic_memory.index.git_projection_worker import GitPassOutcome, GitProjectionWorker


async def waits_after(
    outcomes: Sequence[GitPassOutcome | Exception], *, vectors: bool, monkeypatch
) -> tuple[list[float], int]:
    """Run the loop over `outcomes`; return each wait's timeout and the bare yields."""
    worker = GitProjectionWorker(
        SimpleNamespace(project_id=1, session_maker=None),  # ty: ignore[invalid-argument-type]
        poll_interval=1,
    )
    pending = list(outcomes)
    waits: list[float] = []
    yields = 0

    async def scripted_pass(*, vectors: bool) -> GitPassOutcome:
        if not pending:
            raise asyncio.CancelledError
        outcome = pending.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def recorded_wait(awaitable, timeout):
        awaitable.close()
        waits.append(timeout)
        raise TimeoutError

    async def recorded_sleep(delay):
        nonlocal yields
        assert delay == 0
        yields += 1

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_pass", scripted_pass)
        patch.setattr(
            worker_module, "asyncio", SimpleNamespace(wait_for=recorded_wait, sleep=recorded_sleep)
        )
        with pytest.raises(asyncio.CancelledError):
            await worker._loop(vectors=vectors)
    return waits, yields


@pytest.mark.asyncio
async def test_superseded_keeps_the_failure_count_and_does_not_wait(monkeypatch):
    failed, superseded, done = GitPassOutcome.failed, GitPassOutcome.superseded, GitPassOutcome.done
    # A failure doubles the wait; a superseded pass between two failures neither resets
    # the count nor waits, so the second failure waits four poll intervals.
    assert await waits_after(
        [failed, superseded, failed], vectors=False, monkeypatch=monkeypatch
    ) == (
        [2, 4],
        1,
    )
    assert await waits_after(
        [RuntimeError("disk"), superseded, superseded, done], vectors=False, monkeypatch=monkeypatch
    ) == ([2, 1], 2)


@pytest.mark.asyncio
async def test_superseded_vectors_wait_one_poll_for_the_index_lane(monkeypatch):
    """Vectors cannot re-target themselves; the index lane wakes them after the graph."""
    outcomes = [GitPassOutcome.failed, GitPassOutcome.superseded, GitPassOutcome.failed]
    assert await waits_after(outcomes, vectors=True, monkeypatch=monkeypatch) == ([2, 1, 4], 0)


@pytest.mark.asyncio
async def test_deferred_resets_failures_and_waits_the_capped_interval(monkeypatch):
    """Waiting for another project (I1) is not a failure: no backoff, a longer poll."""
    failed, deferred = GitPassOutcome.failed, GitPassOutcome.deferred
    # poll_interval=1: a deferral waits 6 s (the owner's wake-up is the real trigger);
    # a failure after it starts counting from zero again.
    assert await waits_after(
        [failed, failed, deferred, deferred, failed], vectors=False, monkeypatch=monkeypatch
    ) == ([2, 4, 6, 6, 2], 0)


def test_deferred_interval_is_capped():
    def interval(poll: float) -> float:
        return GitProjectionWorker(
            SimpleNamespace(project_id=1, session_maker=None),  # ty: ignore[invalid-argument-type]
            poll_interval=poll,
        ).deferred_interval

    assert interval(5) == 30
    assert interval(20) == 60
    assert interval(90) == 90  # never shorter than the ordinary poll


def test_content_outcomes_drive_the_host_wake_ups():
    from basic_memory.repository.git_projection_repository import (
        GitProjectionDeferred,
        GitWaitingNote,
    )
    from uuid import uuid4

    calls: list[tuple[str, object]] = []
    identities = SimpleNamespace(
        wait_for=lambda project_id, owners: calls.append(("wait", (project_id, set(owners)))),
        settled=lambda project_id: calls.append(("settled", project_id)),
    )
    worker = GitProjectionWorker(
        SimpleNamespace(project_id=1, session_maker=None, identities=identities),  # ty: ignore[invalid-argument-type]
    )
    deferral = GitProjectionDeferred([GitWaitingNote(uuid4(), 2, uuid4(), "owner")])
    worker._content_deferred("c" * 40, deferral)
    assert (worker.deferral, worker.deferral_commit) == (deferral, "c" * 40)
    worker._content_settled()
    assert worker.deferral is None and worker.deferral_commit is None
    assert calls == [("wait", (1, {2})), ("settled", 1)]
    worker.wake_index()
    assert worker._wake_index.is_set() and not worker._wake_vectors.is_set()
