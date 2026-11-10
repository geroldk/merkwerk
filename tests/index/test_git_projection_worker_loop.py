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
