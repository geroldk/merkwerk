"""settle_atomic: one bounded step runs to its outcome past a cancelled caller (B4).

MCP cancels a call whose client disconnects through an anyio cancel scope, which
delivers its cancellation again on every loop tick; a stopping server or a test may
also call task.cancel(), which passes an anyio shield. Both must wait for the step.
"""

import asyncio
import threading
import time

import anyio
import pytest
from loguru import logger

from basic_memory.repository.git_project_repository import settle_atomic, settle_git_operation
from basic_memory.runtime.call_publications import recording_publications


@pytest.mark.asyncio
async def test_a_scope_cancellation_waits_for_the_step_then_reaches_the_caller():
    started, finished = asyncio.Event(), asyncio.Event()
    after_step: list[str] = []

    async def step() -> str:
        started.set()
        await asyncio.sleep(0.2)
        finished.set()
        return "done"

    async def caller() -> None:
        with recording_publications() as record:
            try:
                await settle_atomic(step())
            finally:
                after_step.append(f"finished={finished.is_set()} held={record.held}")

    async with anyio.create_task_group() as group:
        group.start_soon(caller)
        await started.wait()
        group.cancel_scope.cancel()
    # The caller saw its cancellation only once the step had finished.
    assert after_step == ["finished=True held=1"]


@pytest.mark.asyncio
async def test_repeated_native_cancellation_waits_for_the_step():
    release = asyncio.Event()

    async def step() -> str:
        await release.wait()
        return "done"

    task = asyncio.create_task(settle_atomic(step()))
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_an_uncancelled_step_returns_its_result_or_raises_its_error():
    async def value() -> int:
        return 7

    async def failure() -> int:
        raise ValueError("refused")

    with recording_publications() as record:
        assert await settle_atomic(value()) == 7
        with pytest.raises(ValueError, match="refused"):
            await settle_atomic(failure())
    assert record.held == 0


@pytest.mark.asyncio
async def test_a_step_that_fails_while_its_caller_is_cancelled_is_logged():
    """The cancellation replaces the step's error; nobody else would ever report it."""
    started = asyncio.Event()
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")

    async def step() -> None:
        started.set()
        await asyncio.sleep(0.1)
        raise RuntimeError("Git published; SQL pending. Check status before retry.")

    async def caller() -> None:
        await settle_atomic(step())

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(caller)
            await started.wait()
            group.cancel_scope.cancel()
    finally:
        logger.remove(sink)
    assert lines == [
        "Atomic Git step failed after its caller was cancelled: "
        "RuntimeError: Git published; SQL pending. Check status before retry."
    ]


@pytest.mark.asyncio
async def test_a_cancelled_caller_does_not_spin_while_a_git_thread_finishes():
    """The wait sleeps: before B4, an anyio cancellation made it a busy loop."""
    release = threading.Event()

    def blocked_git_call() -> str:
        assert release.wait(5)
        return "published"

    async def caller() -> None:
        await settle_git_operation(blocked_git_call)

    async with anyio.create_task_group() as group:
        group.start_soon(caller)
        await asyncio.sleep(0.05)
        group.cancel_scope.cancel()
        # Measured outside the cancelled scope: this test's own task keeps running.
        with anyio.CancelScope(shield=True):
            cpu = time.process_time()
            await asyncio.sleep(0.5)
            spent = time.process_time() - cpu
            release.set()
    # Half a second of a sleeping event loop costs far less than 0.25 s of CPU; a loop
    # that is woken on every tick costs nearly all of it.
    assert spent < 0.25, f"the cancelled wait used {spent:.2f} s of CPU in 0.5 s"
