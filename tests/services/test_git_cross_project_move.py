"""The parts of the cross-project move that need no host (I1)."""

import asyncio

import anyio
import pytest

from basic_memory.services.git_cross_project_move import GitMoveRefused, _settled


@pytest.mark.asyncio
async def test_a_cancelled_caller_waits_until_the_move_has_settled():
    """The draft, the delete and the commit belong together; a cancel waits for them."""
    release, finished = asyncio.Event(), asyncio.Event()

    async def move() -> int:
        await release.wait()
        finished.set()
        return 7

    caller = asyncio.create_task(_settled(move()))
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0.02)
    assert not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert finished.is_set()
    assert await _settled(asyncio.sleep(0, result="done")) == "done"


@pytest.mark.asyncio
async def test_an_anyio_cancel_scope_waits_once_instead_of_spinning(monkeypatch):
    """anyio re-delivers a scope's cancellation on every tick; one shielded wait suffices."""
    waits = 0
    wait = asyncio.wait

    def counting(tasks):
        nonlocal waits
        waits += 1
        return wait(tasks)

    # settle_atomic waits inside an anyio shield (B4); a wait that each tick's
    # re-delivered cancellation interrupted would be entered again and again.
    monkeypatch.setattr(asyncio, "wait", counting)
    finished = asyncio.Event()

    async def move() -> None:
        await asyncio.sleep(0.3)
        finished.set()

    with anyio.CancelScope() as scope:
        asyncio.get_running_loop().call_later(0.05, scope.cancel)
        await _settled(move())
    assert scope.cancelled_caught
    assert finished.is_set()
    assert waits == 1


def test_a_refusal_carries_its_kind_and_details_into_the_body():
    refused = GitMoveRefused(409, "duplicate_title", "Taken. Nothing was moved.", title="A")
    assert refused.body() == {
        "message": "Taken. Nothing was moved.",
        "kind": "duplicate_title",
        "title": "A",
    }
