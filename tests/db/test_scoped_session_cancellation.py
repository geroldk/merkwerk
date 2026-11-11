"""A session scope cancelled like a disconnected MCP call returns its connection (B4)."""

import asyncio
import gc
import warnings

import anyio
import pytest
from sqlalchemy import text

from basic_memory import db


@pytest.mark.asyncio
async def test_a_cancelled_scope_checks_the_connection_back_in(engine_factory):
    engine, session_maker = engine_factory
    entered = asyncio.Event()

    async def work() -> None:
        async with db.scoped_session(session_maker) as session:
            await session.execute(text("SELECT 1"))
            entered.set()
            await asyncio.sleep(5)  # the cancellation lands here

    async with anyio.create_task_group() as group:
        group.start_soon(work)
        await entered.wait()
        # What MCP's watch_disconnect does when the client closes the connection.
        group.cancel_scope.cancel()

    assert engine.pool.checkedout() == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gc.collect()
    assert not [w for w in caught if "non-checked-in" in str(w.message)]
    # The connection is usable again, not terminated.
    async with db.scoped_session(session_maker) as session:
        assert (await session.execute(text("SELECT 1"))).scalar() == 1
