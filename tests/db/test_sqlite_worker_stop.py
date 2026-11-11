"""A SQLite connection whose close is cut by a cancellation still closes (B4 review, finding 1).

When a cancellation reaches SQLAlchemy while it terminates an invalidated aiosqlite
connection, SQLAlchemy stops the worker thread itself and lets the graceful close go
on in a shielded task. aiosqlite 0.22's close() then stops the thread a second time and
waits for an answer the ended thread never gives. A server stopped with writes in flight
hung on such a task until it was killed. These tests use the engine Basic Memory builds.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.util import greenlet_spawn

from basic_memory import db


@pytest.fixture
async def sqlite_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    engine = db._create_sqlite_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'stop.db'}", db.DatabaseType.FILESYSTEM
    )
    try:
        yield engine
    finally:
        await engine.dispose()


def graceful_closes() -> list[asyncio.Task[Any]]:
    return [
        task for task in asyncio.all_tasks() if "_terminate_graceful_close" in repr(task.get_coro())
    ]


@pytest.mark.asyncio
async def test_a_terminate_cut_by_a_cancellation_lets_its_graceful_close_finish(sqlite_engine):
    """The path of the stop that hung: terminate() in a greenlet, cancelled at once."""
    async with sqlite_engine.connect() as connection:
        raw = await connection.get_raw_connection()
        terminating = asyncio.create_task(greenlet_spawn(raw.dbapi_connection.terminate))
        await asyncio.sleep(0)  # terminate() now waits for its shielded graceful close
        terminating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await terminating
        closes = graceful_closes()
        assert len(closes) == 1
        done, pending = await asyncio.wait(closes, timeout=5)
        for task in pending:
            task.cancel()  # do not leave the hung task to the next test
        assert not pending, "the graceful close waits for a stop the ended thread never answers"
        await connection.invalidate()


@pytest.mark.asyncio
async def test_a_close_after_a_stop_returns(sqlite_engine):
    """SQLAlchemy's forced close, then aiosqlite's own close of the same connection."""
    async with sqlite_engine.connect() as connection:
        raw = await connection.get_raw_connection()
        driver = raw.driver_connection
        driver.stop()
        # The connection is gone already, so close() reports it; it must not hang.
        with contextlib.suppress(ValueError):
            await asyncio.wait_for(driver.close(), 5)
        # A third stop has nothing left to wait for.
        assert driver.stop() is None
        await connection.invalidate()
