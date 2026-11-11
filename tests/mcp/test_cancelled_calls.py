"""A tool call cancelled before it answers is logged with what it published (B4)."""

import asyncio
from collections.abc import Iterator

import anyio
import mcp.types as mt
import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools import ToolResult
from loguru import logger

from basic_memory.mcp.cancelled_calls import CancelledCallMiddleware
from basic_memory.runtime.call_publications import note_held_cancellation, note_publication


@pytest.fixture
def warnings_logged() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")
    yield lines
    logger.remove(sink)


def _context(name: str, **arguments: str) -> MiddlewareContext[mt.CallToolRequestParams]:
    return MiddlewareContext(
        message=mt.CallToolRequestParams(name=name, arguments=arguments), method="tools/call"
    )


@pytest.mark.asyncio
async def test_a_cancelled_write_that_published_is_logged_as_completed(warnings_logged):
    async def published_then_cancelled(context) -> ToolResult:
        # The publication point is passed in a Git thread, as in the real write.
        await asyncio.to_thread(note_publication, "main.git", "3f2a9c1d0b7e" + "0" * 28)
        note_held_cancellation()
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await CancelledCallMiddleware().on_call_tool(
            _context("bm_move", project="main", destination_project="archive"),
            published_then_cancelled,
        )

    assert warnings_logged == [
        "MCP tool call cancelled before it answered (client gone or server stopping): "
        "tool=bm_move project=main destination_project=archive outcome=completed "
        "published=main.git@3f2a9c1d0b7e (the change is in effect; check before retrying); "
        "the cancel waited for 1 atomic step(s)"
    ]


@pytest.mark.asyncio
async def test_a_call_cancelled_before_publishing_says_nothing_was_published(warnings_logged):
    async def cancelled(context) -> ToolResult:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await CancelledCallMiddleware().on_call_tool(_context("bm_edit", project="main"), cancelled)

    assert warnings_logged == [
        "MCP tool call cancelled before it answered (client gone or server stopping): "
        "tool=bm_edit project=main outcome=not-published (no Git commit was made by this call)"
    ]


@pytest.mark.asyncio
async def test_an_answered_call_logs_nothing(warnings_logged):
    async def answered(context) -> ToolResult:
        note_publication("main.git", "a" * 40)
        return ToolResult(content=[mt.TextContent(type="text", text="ok")])

    result = await CancelledCallMiddleware().on_call_tool(_context("bm_write"), answered)

    assert result.content[0].text == "ok"  # ty: ignore[unresolved-attribute]
    assert warnings_logged == []


@pytest.mark.asyncio
async def test_a_call_cancelled_through_its_request_scope_reaches_the_middleware(
    warnings_logged,
):
    """The real path: FastMCP runs the tool inside the request's cancel scope."""
    server = FastMCP("cancelled-calls")
    server.add_middleware(CancelledCallMiddleware())
    entered = asyncio.Event()

    @server.tool
    async def slow_write(project: str) -> str:
        note_publication(f"{project}.git", "b" * 40)
        entered.set()
        await asyncio.sleep(5)
        return "answered"

    async with Client(server) as client:
        async with anyio.create_task_group() as group:
            group.start_soon(client.call_tool, "slow_write", {"project": "main"})
            await entered.wait()
            group.cancel_scope.cancel()
        # The client's cancellation reaches the server as a cancel notification.
        async with asyncio.timeout(5):
            while not warnings_logged:
                await asyncio.sleep(0.01)

    assert warnings_logged == [
        "MCP tool call cancelled before it answered (client gone or server stopping): "
        "tool=slow_write project=main outcome=completed published=main.git@bbbbbbbbbbbb "
        "(the change is in effect; check before retrying)"
    ]
