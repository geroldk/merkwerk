"""Log one line for every tool call that is cancelled before it answers.

A client that closes its connection (a reconnect, a crash, an aborted agent) cancels
the calls it is still waiting for, and a stopping server cancels its running calls.
Without this the server logs nothing, and nobody can tell afterwards whether such a
call changed anything. The line names the tool, the project and what the call
published: on a Git host a write takes effect at its publication, so "published"
means the change is in effect although the client never saw the answer.
"""

import asyncio
from typing import Any, override

import mcp.types as mt
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from loguru import logger

from basic_memory.runtime.call_publications import CallPublications, recording_publications


class CancelledCallMiddleware(Middleware):
    """Record what a tool call publishes; log it when the call is cancelled."""

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        with recording_publications() as publications:
            try:
                return await call_next(context)
            except asyncio.CancelledError:
                arguments = context.message.arguments or {}
                logger.warning(
                    "MCP tool call cancelled before it answered (client gone or server "
                    "stopping): tool={} project={}{} {}",
                    context.message.name,
                    _argument(arguments, "project"),
                    _destination(arguments),
                    outcome(publications),
                )
                raise


def outcome(publications: CallPublications) -> str:
    """What the cancelled call did, in words a person checking the log can act on."""
    # Trigger: the cancellation arrived while an atomic step ran (held > 0).
    # Why: such a step finishes first, so the cancel came "before" its publication point
    #   in time but did not stop it; the reader must not take "cancelled" for "undone".
    # Outcome: the line says so next to what was published.
    held = (
        f"; the cancel waited for {publications.held} atomic step(s)" if publications.held else ""
    )
    if publications.commits:
        return (
            f"outcome=completed published={','.join(publications.commits)} "
            f"(the change is in effect; check before retrying){held}"
        )
    return f"outcome=not-published (no Git commit was made by this call){held}"


def _argument(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    return value if isinstance(value, str) and value else "-"


def _destination(arguments: dict[str, Any]) -> str:
    """A move into another project publishes there too; name it."""
    destination = arguments.get("destination_project")
    return f" destination_project={destination}" if isinstance(destination, str) else ""
