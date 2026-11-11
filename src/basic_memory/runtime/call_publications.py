"""What one MCP tool call published, for the line logged when the call is cancelled.

A client that closes its connection cancels the call it is waiting for (MCP closes the
request's cancel scope), and a stopping server cancels running calls the same way. The
client then knows only that the call "may have run". The server logs one line saying
what really happened, and that line needs to know whether the call reached a
publication point, the single Git ref update that makes a change take effect.

The MCP layer opens a record per tool call; the Git publication primitives add to it.
A ContextVar carries the record into the in-process API request, into threads
(`asyncio.to_thread` copies the context) and into settled steps (a new task copies
the context too). Those copies share the one mutable record, which is why it is an
object that is changed and not a value that is set again.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class CallPublications:
    """The publications of one tool call, in the order they happened."""

    # "<repository folder>@<commit>" per publication; a thread appends, so a lock guards it.
    commits: list[str] = field(default_factory=list)
    # Atomic steps that a cancellation reached while they ran; each finished first.
    held: int = 0
    # A call that changes several notes one after another, each on its own (a folder move
    # into another project): how many it set out to change, and how many reached their
    # outcome. None for a call that is one step.
    items: int | None = None
    items_done: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def published(self, repository: str, commit_id: str) -> None:
        with self._lock:
            self.commits.append(f"{repository}@{commit_id[:12]}")

    def held_cancellation(self) -> None:
        with self._lock:
            self.held += 1

    @property
    def cut_off(self) -> bool:
        """A call of several items was cancelled before its last item reached an outcome."""
        return self.items is not None and self.items_done < self.items


_current: ContextVar[CallPublications | None] = ContextVar("bm_call_publications", default=None)


@contextmanager
def recording_publications() -> Iterator[CallPublications]:
    """Open the record of one tool call; nested calls get their own."""
    record = CallPublications()
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)


def note_publication(repository: str, commit_id: str) -> None:
    """A publication point was passed: the change of the current call is in effect."""
    if (record := _current.get()) is not None:
        record.published(repository, commit_id)


def note_held_cancellation() -> None:
    """A cancellation waited for an atomic step of the current call to finish."""
    if (record := _current.get()) is not None:
        record.held_cancellation()


def note_items(count: int) -> None:
    """The current call changes `count` items one after another, each on its own."""
    if (record := _current.get()) is not None:
        record.items, record.items_done = count, 0


def note_item_done() -> None:
    """One more item of the current call reached its outcome (changed, or refused)."""
    if (record := _current.get()) is not None:
        record.items_done += 1
