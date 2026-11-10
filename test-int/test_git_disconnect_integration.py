"""A client that disconnects during a Git write cannot cut the write in half (B4).

When an MCP client closes its connection, MCP cancels the running tool call through an
anyio cancel scope (`watch_disconnect` in mcp's streamable HTTP transport). These tests
cancel real API writes the same way, at each point of one write:

- while the runner accepts the mutation into the session (before the candidate commit),
- after the candidate commit, before its publication,
- inside the publication, before and after the ref update,
- after the write returned, before the answer reached the client.

Whatever the point, the write afterwards is applied completely or not at all: Git, SQL
and (once the files stage ran) the disk agree when the cancelled call ends, no private
candidate ref is left, and the database connection went back to the pool. The
projection worker is paused around the cancel, so SQL must agree without its help.
"""

import asyncio
import gc
import threading
import warnings
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.pool import AsyncAdaptedQueuePool

import basic_memory.services.git_project_runtime as runtime_module
from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.index.git_projection_worker import GitProjectionWorker
from basic_memory.models import Entity, NoteContent
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.services.git_project_runtime import GitProjectRuntime

from tests.git_host_support import HostSetup, host_setup as host_setup, ready

RUNNERS = (
    "run_accepted_note_create",
    "run_accepted_note_update",
    "run_accepted_note_edit",
    "run_accepted_note_move",
    "run_accepted_note_delete",
)
POINTS = ("runner", "after_prepare", "before_publish", "after_publish", "after_answer")
OPERATIONS = ("write", "edit", "move", "delete")


def note_text(title: str) -> str:
    return (
        f"---\ntitle: {title}\ntype: note\npermalink: test-project/notes/{title.lower()}\n---\n\n"
        f"Body of {title}\n"
    )


def seed(repository: GitProjectRepository, titles: list[str]) -> dict[str, UUID]:
    base = repository.snapshot()
    files = {f"notes/{title}.md": note_text(title) for title in titles}
    ids = {path: uuid4() for path in files}
    repository.publish(
        repository.prepare(
            transaction_id=uuid4(),
            base_commit=base.commit_id,
            files={**base.files, **files},
            note_ids={**base.manifest.note_ids, **ids},
        )
    )
    return {path.removeprefix("notes/").removesuffix(".md"): ids[path] for path in ids}


def candidate_refs(repository: GitProjectRepository) -> list[str]:
    refs = repository._git("for-each-ref", "--format=%(refname)", "refs/bm/candidates")
    return refs.decode().split()


async def sql_notes(setup: HostSetup) -> dict[str, str]:
    """file_path -> accepted markdown, as SQL holds it now."""
    _, factory = await db.get_or_create_db(setup.config.database_path)
    async with factory() as session:
        rows = await session.execute(
            select(Entity.file_path, NoteContent.markdown_content).join(
                NoteContent, NoteContent.entity_id == Entity.id
            )
        )
        return {path: text for path, text in rows}


def runtime(setup: HostSetup) -> GitProjectRuntime:
    host = resolve_container().git_host
    assert host is not None
    return host.runtime_for(setup.project.external_id)


@dataclass
class Fault:
    """One point inside a write where the test's disconnect lands."""

    point: str
    loop: asyncio.AbstractEventLoop
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: threading.Event = field(default_factory=threading.Event)
    armed: bool = True

    def take(self) -> bool:
        """The first write through the point meets the fault; later ones pass."""
        armed, self.armed = self.armed, False
        return armed

    def reach_in_thread(self) -> None:
        """Called from a Git thread: report the point, then hold the thread there."""
        if self.take():
            self.loop.call_soon_threadsafe(self.reached.set)
            assert self.release.wait(10)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fault = self
        match self.point:
            case "runner":
                for name in RUNNERS:
                    original = getattr(runtime_module, name)

                    async def runner(session, *, request, dependencies, original=original):
                        if fault.take():
                            fault.reached.set()
                            await asyncio.sleep(5)  # the disconnect lands here
                        return await original(session, request=request, dependencies=dependencies)

                    monkeypatch.setattr(runtime_module, name, runner)
            case "after_prepare":
                prepare = GitProjectRepository.prepare

                def prepared(self, **kwargs):
                    candidate = prepare(self, **kwargs)
                    fault.reach_in_thread()
                    return candidate

                monkeypatch.setattr(GitProjectRepository, "prepare", prepared)
            case "before_publish" | "after_publish":
                publish = GitProjectRepository.publish

                def published(self, prepared):
                    if fault.point == "before_publish":
                        fault.reach_in_thread()
                    publication = publish(self, prepared)
                    if fault.point == "after_publish":
                        fault.reach_in_thread()
                    return publication

                monkeypatch.setattr(GitProjectRepository, "publish", published)
            case "after_answer":
                accept = GitProjectRuntime.accept

                async def answered(self, request):
                    change = await accept(self, request)
                    if fault.take():
                        fault.reached.set()
                        await asyncio.sleep(5)  # the disconnect lands here
                    return change

                monkeypatch.setattr(GitProjectRuntime, "accept", answered)
            case _:
                raise AssertionError(self.point)


async def disconnect_at(
    fault: Fault, call: Callable[[], Coroutine[Any, Any, httpx.Response]]
) -> None:
    """Run `call` and cancel it at the fault point, as MCP does when the client leaves."""
    async with anyio.create_task_group() as group:

        async def request() -> None:
            await call()  # the answer is lost with the connection

        group.start_soon(request)
        await fault.reached.wait()
        # What watch_disconnect does on "http.disconnect": cancel the request's scope.
        group.cancel_scope.cancel()
        # A held Git thread goes on only after the cancellation was delivered.
        fault.loop.call_later(0.2, fault.release.set)


@pytest.fixture
def paused_worker(monkeypatch) -> Iterator[asyncio.Event]:
    """Projection passes wait while the event is clear: SQL may not lean on the worker."""
    resume = asyncio.Event()
    resume.set()
    passes = GitProjectionWorker._pass

    async def gated(self, *, vectors):
        await resume.wait()
        return await passes(self, vectors=vectors)

    monkeypatch.setattr(GitProjectionWorker, "_pass", gated)
    yield resume


def leaked_connection_warnings() -> list[str]:
    """Collect garbage now; SQLAlchemy warns for each pooled connection never checked in."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gc.collect()
    return [str(w.message) for w in caught if "non-checked-in" in str(w.message)]


def request_for(
    client: AsyncClient, project_id: str, operation: str, note_id: UUID
) -> Callable[[], Coroutine[Any, Any, httpx.Response]]:
    base = f"/v2/projects/{project_id}/knowledge"
    match operation:
        case "write":
            body: dict[str, Any] = {
                "note": {"title": "Fresh", "directory": "notes", "content": "Fresh body"}
            }
            return lambda: client.post(base + "/write", json=body)
        case "edit":
            body = {"operation": "append", "content": "\n- [fact] Disconnectprobe\n"}
            return lambda: client.patch(base + f"/entities/{note_id}", json=body)
        case "move":
            body = {"destination_path": "moved/Target.md"}
            return lambda: client.put(base + f"/entities/{note_id}/move", json=body)
        case "delete":
            return lambda: client.delete(base + f"/entities/{note_id}")
        case _:
            raise AssertionError(operation)


def applied(operation: str, files: dict[str, str]) -> bool:
    """Whether Git's head holds the operation's change."""
    match operation:
        case "write":
            return "notes/Fresh.md" in files
        case "edit":
            return "Disconnectprobe" in files["notes/Target.md"]
        case "move":
            return "moved/Target.md" in files and "notes/Target.md" not in files
        case "delete":
            return "notes/Target.md" not in files
        case _:
            raise AssertionError(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_a_disconnect_leaves_the_write_whole_or_absent(
    host_setup, monkeypatch, paused_worker, operation, point
):
    from basic_memory.api.app import app

    project_id = host_setup.project.external_id
    repository = host_setup.repository
    ids = seed(repository, ["Target", "Other"])
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await ready(client, project_id)
            head = repository.head()
            refs = candidate_refs(repository)
            fault = Fault(point, asyncio.get_running_loop())
            fault.install(monkeypatch)
            paused_worker.clear()

            await disconnect_at(fault, request_for(client, project_id, operation, ids["Target"]))

            # --- Right when the cancelled call ends, before the worker may help ---
            files = repository.snapshot().files
            # Cancelled while the runner still accepted the mutation: nothing happened.
            # From the candidate commit on, the write is one step and completes.
            assert applied(operation, files) == (point != "runner")
            assert (repository.head() != head) == (point != "runner")
            assert await sql_notes(host_setup) == files
            assert runtime(host_setup)._sql_commit == repository.head()
            assert candidate_refs(repository) == refs
            assert not runtime(host_setup).gate.locked()

            # --- The worker catches up the disk; the project takes the next write ---
            paused_worker.set()
            await ready(client, project_id)
            root = Path(host_setup.project.path)
            on_disk = {
                path.relative_to(root).as_posix(): path.read_text() for path in root.rglob("*.md")
            }
            assert on_disk == repository.snapshot().files
            response = await client.patch(
                f"/v2/projects/{project_id}/knowledge/entities/{ids['Other']}",
                json={"operation": "append", "content": "\n- [fact] Nextprobe\n"},
            )
            assert response.status_code == 202, response.text
            assert "Nextprobe" in repository.snapshot().files["notes/Other.md"]
            engine, _ = await db.get_or_create_db(host_setup.config.database_path)
            await ready(client, project_id)
            assert isinstance(engine.pool, AsyncAdaptedQueuePool)
            assert engine.pool.checkedout() == 0
    assert leaked_connection_warnings() == []


@pytest.mark.asyncio
async def test_repeated_native_cancellation_after_the_candidate_commit_still_completes(
    host_setup, monkeypatch, paused_worker
):
    """A plain task.cancel() (twice) passes an anyio shield; the write must still finish."""
    from basic_memory.api.app import app

    project_id = host_setup.project.external_id
    repository = host_setup.repository
    ids = seed(repository, ["Target"])
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await ready(client, project_id)
            refs = candidate_refs(repository)
            fault = Fault("after_prepare", asyncio.get_running_loop())
            fault.install(monkeypatch)
            paused_worker.clear()
            task = asyncio.create_task(request_for(client, project_id, "edit", ids["Target"])())
            await fault.reached.wait()
            task.cancel()
            await asyncio.sleep(0.02)
            task.cancel()
            await asyncio.sleep(0.02)
            # The write still holds the project: no other write may slip in between.
            assert not task.done() and runtime(host_setup).gate.locked()
            fault.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            files = repository.snapshot().files
            assert "Disconnectprobe" in files["notes/Target.md"]
            assert await sql_notes(host_setup) == files
            assert candidate_refs(repository) == refs
            paused_worker.set()
            await ready(client, project_id)
