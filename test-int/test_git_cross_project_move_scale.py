"""How long a folder takes to move into a large Git project, keeping every id (I1).

bm_move(path=..., destination_project=...) on a Git host moves note by note through the
server's move route (a draft in the target, the delete in the source, the commit), and
catches the target up once at the end. Measured with a target of 1000 notes, for folders
of 300 and 600 notes, after both projects have settled (graph and vectors done), so the
number is the move itself. Slow, and a benchmark: an ordinary run leaves it out.

    python -m pytest -s -m benchmark test-int/test_git_cross_project_move_scale.py
"""

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest
from fastmcp import Client

from basic_memory.api.container import resolve_container
from basic_memory.repository.git_project_repository import GitProjectRepository

from tests.git_host_support import add_git_project, host_setup as host_setup

TARGET_NOTES = 1000


def note(title: str, path: str) -> str:
    return (
        f"---\ntitle: {title}\ntype: note\npermalink: {path[:-3]}\n"
        "created: '2026-01-01T10:00:00+01:00'\nmodified: '2026-01-02T10:00:00+01:00'\n---\n\n"
        f"# {title}\n\n- [fact] Fakt zu {title}\n- relates_to [[{title} Partner]]\n"
    )


def seed(repository: GitProjectRepository, folder: str, count: int) -> dict[str, Any]:
    files = {
        f"{folder}/{number:04d}.md": note(f"{folder} {number:04d}", f"{folder}/{number:04d}.md")
        for number in range(count)
    }
    ids = {path: uuid4() for path in files}
    prepared = repository.prepare(
        transaction_id=uuid4(), base_commit=repository.head(), files=files, note_ids=ids
    )
    repository.publish(prepared)
    return ids


async def settled(*project_ids: str, timeout: float) -> None:
    """Every stage of every project caught up (content, files, graph, vectors)."""
    async with asyncio.timeout(timeout):
        for project_id in project_ids:
            host = resolve_container().git_host
            assert host is not None
            worker = host.runtime_for(project_id).worker
            while not (await worker.status()).caught_up:
                worker.notify()
                await asyncio.sleep(0.2)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.timeout(5400)
@pytest.mark.asyncio
@pytest.mark.parametrize("moving", [300, 600])
async def test_a_folder_moves_into_a_large_project_with_every_id_kept(
    host_setup, config_manager, moving
):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    source = host_setup.project.external_id
    second, target_repo = await add_git_project(host_setup, config_manager, "second")
    moved_ids = seed(host_setup.repository, "mappe", moving)
    seed(target_repo, "bestand", TARGET_NOTES)
    async with Client(mcp) as client:
        started = time.perf_counter()
        await settled(source, second.external_id, timeout=1800)
        settle = time.perf_counter() - started

        started = time.perf_counter()
        answer = await client.call_tool(
            "bm_move",
            {
                "path": "mappe",
                "destination": "neu",
                "project": "test-project",
                "destination_project": "second",
                "output_format": "json",
            },
        )
        call = time.perf_counter() - started
        started = time.perf_counter()
        await settled(source, second.external_id, timeout=1800)
        after = time.perf_counter() - started
    assert not answer.is_error, answer.content
    result = answer.data
    print(
        f"\nI1 scale: {moving} notes into {TARGET_NOTES}: bm_move {call:.1f} s "
        f"({call / moving:.2f} s per note), then every stage caught up after {after:.1f} s "
        f"more (initial settle {settle:.1f} s)"
    )
    # Every note is listed, with its id kept, and no response cap applies.
    assert result["total_files"] == moving and result["failed"] == [] and result["pending"] == []
    assert {entry["external_id"] for entry in result["moved_notes"]} == {
        str(note_id) for note_id in moved_ids.values()
    }
    assert all(
        entry["external_id"] == entry["previous_external_id"] for entry in result["moved_notes"]
    )
    held = target_repo.snapshot().manifest.note_ids
    assert {path: held[path] for path in held if path.startswith("neu/")} == {
        path.replace("mappe/", "neu/", 1): note_id for path, note_id in moved_ids.items()
    }
    assert host_setup.repository.snapshot().files == {}
