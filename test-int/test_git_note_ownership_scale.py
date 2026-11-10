"""How long the ownership rule takes for a folder that arrives by 600 single moves (I1).

On a peer that syncs the target first, every moved note is still in the source's head,
so each one needs its move proof: the commit that introduced it and its trailers. This
runs inside the sync guard's admission, so it is measured on a history of the size a
cross-project folder move leaves (1000 notes, then 600 move commits).
"""

import json
import subprocess
import time
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from basic_memory.models import Entity, Project
from basic_memory.repository.git_host_identities import GitHostIdentities, GitHostProject
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.schemas.git_project import GIT_PROJECT_MANIFEST_PATH, GitProjectManifest

NOTES, MOVES = 1000, 600


def manifest(project: UUID, notes: dict[str, UUID]) -> bytes:
    return json.dumps(
        {
            "format_version": 1,
            "project_external_id": str(project),
            "notes": [
                {"external_id": str(note), "path": path} for path, note in sorted(notes.items())
            ],
        },
        sort_keys=True,
        indent=2,
    ).encode()


def fast_import(repository: GitProjectRepository, commits: list[tuple[str, dict[str, bytes]]]):
    """Append commits to main in one `git fast-import` (prepare() would hash every note)."""
    stream = bytearray()
    for number, (message, changes) in enumerate(commits):
        stream += b"commit refs/heads/main\n"
        stream += f"committer BM <bm@localhost> {1_700_000_000 + number} +0000\n".encode()
        body = message.encode()
        stream += f"data {len(body)}\n".encode() + body + b"\n"
        if number == 0:
            stream += b"from refs/heads/main^0\n"
        for path, content in changes.items():
            stream += f"M 100644 inline {path}\ndata {len(content)}\n".encode() + content + b"\n"
        stream += b"\n"
    subprocess.run(
        ["git", "--git-dir", str(repository.path), "fast-import", "--quiet"],
        input=bytes(stream),
        check=True,
        capture_output=True,
        timeout=300,
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_600_moved_notes_are_classified_in_bounded_time(
    engine_factory, test_project, tmp_path
):
    _, session_maker = engine_factory
    async with session_maker() as session, session.begin():
        target = Project(name="target", path=str(tmp_path / "target"), is_active=True)
        session.add(target)
        await session.flush()
        target_id, target_uuid = target.id, UUID(target.external_id)
    source_uuid = UUID(test_project.external_id)
    source = GitProjectRepository.create(tmp_path / "source.git", project_external_id=source_uuid)
    target_repo = GitProjectRepository.create(
        tmp_path / "target.git", project_external_id=target_uuid
    )
    moved = {f"moved/m{index}.md": uuid4() for index in range(MOVES)}
    note = b"---\ntitle: n\ntype: note\npermalink: n\n---\n\nBody\n"
    # The source still holds every moved note: its sync has not happened yet.
    fast_import(
        source,
        [
            (
                "Seed\n",
                {
                    **{path: note for path in moved},
                    GIT_PROJECT_MANIFEST_PATH: manifest(source_uuid, moved),
                },
            )
        ],
    )
    notes = {f"notes/n{index}.md": uuid4() for index in range(NOTES)}
    commits = [
        (
            "Seed\n",
            {
                **{path: note for path in notes},
                GIT_PROJECT_MANIFEST_PATH: manifest(target_uuid, notes),
            },
        )
    ]
    held = dict(notes)
    for path, external_id in moved.items():
        held[path] = external_id
        commits.append(
            (
                f"Move {path}\n\nBasic-Memory-Transaction: {uuid4()}\n"
                f"Basic-Memory-Moved-Note: {external_id} from {source_uuid}\n",
                {path: note, GIT_PROJECT_MANIFEST_PATH: manifest(target_uuid, held)},
            )
        )
    fast_import(target_repo, commits)
    async with session_maker() as session, session.begin():
        for path, external_id in moved.items():
            session.add(
                Entity(
                    project_id=test_project.id,
                    external_id=str(external_id),
                    title=path,
                    note_type="note",
                    content_type="text/markdown",
                    file_path=path,
                )
            )
    identities = GitHostIdentities(
        [
            GitHostProject(test_project.id, source_uuid, test_project.name, source),
            GitHostProject(target_id, target_uuid, "target", target_repo),
        ]
    )
    head = target_repo.head()
    claimed = GitProjectManifest.model_validate_json(
        target_repo._git("cat-file", "blob", f"{head}:{GIT_PROJECT_MANIFEST_PATH}")
    )
    async with session_maker() as session:
        started = time.monotonic()
        foreign = await identities.classify_foreign(
            session, project_id=target_id, manifest=claimed, history=(target_repo, head)
        )
        first = time.monotonic() - started
        started = time.monotonic()
        await identities.classify_foreign(
            session, project_id=target_id, manifest=claimed, history=(target_repo, head)
        )
        again = time.monotonic() - started
        assert await session.scalar(select(Entity.id).limit(1)) is not None
    assert len(foreign.deferred) == MOVES and not foreign.duplicates
    print(f"classified {MOVES} moved notes: first {first:.1f} s, cached {again:.1f} s")
    # Measured 0.8 s and 0.0 s (2026-11-10); a per-note walk of the history took 86 s.
    assert first < 20
    assert again < 5
