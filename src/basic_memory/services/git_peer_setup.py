"""Point Git projects at a peer, by project name, on a stopped host (BM-009).

`git_sync_remotes` and `git_sync_merge` are keyed by project UUID and read only when the
service starts. This tool takes the names, finds the UUIDs, creates the bare repositories
of a local peer when asked, checks the result against the host's start rules and replaces
the configuration atomically. Which projects have a peer is configuration, not code: a
project that is not named here keeps whatever it had, and a project may have none.
"""

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from sqlalchemy import select

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, ConfigManager
from basic_memory.config_git import GitSyncRemote
from basic_memory.index.git_host import require_git_peer_settings
from basic_memory.models import Project
from basic_memory.services.git_project_create import (
    GitProjectCreateError,
    _overlaps,
    _write_config_atomically,
    cancel_on_termination,
    require_host_stopped,
)

type MergeMode = Literal["auto", "off"]


@dataclass(frozen=True, slots=True)
class PeerChange:
    name: str
    project_id: UUID
    url: str | None  # None: the peer was removed
    merge: MergeMode


def _peer_url(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}.git"


def _create_local_peer(url: str) -> Path | None:
    """Create an empty bare repository at a local path.

    Returns the topmost directory this call made (so a rollback removes the parents it created
    too), or None when the path already existed. Any failure, also a timeout or a cancellation,
    removes what the call made.
    """
    path = Path(url)
    if path.exists():
        return None
    topmost = path
    while not topmost.parent.exists():
        topmost = topmost.parent
    path.mkdir(mode=0o700, parents=True)
    try:
        result = subprocess.run(
            ["git", "init", "--bare", "--initial-branch=main", str(path)],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise GitProjectCreateError(result.stderr.decode("utf-8", errors="replace"))
    except BaseException:
        shutil.rmtree(topmost, ignore_errors=True)
        raise
    return topmost


def _require_peer_apart_from_projects(
    url: str, roots: list[Path], repositories: list[Path], name: str
) -> None:
    """A local peer must be a place of its own: not a note root and not a project's own
    repository (nor over or inside one). Otherwise a project would be its own peer, every
    sync "equal", or the peer's history would end up among the notes."""
    peer = Path(url).resolve()
    if any(_overlaps(peer, other) for other in roots + repositories):
        raise GitProjectCreateError(
            f"Peer path {peer} for {name!r} overlaps a project root or a project repository"
        )


async def setup_git_peers(
    names: Iterable[str],
    *,
    peer_base: str | None,
    merge: MergeMode = "off",
    manager: ConfigManager | None = None,
) -> list[PeerChange]:
    """Give each named project a peer at `<peer_base>/<name>.git` (or remove it).

    `peer_base=None` removes the peers (and any merge setting) of the named projects.
    """
    with cancel_on_termination():
        return await _setup_git_peers(names, peer_base=peer_base, merge=merge, manager=manager)


async def _setup_git_peers(
    names: Iterable[str],
    *,
    peer_base: str | None,
    merge: MergeMode,
    manager: ConfigManager | None,
) -> list[PeerChange]:
    manager = manager or ConfigManager()
    config_file = Path(manager.config_file)
    config = BasicMemoryConfig.model_validate_json(config_file.read_bytes())
    if not config.git_repositories:
        raise GitProjectCreateError("This profile is not a Git host")
    wanted = list(dict.fromkeys(names))
    if not wanted:
        raise GitProjectCreateError("No project named")
    _, maker = await db.get_or_create_db(
        config.database_path, ensure_migrations=False, config=config
    )
    async with maker() as session:
        projects = {p.name: p for p in await session.scalars(select(Project))}
    unknown = [name for name in wanted if name not in projects]
    if unknown:
        raise GitProjectCreateError(f"Unknown project(s): {', '.join(unknown)}")
    require_host_stopped(
        (Path(p.path), config.git_repositories[UUID(p.external_id)]) for p in projects.values()
    )
    remotes = dict(config.git_sync_remotes)
    merges = dict(config.git_sync_merge)
    changes: list[PeerChange] = []
    made: list[Path] = []
    try:
        for name in wanted:
            project_id = UUID(projects[name].external_id)
            if peer_base is None:
                remotes.pop(project_id, None)
                merges.pop(project_id, None)
                changes.append(PeerChange(name, project_id, None, "off"))
                continue
            url = _peer_url(peer_base, name)
            remote = GitSyncRemote(url=url)  # validates the endpoint
            if Path(url).is_absolute():
                _require_peer_apart_from_projects(
                    url,
                    [Path(p.path).resolve() for p in projects.values()],
                    [r.resolve() for r in config.git_repositories.values()],
                    name,
                )
                if (created_root := _create_local_peer(url)) is not None:
                    made.append(created_root)
            remotes[project_id] = remote
            if merge == "auto":
                merges[project_id] = "auto"
            else:
                merges.pop(project_id, None)
            changes.append(PeerChange(name, project_id, url, merge))
        future = config.model_copy(update={"git_sync_remotes": remotes, "git_sync_merge": merges})
        require_git_peer_settings(future)
        raw = json.loads(config_file.read_text())
        raw["git_sync_remotes"] = {str(k): {"url": v.url} for k, v in remotes.items()}
        raw["git_sync_merge"] = {str(k): v for k, v in merges.items()}
        BasicMemoryConfig.model_validate(raw)
        _write_config_atomically(config_file, raw)
    except BaseException:
        import shutil

        for path in reversed(made):
            shutil.rmtree(path, ignore_errors=True)
        raise
    return changes
