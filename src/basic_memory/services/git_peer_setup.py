"""Point Git projects at a peer, by project name, on a stopped host (BM-009).

`git_sync_remotes` and `git_sync_merge` are keyed by project UUID and read only when the
service starts. This tool takes the names, finds the UUIDs, creates the bare repositories
of a local peer when asked, checks the result against the host's start rules and replaces
the configuration atomically. Which projects have a peer is configuration, not code: a
project that is not named here keeps whatever it had, and a project may have none.
"""

import json
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
    _write_config_atomically,
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


def _create_local_peer(url: str) -> bool:
    """Create an empty bare repository at a local path; True when it was created."""
    path = Path(url)
    if path.exists():
        return False
    path.mkdir(mode=0o700, parents=True)
    result = subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(path)],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise GitProjectCreateError(result.stderr.decode("utf-8", errors="replace"))
    return True


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
            if Path(url).is_absolute() and _create_local_peer(url):
                made.append(Path(url))
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
