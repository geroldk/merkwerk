"""Remove empty Git projects from a stopped host (BM-009): the way back for `create-project`.

Only a project that never held a note can go: no entity rows, a repository that still has
nothing but its initialization commit, and a note root that holds nothing but the control
directory. Everything else is refused before anything changes, because a published history
is not this tool's to delete.

The database rows and the configuration change together: the rows are deleted inside a
transaction that stays open while the configuration is replaced atomically, and the
transaction commits last. If either step fails, both are as they were. The directories are
removed after that; a directory that cannot be removed is reported, not hidden.
"""

import asyncio
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, ConfigManager
from basic_memory.index.git_host import require_git_peer_settings, require_git_project_roots
from basic_memory.models import Entity, Project
from basic_memory.repository.git_project_repository import (
    GitProjectRepository,
    GitPublicationError,
)
from basic_memory.services.git_project_create import (
    GitProjectCreateError,
    _alembic_at_head,
    _overlaps,
    _write_config_atomically,
    require_host_stopped,
)

CONTROL_DIRECTORY = ".basic-memory"


@dataclass(frozen=True, slots=True)
class RemovedGitProject:
    name: str
    project_id: UUID
    path: Path
    repository: Path
    had_peer: bool
    leftovers: tuple[Path, ...] = field(default=())


def _require_unused_repository(name: str, repository_path: Path) -> None:
    """The repository must still be exactly what `create-project` made."""
    try:
        GitProjectRepository(repository_path).initial_snapshot()
    except (GitPublicationError, ValueError, OSError) as error:
        raise GitProjectCreateError(
            f"Project {name!r} has been used or its repository is unreadable ({error}); not removed"
        ) from error


def _require_empty_root(name: str, root: Path) -> None:
    if root.is_symlink():
        raise GitProjectCreateError(f"Project root {root} of {name!r} is a symbolic link")
    if not root.exists():
        return
    if not root.is_dir():
        raise GitProjectCreateError(f"Project root {root} of {name!r} is not a directory")
    others = sorted(entry.name for entry in root.iterdir() if entry.name != CONTROL_DIRECTORY)
    if others:
        raise GitProjectCreateError(
            f"Project root {root} of {name!r} holds files ({', '.join(others[:3])}); not removed"
        )
    control = root / CONTROL_DIRECTORY
    if control.is_symlink():
        raise GitProjectCreateError(f"Control directory {control} is a symbolic link")


async def remove_empty_git_projects(
    names: Iterable[str],
    *,
    manager: ConfigManager | None = None,
) -> list[RemovedGitProject]:
    """Remove the named, still empty projects, or leave the installation as it was."""
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
        await _alembic_at_head(session)
        projects = list(await session.scalars(select(Project).order_by(Project.id)))
        by_name = {project.name: project for project in projects}
        unknown = [name for name in wanted if name not in by_name]
        if unknown:
            raise GitProjectCreateError(f"Unknown project(s): {', '.join(unknown)}")
        used = {
            name
            for name in wanted
            if await session.scalar(
                select(func.count())
                .select_from(Entity)
                .where(Entity.project_id == by_name[name].id)
            )
        }
    require_git_peer_settings(config)
    require_git_project_roots(config, projects)
    require_host_stopped(
        (Path(project.path), config.git_repositories[UUID(project.external_id)])
        for project in projects
    )
    if used:
        raise GitProjectCreateError(
            f"Project(s) with notes are not removed: {', '.join(sorted(used))}"
        )
    if len(wanted) >= len(projects):
        raise GitProjectCreateError("At least one project must remain")
    if config.default_project in wanted:
        raise GitProjectCreateError(
            f"{config.default_project!r} is the default project; choose another default first"
        )

    removed: list[RemovedGitProject] = []
    keep = [project for project in projects if project.name not in wanted]
    kept_paths = [Path(p.path).resolve() for p in keep] + [
        config.git_repositories[UUID(p.external_id)].resolve() for p in keep
    ]
    for name in wanted:
        project = by_name[name]
        project_id = UUID(project.external_id)
        root = Path(project.path)
        repository_path = config.git_repositories[project_id]
        if any(_overlaps(root.resolve(), other) for other in kept_paths) or any(
            _overlaps(repository_path.resolve(), other) for other in kept_paths
        ):
            raise GitProjectCreateError(f"Project {name!r} shares a directory with another project")
        _require_unused_repository(name, repository_path)
        _require_empty_root(name, root)
        removed.append(
            RemovedGitProject(
                name,
                project_id,
                root,
                repository_path,
                had_peer=project_id in config.git_sync_remotes,
            )
        )

    future = config.model_copy(deep=True)
    raw_text = config_file.read_text()
    raw = json.loads(raw_text)
    for item in removed:
        future.projects.pop(item.name, None)
        future.git_repositories.pop(item.project_id, None)
        future.git_sync_remotes.pop(item.project_id, None)
        future.git_sync_merge.pop(item.project_id, None)
        for section, key in (
            ("projects", item.name),
            ("git_repositories", str(item.project_id)),
            ("git_sync_remotes", str(item.project_id)),
            ("git_sync_merge", str(item.project_id)),
        ):
            raw.get(section, {}).pop(key, None)
    require_git_peer_settings(future)
    require_git_project_roots(future, keep)
    BasicMemoryConfig.model_validate(raw)  # the profile that will be written must load

    try:
        async with maker() as session, session.begin():
            for item in removed:
                row = await session.scalar(
                    select(Project).where(Project.external_id == str(item.project_id))
                )
                if row is not None:
                    await session.delete(row)
            await session.flush()
            _write_config_atomically(config_file, raw)
        # leaving the block committed the deletion
    except BaseException:
        # The rows roll back by themselves; the configuration is put back whenever it differs
        # from what it was, wherever in the write or the commit the failure happened.
        if config_file.read_text() != raw_text:
            _write_text_atomically(config_file, raw_text)
        raise
    return [await asyncio.to_thread(_remove_directories, item) for item in removed]


def _write_text_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.restore")
    temporary.write_text(text)
    os.replace(temporary, path)


def _remove_directories(item: RemovedGitProject) -> RemovedGitProject:
    leftovers: list[Path] = []
    for path in (item.path, item.repository):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError:
            leftovers.append(path)
    return RemovedGitProject(
        item.name, item.project_id, item.path, item.repository, item.had_peer, tuple(leftovers)
    )
