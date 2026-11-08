"""Create new, empty Git projects on a stopped host (BM-009).

The Git host builds one runtime per project when it starts, and refuses a profile in which
a registered project is not Git-managed. A new project is therefore created offline, with
the service stopped, and becomes visible at the next start.

One call registers every requested project or none: the database rows, the bare
repositories, the note roots and the configuration entries. The configuration is replaced
last, atomically, and only after the future profile passed the host's own start rules.
"""

import asyncio
import json
import os
import re
import shutil
import signal
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from filelock import FileLock, Timeout
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, ConfigManager, ProjectEntry
from basic_memory.index.git_host import require_git_peer_settings, require_git_project_roots
from basic_memory.index.git_project_ownership import own_git_project
from basic_memory.models import Project
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_projection_repository import register_git_project
from basic_memory.utils import generate_permalink

NAME_PATTERN = re.compile(r"[a-z][a-z0-9-]{1,31}")
# Directories below the base directory that hold something other than notes.
RESERVED_DIRECTORIES = (
    "repositories",
    "peer",
    "operations",
    "code",
    "vendor",
    "bin",
    "instanz",
    "systemd",
)


class GitProjectCreateError(ValueError):
    """The request cannot be carried out; nothing was changed."""


@dataclass(frozen=True, slots=True)
class NewGitProject:
    name: str
    path: Path | None = None  # default: <base_dir>/<name>


@dataclass(frozen=True, slots=True)
class CreatedGitProject:
    name: str
    project_id: UUID
    path: Path
    repository: Path


def _overlaps(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


def plan_new_projects(
    requests: Iterable[NewGitProject],
    *,
    base_dir: Path,
    repositories_dir: Path,
    existing_names: Iterable[str],
    existing_permalinks: Iterable[str],
    existing_roots: Iterable[Path],
) -> list[tuple[str, Path, Path]]:
    """Validate the request and return (name, root, repository) per project, or raise."""
    base = base_dir.resolve()
    repositories = repositories_dir.resolve()
    taken_names = {name.lower() for name in existing_names}
    taken_permalinks = set(existing_permalinks)
    roots = [root.resolve() for root in existing_roots]
    reserved = [base / name for name in RESERVED_DIRECTORIES]
    planned: list[tuple[str, Path, Path]] = []
    for request in requests:
        name = request.name
        if not NAME_PATTERN.fullmatch(name):
            raise GitProjectCreateError(
                f"Project name {name!r} must match {NAME_PATTERN.pattern} (lower case, 2-32 characters)"
            )
        permalink = generate_permalink(name)
        if name in taken_names or name.lower() in taken_names or permalink in taken_permalinks:
            raise GitProjectCreateError(f"Project name {name!r} is already taken")
        if request.path is not None and not request.path.is_absolute():
            raise GitProjectCreateError(f"Project root {request.path} must be an absolute path")
        root = (request.path or base / name).resolve()
        if root == base or base.is_relative_to(root):
            raise GitProjectCreateError(
                f"Project root {root} must not be the base directory {base} or contain it"
            )
        if any(_overlaps(root, other) for other in roots):
            raise GitProjectCreateError(f"Project root {root} overlaps another project root")
        if any(_overlaps(root, other) for other in reserved) or _overlaps(root, repositories):
            raise GitProjectCreateError(f"Project root {root} lies in a reserved directory")
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise GitProjectCreateError(f"Project root {root} exists and is not an empty directory")
        repository = repositories / f"{name}.git"
        if repository.exists() or _overlaps(repository, root):
            raise GitProjectCreateError(f"Repository path {repository} is already in use")
        taken_names.add(name.lower())
        taken_permalinks.add(permalink)
        roots.append(root)
        planned.append((name, root, repository))
    if not planned:
        raise GitProjectCreateError("No project requested")
    return planned


def require_host_stopped(roots_and_repositories: Iterable[tuple[Path, Path]]) -> None:
    """Refuse when a service still owns any existing project (it holds these locks)."""
    for root, repository in roots_and_repositories:
        for lock in (
            root / ".basic-memory" / "git-projection.lock",
            repository / "bm-projection.lock",
        ):
            if not lock.parent.is_dir():
                continue
            try:
                with FileLock(lock, timeout=0, thread_local=False):
                    pass
            except Timeout as error:
                raise GitProjectCreateError(
                    "A Basic Memory service still owns a project; stop it first"
                ) from error


def write_config_text_atomically(path: Path, text: str) -> None:
    """Replace `path` with `text`, keeping the rights it had (a new file gets 0600).

    The configuration can hold credentials; a plain write_text under umask 0022 would turn
    a 0600 file into 0644 for good.
    """
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    temporary = path.with_name(f".{path.name}.new")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)  # os.open honours the umask; this does not
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_config_atomically(path: Path, data: dict[str, object]) -> None:
    write_config_text_atomically(path, json.dumps(data, indent=2))


@contextmanager
def cancel_on_termination() -> Iterator[None]:
    """Turn SIGTERM and SIGHUP into a cancellation of the running task.

    Without it a closed terminal or a `kill` ends the process between two steps, past the
    `except BaseException` rollback, and leaves database rows the configuration does not know:
    the service then refuses to start. A cancelled task reaches its rollback.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    installed: list[signal.Signals] = []
    if task is not None:
        for number in (signal.SIGTERM, signal.SIGHUP):
            try:
                loop.add_signal_handler(number, task.cancel)
                installed.append(number)
            except (NotImplementedError, RuntimeError, ValueError):  # not the main thread
                continue
    try:
        yield
    finally:
        for number in installed:
            loop.remove_signal_handler(number)


async def _alembic_at_head(session: AsyncSession) -> None:
    from alembic.config import Config as AlembicConfig
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    current = (await session.execute(text("select version_num from alembic_version"))).scalar()
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(Path(db.__file__).parent / "alembic"))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    if current not in heads:
        raise GitProjectCreateError(
            f"Database is at {current}, code head is {heads}; migrate deliberately first"
        )


async def create_git_projects(
    requests: Iterable[NewGitProject],
    *,
    base_dir: Path,
    repositories_dir: Path,
    manager: ConfigManager | None = None,
) -> list[CreatedGitProject]:
    """Create the requested projects, or leave the installation exactly as it was.

    A SIGTERM or SIGHUP cancels the work and reaches the rollback like any other failure;
    only SIGKILL, a power cut or an out-of-memory kill can leave a half-made state (the
    runbook takes a backup first for that).
    """
    with cancel_on_termination():
        return await _create_git_projects(
            requests, base_dir=base_dir, repositories_dir=repositories_dir, manager=manager
        )


async def _create_git_projects(
    requests: Iterable[NewGitProject],
    *,
    base_dir: Path,
    repositories_dir: Path,
    manager: ConfigManager | None,
) -> list[CreatedGitProject]:
    manager = manager or ConfigManager()
    config_file = Path(manager.config_file)
    config = BasicMemoryConfig.model_validate_json(config_file.read_bytes())
    if not config.git_repositories:
        raise GitProjectCreateError("This profile is not a Git host; adopt its projects first")
    _, maker = await db.get_or_create_db(
        config.database_path, ensure_migrations=False, config=config
    )
    created: list[CreatedGitProject] = []
    registered_ids: list[UUID] = []  # rows committed so far, also before the project is "created"
    made_paths: list[Path] = []
    try:
        async with maker() as session:
            await _alembic_at_head(session)
            projects = list(await session.scalars(select(Project).order_by(Project.id)))
        require_git_peer_settings(config)
        require_git_project_roots(config, projects)
        require_host_stopped(
            (Path(project.path), config.git_repositories[UUID(project.external_id)])
            for project in projects
        )
        planned = plan_new_projects(
            requests,
            base_dir=base_dir,
            repositories_dir=repositories_dir,
            existing_names=[p.name for p in projects] + list(config.projects),
            existing_permalinks=[p.permalink for p in projects],
            existing_roots=[Path(p.path) for p in projects]
            + [Path(entry.path) for entry in config.projects.values()],
        )
        _make_directory(repositories_dir, made_paths)
        future = config.model_copy(deep=True)
        for name, root, repository_path in planned:
            project_id = uuid4()
            # Noted before the call: the plan guarantees the path is free, and a failure inside
            # `create` (git init, first commit) must not leave a half-made repository behind.
            made_paths.append(repository_path)
            repository = GitProjectRepository.create(
                repository_path, project_external_id=project_id
            )
            _make_directory(root, made_paths)
            async with maker() as session, session.begin():
                project = await register_git_project(
                    session, repository.snapshot().manifest, name=name, path=root
                )
            registered_ids.append(project_id)
            # Writes the same binding markers the service would, so a backup taken before
            # the first start finds them.
            await _bind(repository, project, root, maker)
            future.projects[name] = ProjectEntry(path=str(root))
            future.git_repositories[project_id] = repository_path
            created.append(CreatedGitProject(name, project_id, root, repository_path))
        async with maker() as session:
            registered = list(await session.scalars(select(Project).order_by(Project.id)))
        require_git_peer_settings(future)
        require_git_project_roots(future, registered)
        raw = json.loads(config_file.read_text())
        raw.setdefault("projects", {})
        raw.setdefault("git_repositories", {})
        for item in created:
            raw["projects"][item.name] = {"path": str(item.path), "mode": "local"}
            raw["git_repositories"][str(item.project_id)] = str(item.repository)
        BasicMemoryConfig.model_validate(raw)  # the profile that will be written must load
        _write_config_atomically(config_file, raw)
    except BaseException:
        await _roll_back(maker, registered_ids, made_paths)
        raise
    return created


def _make_directory(path: Path, made_paths: list[Path]) -> None:
    """Create `path`, remembering the topmost directory this call brought into being."""
    if path.exists():
        return
    topmost = path
    while not topmost.parent.exists():
        topmost = topmost.parent
    path.mkdir(mode=0o700, parents=True)
    made_paths.append(topmost)


async def _bind(
    repository: GitProjectRepository,
    project: Project,
    root: Path,
    maker: async_sessionmaker[AsyncSession],
) -> None:
    async with own_git_project(
        repository,
        project_id=project.id,
        project_external_id=project.external_id,
        project_root=root,
        session_maker=maker,
    ):
        pass


async def _roll_back(
    maker: async_sessionmaker[AsyncSession],
    registered_ids: list[UUID],
    made_paths: list[Path],
) -> None:
    """Undo what this call made: database rows, then only the directories it created."""
    async with maker() as session, session.begin():
        for project_id in registered_ids:
            row = await session.scalar(
                select(Project).where(Project.external_id == str(project_id))
            )
            if row is not None:
                await session.delete(row)
    for path in reversed(made_paths):
        shutil.rmtree(path, ignore_errors=True)
