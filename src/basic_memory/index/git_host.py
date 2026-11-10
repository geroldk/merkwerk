"""Explicit local Git host ownership; externally controlled adoption comes first."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory.config import BasicMemoryConfig, DatabaseBackend, ProjectMode
from basic_memory.index.git_projection import build_git_projection_pipeline
from basic_memory.index.local_notes import LocalAcceptedNotePreparerFactory
from basic_memory.indexing.accepted_note_mutation_runner import (
    AcceptedNoteMutationDependencies,
    AcceptedNoteMutationMovePolicy,
)
from basic_memory.models import Project
from basic_memory.read_cache import ReadCacheInvalidator
from basic_memory.repository.accepted_note_repositories import AcceptedNoteRepositories
from basic_memory.repository.git_host_identities import GitHostIdentities, GitHostProject
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.repository.project_repository import ProjectRepository
from basic_memory.services.git_cross_project_move import (
    reserve_open_moves,
    resume_cross_project_moves,
)
from basic_memory.services.git_project_runtime import GitProjectRuntime


@dataclass(frozen=True, slots=True)
class GitHost:
    """All local projects are Git-owned; no mutable registry operations in this slice."""

    runtimes: Mapping[str, GitProjectRuntime]
    # Host-wide note ownership (I1), shared with every runtime's projection pipeline.
    identities: GitHostIdentities | None = None
    # One cross-project move (or move resume) at a time on this host (I1).
    move_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def runtime_for(self, project_external_id: str) -> GitProjectRuntime:
        runtime = self.runtimes.get(project_external_id)
        if runtime is None:
            raise GitProjectionConflict("Project is not owned by this Git host")
        return runtime


def require_git_peer_settings(app_config: BasicMemoryConfig) -> None:
    """Peers and automatic merging may only name configured Git projects."""
    if not set(app_config.git_sync_remotes) <= set(app_config.git_repositories):
        raise GitProjectionConflict("Git peers must refer to configured Git project UUIDs")
    if not set(app_config.git_sync_merge) <= set(app_config.git_sync_remotes):
        raise GitProjectionConflict("Git merge settings must refer to projects with a Git peer")


def require_git_project_roots(app_config: BasicMemoryConfig, projects: list[Project]) -> None:
    """Every registered project is configured, local, active and Git-managed (host start rule).

    Shared with the offline project tools so that nothing they write can leave a host that
    refuses to start.
    """
    if (
        not projects
        or {UUID(project.external_id) for project in projects} != set(app_config.git_repositories)
        or len(projects) != len(app_config.projects)
    ):
        raise GitProjectionConflict("Every configured and registered project must be Git-managed")
    for project in projects:
        entry = app_config.project_entry(project.name)
        repository_path = app_config.git_repositories[UUID(project.external_id)]
        if (
            entry is None
            or entry.mode is not ProjectMode.LOCAL
            or not project.is_active
            or not Path(project.path).is_absolute()
            or not Path(entry.path).is_absolute()
            or Path(project.path).resolve() != Path(entry.path).resolve()
            or not repository_path.is_absolute()
        ):
            raise GitProjectionConflict("Git host requires matching active local project roots")


@asynccontextmanager
async def run_git_host(
    app_config: BasicMemoryConfig,
    session_maker: async_sessionmaker[AsyncSession],
    *,
    read_cache: ReadCacheInvalidator | None = None,
) -> AsyncIterator[GitHost | None]:
    """Open adopted repositories and recover SQL before admitting any requests.

    A mixed legacy/Git host could change a referenced target without invalidating
    its Git source. Refuse that topology until every legacy path can participate.
    This context does not stop external processes, editors or Cloud sync jobs.
    """
    require_git_peer_settings(app_config)
    if not app_config.git_repositories:
        yield None
        return
    if (
        app_config.database_backend is not DatabaseBackend.SQLITE
        or app_config.skip_local_initialization
    ):
        raise GitProjectionConflict("Git hosting currently requires standalone local SQLite")
    async with session_maker() as session:
        projects = list(await session.scalars(select(Project).order_by(Project.id)))
    require_git_project_roots(app_config, projects)
    repositories = AcceptedNoteRepositories()
    dependencies = AcceptedNoteMutationDependencies(
        project_repository=ProjectRepository(),
        lookup_repositories=repositories,
        write_repositories=repositories,
        preparer_factory=LocalAcceptedNotePreparerFactory(
            session_maker=session_maker, app_config=app_config, stamp_note_times=True
        ),
        move_policy=AcceptedNoteMutationMovePolicy(
            update_permalinks_on_move=app_config.update_permalinks_on_move
        ),
        verify_storage_absent_on_create=True,
    )
    # Every repository first: each projection asks who else holds a note UUID (I1).
    repositories: dict[int, GitProjectRepository] = {}
    for project in projects:
        repository = GitProjectRepository(app_config.git_repositories[UUID(project.external_id)])
        snapshot = await asyncio.to_thread(repository.snapshot)
        if snapshot.manifest.project_external_id != UUID(project.external_id):
            raise GitProjectionConflict("Configured Git repository belongs to another project UUID")
        repositories[project.id] = repository
    identities = GitHostIdentities(
        GitHostProject(
            project.id, UUID(project.external_id), project.name, repositories[project.id]
        )
        for project in projects
    )
    runtimes: dict[str, GitProjectRuntime] = {}
    for project in projects:
        runtimes[project.external_id] = GitProjectRuntime(
            build_git_projection_pipeline(
                repositories[project.id],
                project=project,
                session_maker=session_maker,
                app_config=app_config,
                read_cache=read_cache,
                identities=identities,
            ),
            dependencies,
            transaction_idle_seconds=app_config.git_transaction_idle_seconds,
            transaction_retention_seconds=app_config.git_transaction_retention_seconds,
            sync_remote=app_config.git_sync_remotes.get(UUID(project.external_id)),
            sync_merge=app_config.git_sync_merge.get(UUID(project.external_id)) == "auto",
        )
        identities.register(project.id, runtimes[project.external_id].worker.wake_index)
    async with AsyncExitStack() as stack:
        for runtime in runtimes.values():
            await stack.enter_async_context(runtime.run())
        host = GitHost(MappingProxyType(runtimes), identities)
        # Trigger: a cross-project move was interrupted (crash, restart) (I1).
        # Why: its note may be in no project's current state; its UUID must not be taken
        #   by a write before the move finishes, and every startup recover has run now.
        # Outcome: the UUIDs are reserved, then each move is finished, undone or kept as
        #   stuck, one by one; a failing one is logged and never stops the host.
        reserve_open_moves(host)
        await resume_cross_project_moves(host)
        # Runtimes start in project order; one that waited for a later owner retries now.
        identities.nudge_waiting()
        yield host
