"""Opt-in local Git projection through BM's guarded file and real index pipeline.

No normal startup or write dependency enables this experimental composition yet.
The caller owns the project and must exclude legacy writers and other workers.
"""

import asyncio
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory.config import BasicMemoryConfig
from basic_memory.file_utils import compute_checksum
from basic_memory.index.local_dependencies import build_local_markdown_file_indexer
from basic_memory.index.git_project_ownership import require_registered_git_root
from basic_memory.index.note_content_materialization import (
    LocalNoteContentStorage,
    run_recovery_materialization,
)
from basic_memory.indexing.batch_indexer import BatchIndexer
from basic_memory.indexing.index_file_runner import IndexFileExecutor
from basic_memory.indexing.models import IndexFrontmatterUpdate, IndexFrontmatterWriteResult
from basic_memory.markdown import EntityParser, MarkdownProcessor
from basic_memory.models import Entity, NoteContent, Project
from basic_memory.read_cache import ReadCacheInvalidator, invalidate_cache
from basic_memory.repository import EntityRepository, ObservationRepository, RelationRepository
from basic_memory.repository.git_progress_repository import GitProjectionStage
from basic_memory.repository.git_graph_inputs import git_graph_identities_unchanged
from basic_memory.repository.git_layout_repository import GitLayoutRepository
from basic_memory.repository.git_project_repository import GitProjectRepository, GitSnapshot
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.repository.search_repository import create_search_repository
from basic_memory.runtime.note_content import (
    RuntimeNoteMaterializationJobRequest,
    RuntimeNoteMaterializationStatus,
)
from basic_memory.runtime.cleanup import RuntimeGuardedFileDeleteOutcome
from basic_memory.services import EntityService, FileService
from basic_memory.services.git_project_projection import project_git_content
from basic_memory.services.link_resolver import LinkResolver
from basic_memory.services.search_service import SearchService
from basic_memory.workspace_context import current_workspace_permalink_context


class PublishedGitIndexWriter:
    """Indexing may derive state, but never normalize already published Markdown."""

    async def write_frontmatter(
        self, update: IndexFrontmatterUpdate
    ) -> IndexFrontmatterWriteResult:
        raise GitProjectionConflict(f"Indexing would rewrite published Git bytes: {update.path}")


@dataclass(frozen=True, slots=True)
class GitProjectionPipeline:
    repository: GitProjectRepository
    project_id: int
    project_external_id: str
    session_maker: async_sessionmaker[AsyncSession]
    file_service: FileService
    file_indexer: IndexFileExecutor
    search_service: SearchService
    vector_key: str
    read_cache: ReadCacheInvalidator | None = None
    graph_key: str = "1"

    async def apply(
        self,
        stage: GitProjectionStage,
        commit_id: str,
        *,
        graph_base_commit: str | None = None,
    ) -> None:
        await self.require_registration()
        if stage == "content":
            await self.recover_content(commit_id)
            return

        snapshot = await asyncio.to_thread(self.repository.snapshot, commit_id)
        graph_paths: set[str] | None = None
        if stage == "graph" and graph_base_commit is not None:
            previous = await asyncio.to_thread(self.repository.snapshot, graph_base_commit)
            if git_graph_identities_unchanged(
                previous.files,
                previous.manifest.note_ids,
                snapshot.files,
                snapshot.manifest.note_ids,
            ):
                graph_paths = {
                    path
                    for path, content in snapshot.files.items()
                    if previous.files[path] != content
                }
        invalidation = (
            invalidate_cache(self.read_cache, self.project_external_id)
            if self.read_cache is not None
            else nullcontext()
        )
        async with invalidation:
            if stage == "files":
                await self.retire_old_files(snapshot)
            if stage == "vectors" and not await self.search_service.semantic_effectively_enabled():
                # A missing backend must remain visible as unavailable, never "ready".
                raise GitProjectionConflict("Semantic backend unavailable; vectors remain pending")
            for note in snapshot.manifest.notes:
                if graph_paths is not None and note.path not in graph_paths:
                    continue
                await self.require_head(commit_id)
                async with self.session_maker() as session:
                    entity = await session.scalar(
                        select(Entity).where(
                            Entity.project_id == self.project_id,
                            Entity.external_id == str(note.external_id),
                            Entity.file_path == note.path,
                        )
                    )
                    content = (
                        await session.get(NoteContent, entity.id) if entity is not None else None
                    )
                    if content is None or content.markdown_content != snapshot.files[note.path]:
                        raise GitProjectionConflict("SQL content has not reached this Git target")
                    entity_id, version, checksum = (
                        content.entity_id,
                        content.db_version,
                        content.db_checksum,
                    )
                if stage == "files":
                    result = await run_recovery_materialization(
                        RuntimeNoteMaterializationJobRequest(
                            project_id=self.project_id,
                            entity_id=entity_id,
                            db_version=version,
                            db_checksum=checksum,
                            source="git-projection",
                        ),
                        session_maker=self.session_maker,
                        file_service=self.file_service,
                    )
                    if result.status is not RuntimeNoteMaterializationStatus.written:
                        raise GitProjectionConflict(f"File materialization: {result.status.value}")
                await self.require_file(snapshot, note.path)
                if stage == "graph":
                    indexed = await self.file_indexer.index_file(note.path, source="git-projection")
                    if (
                        indexed.content_superseded
                        or indexed.external_id != str(note.external_id)
                        or indexed.content_checksum
                        != await compute_checksum(snapshot.files[note.path])
                    ):
                        raise GitProjectionConflict(
                            "Index did not publish the target note generation"
                        )
                    await self.require_file(snapshot, note.path)
                elif stage == "vectors":
                    vectors = await self.search_service.sync_entity_vectors(entity_id)
                    if (
                        vectors.entities_failed
                        or vectors.entities_deferred
                        # An unchanged entity is both synced and skipped by BM's
                        # repository; these counters are not disjoint outcomes.
                        or not (vectors.entities_synced == 1 or vectors.entities_skipped == 1)
                    ):
                        raise GitProjectionConflict(
                            "Vector sync incomplete: " + "; ".join(vectors.sample_errors)
                        )
            if stage == "graph":
                await self.search_service.repository.purge_stale_search_rows()
            elif stage == "vectors":
                await self.search_service.repository.delete_stale_vector_rows()
                await self.search_service.repository.reconcile_vector_index()
        await self.require_head(commit_id)

    async def recover_content(
        self,
        commit_id: str,
        *,
        redrive_materialization: bool = True,
        known_ancestor: GitSnapshot | None = None,
    ) -> None:
        await self.require_registration()
        await project_git_content(
            self.repository,
            project_id=self.project_id,
            session_maker=self.session_maker,
            read_cache=self.read_cache,
            commit_id=commit_id,
            redrive_materialization=redrive_materialization,
            known_ancestor=known_ancestor,
        )

    async def retire_old_files(self, snapshot: GitSnapshot) -> None:
        """Remove only proven older bytes; a private editor file blocks completion."""
        retirements = GitLayoutRepository(self.project_id)
        async with self.session_maker() as session:
            pending = await retirements.retirements(session)
        storage = LocalNoteContentStorage(self.file_service)
        for retirement in pending:
            await self.require_head(snapshot.commit_id)
            path = retirement.file_path
            if await storage.exists(path):
                checksum = await storage.compute_checksum(path)
                desired = snapshot.files.get(path)
                if desired is not None and checksum == await compute_checksum(desired):
                    # The path was reused and already holds the new published note.
                    # Its current bytes supersede this older retirement permission.
                    pass
                elif checksum in retirement.expected_checksums:
                    deleted = await storage.delete_file_if_matches(path, expected_checksum=checksum)
                    if deleted is RuntimeGuardedFileDeleteOutcome.changed:
                        raise GitProjectionConflict("Old file changed during guarded Git cleanup")
                else:
                    raise GitProjectionConflict(
                        "Old published path contains an unpublished editor change"
                    )
            async with self.session_maker() as session, session.begin():
                await retirements.complete_retirement(session, retirement)

    async def require_head(self, commit_id: str) -> None:
        if commit_id != await asyncio.to_thread(self.repository.head):
            raise GitProjectionConflict("Git head advanced; retry the latest published target")

    async def require_registration(self) -> None:
        async with self.session_maker() as session:
            await require_registered_git_root(
                session,
                project_id=self.project_id,
                project_external_id=self.project_external_id,
                root=self.file_service.base_path,
            )

    async def require_file(self, snapshot: GitSnapshot, path: str) -> None:
        actual = await self.file_service.read_file_bytes(path)
        if actual != snapshot.files[path].encode("utf-8"):
            raise GitProjectionConflict(
                "File differs from published Git bytes; import requires review"
            )


def build_git_projection_pipeline(
    repository: GitProjectRepository,
    *,
    project: Project,
    session_maker: async_sessionmaker[AsyncSession],
    app_config: BasicMemoryConfig,
    read_cache: ReadCacheInvalidator | None = None,
) -> GitProjectionPipeline:
    """Explicit local composition; no global config, new daemon or default enablement."""
    root = Path(project.path)
    parser = EntityParser(root)
    files = FileService(
        root, MarkdownProcessor(parser, app_config=app_config), app_config=app_config
    )
    entities = EntityRepository(project.id)
    observations = ObservationRepository(project.id)
    relations = RelationRepository(project.id)
    searches = create_search_repository(session_maker, project.id, app_config)
    search = SearchService(searches, entities, files, session_maker)
    entity_service = EntityService(
        parser,
        entities,
        observations,
        relations,
        files,
        LinkResolver(entities, search, session_maker, app_config),
        session_maker,
        search_service=search,
        app_config=app_config,
    )
    batch = BatchIndexer(
        project_id=project.id,
        app_config=app_config,
        entity_service=entity_service,
        entity_repository=entities,
        observation_repository=observations,
        relation_repository=relations,
        search_service=search,
        file_writer=PublishedGitIndexWriter(),
        session_maker=session_maker,
    )
    indexer = build_local_markdown_file_indexer(
        project_id=project.id,
        file_service=files,
        session_maker=session_maker,
        entity_repository=entities,
        batch_indexer=batch,
        search_service=search,
    )
    vector_key = json.dumps(
        {
            "version": 1,
            "model": searches.configured_embedding_model,
            "index": searches.configured_vector_index,
            "dimensions": app_config.semantic_embedding_dimensions,
            "max_tokens": app_config.semantic_embedding_max_tokens,
        },
        sort_keys=True,
    )
    workspace = current_workspace_permalink_context()
    graph_key = json.dumps(
        {
            "version": 2,
            "permalinks_include_project": app_config.permalinks_include_project,
            "ensure_frontmatter_on_sync": app_config.ensure_frontmatter_on_sync,
            "workspace": (
                workspace.workspace_slug
                if workspace is not None and workspace.should_prefix_permalinks
                else None
            ),
        },
        sort_keys=True,
    )
    return GitProjectionPipeline(
        repository,
        project.id,
        project.external_id,
        session_maker,
        files,
        indexer,
        search,
        vector_key,
        read_cache,
        graph_key,
    )
