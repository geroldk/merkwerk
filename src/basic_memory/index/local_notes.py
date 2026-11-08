"""Local runtime implementations for accepted-note mutations and note-file cleanup.

The accepted-note mutation runner, note-content mutation service, and
directory-delete runtime are storage-neutral; this module supplies their
filesystem-backed local implementations. The FastAPI composition root in
``basic_memory.deps.services`` wires these into route dependencies; cloud
composes its own tenant-scoped equivalents behind the same protocols.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, override

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig
from basic_memory.file_utils import FileError
from basic_memory.indexing.accepted_note_mutation_runner import AcceptedNoteMutationPreparer
from basic_memory.indexing.models import FileIndexResult
from basic_memory.read_cache import ReadCacheChange
from basic_memory.indexing.directory_delete_runner import (
    DirectoryDeleteRejected,
    DirectoryDeleteRejection,
    DirectoryDeleteRejectKind,
    DirectoryFileDeleteEnqueueError,
)
from basic_memory.indexing.note_file_delete_runner import run_note_file_delete
from basic_memory.markdown import EntityParser
from basic_memory.markdown.markdown_processor import MarkdownProcessor
from basic_memory.markdown.note_lock import LOCKED_NOTE_MESSAGE, note_is_locked
from basic_memory.models import Entity, Project
from basic_memory.repository.accepted_note_repositories import AcceptedNoteRepositories
from basic_memory.repository.entity_repository import EntityRepository
from basic_memory.schemas.base import Entity as EntitySchema
from basic_memory.runtime.cleanup import (
    RuntimeDirectoryFileSnapshot,
    RuntimeFileDeleteResult,
    RuntimeGuardedFileDeleteOutcome,
    RuntimeNoteFileDeleteJobRequest,
)
from basic_memory.runtime.storage import (
    RuntimeFileChecksum,
    RuntimeFilePath,
    runtime_content_type_is_markdown,
    runtime_file_path_is_markdown_note,
)
from basic_memory.services.exceptions import FileOperationError
from basic_memory.services.file_service import FileService
from basic_memory.services.note_preparation import (
    NotePreparation,
    NotePreparationDependencies,
    PreparedEntityMove,
    PreparedEntityWrite,
)
from basic_memory.services.note_times import keep_move_times, stamp_note_times
from basic_memory.services.search_service import SearchService

# --- Accepted-Note Mutations ---


@dataclass(frozen=True, slots=True)
class TimestampedNotePreparation(NotePreparation):
    """Note preparation that keeps `created`/`modified` in the accepted bytes (BM-010).

    Trigger: Git hosting, where the Markdown bytes are the only durable state.
    Why: a rebuild from Git otherwise dates every note with the projection time.
    Outcome: create/replace/edit stamp both keys, a plain move leaves them untouched.
    """

    @override
    async def prepare_create_entity_content(
        self,
        schema: EntitySchema,
        *,
        check_storage_exists: bool = True,
        skip_conflict_check: bool = False,
        session: AsyncSession | None = None,
    ) -> PreparedEntityWrite:
        prepared = await super().prepare_create_entity_content(
            schema,
            check_storage_exists=check_storage_exists,
            skip_conflict_check=skip_conflict_check,
            session=session,
        )
        return stamp_note_times(prepared, now=datetime.now().astimezone(), current_markdown=None)

    @override
    async def prepare_update_entity_content(
        self,
        entity: Entity,
        schema: EntitySchema,
        existing_content: str,
        *,
        skip_conflict_check: bool = False,
        replace_frontmatter: bool = False,
        session: AsyncSession | None = None,
    ) -> PreparedEntityWrite:
        prepared = await super().prepare_update_entity_content(
            entity,
            schema,
            existing_content,
            skip_conflict_check=skip_conflict_check,
            replace_frontmatter=replace_frontmatter,
            session=session,
        )
        return stamp_note_times(
            prepared,
            now=datetime.now().astimezone(),
            current_markdown=existing_content,
            fallback_created=entity.created_at,
            fallback_modified=entity.updated_at,
        )

    @override
    async def prepare_edit_entity_content(
        self,
        entity: Entity,
        current_content: str,
        *,
        operation: str,
        content: str,
        section: str | None = None,
        find_text: str | None = None,
        expected_replacements: int = 1,
        replace_subsections: bool = True,
        metadata: dict[str, Any] | None = None,
        skip_conflict_check: bool = False,
        session: AsyncSession | None = None,
    ) -> PreparedEntityWrite:
        prepared = await super().prepare_edit_entity_content(
            entity,
            current_content,
            operation=operation,
            content=content,
            section=section,
            find_text=find_text,
            expected_replacements=expected_replacements,
            replace_subsections=replace_subsections,
            metadata=metadata,
            skip_conflict_check=skip_conflict_check,
            session=session,
        )
        return stamp_note_times(
            prepared,
            now=datetime.now().astimezone(),
            current_markdown=current_content,
            fallback_created=entity.created_at,
            fallback_modified=entity.updated_at,
        )

    @override
    async def prepare_move_entity_content(
        self,
        entity: Entity,
        current_content: str,
        destination_path: str,
        *,
        should_update_permalink: bool | None = None,
        session: AsyncSession | None = None,
    ) -> PreparedEntityMove:
        prepared = await super().prepare_move_entity_content(
            entity,
            current_content,
            destination_path,
            should_update_permalink=should_update_permalink,
            session=session,
        )
        return keep_move_times(
            prepared,
            current_markdown=current_content,
            entity_created_at=entity.created_at,
            entity_updated_at=entity.updated_at,
        )


@dataclass(frozen=True, slots=True)
class LocalAcceptedNotePreparerFactory:
    """Construct prepare-only note semantics for local accepted-note mutations."""

    session_maker: async_sessionmaker[AsyncSession]
    app_config: BasicMemoryConfig
    stamp_note_times: bool = False

    def _create_file_service(self, project: Project) -> tuple[EntityParser, FileService]:
        entity_parser = EntityParser(Path(project.path))
        markdown_processor = MarkdownProcessor(entity_parser, app_config=self.app_config)
        return (
            entity_parser,
            FileService(
                Path(project.path),
                markdown_processor,
                app_config=self.app_config,
            ),
        )

    def create_note_preparer(self, project: Project) -> AcceptedNoteMutationPreparer:
        entity_parser, file_service = self._create_file_service(project)
        entity_repository = EntityRepository(project_id=project.id)
        dependencies = NotePreparationDependencies(
            entity_parser=entity_parser,
            entity_repository=entity_repository,
            file_service=file_service,
            session_maker=self.session_maker,
            app_config=self.app_config,
        )
        if self.stamp_note_times:
            return TimestampedNotePreparation(dependencies)
        return NotePreparation(dependencies)

    async def load_current_file_checksum(
        self,
        project: Project,
        file_path: RuntimeFilePath,
    ) -> RuntimeFileChecksum | None:
        """Return the source checksum observed while accepting a path change."""
        _, file_service = self._create_file_service(project)
        if not await file_service.exists(file_path):
            return None
        return await file_service.compute_checksum(file_path)


LocalAcceptedNoteRepositories = AcceptedNoteRepositories


# --- Current-Note Content Freshening ---


class LocalCurrentNoteEntity(Protocol):
    """Entity fields needed to refresh current markdown before route mutation."""

    @property
    def file_path(self) -> str: ...

    @property
    def content_type(self) -> str: ...


class LocalCurrentNoteEntityRepository(Protocol):
    """Entity lookup needed by the local note-content freshener."""

    async def get_by_external_id(
        self,
        session: AsyncSession,
        external_id: str,
        *,
        load_relations: bool = True,
    ) -> LocalCurrentNoteEntity | None: ...


class LocalCurrentNoteFileService(Protocol):
    """Current file-state access needed before mutating accepted note content."""

    async def exists(
        self,
        path: RuntimeFilePath,
    ) -> bool: ...


class LocalCurrentNoteFileIndexer(Protocol):
    """Single-file indexing capability used by the local note-content freshener."""

    async def index_file(
        self,
        file_path: RuntimeFilePath,
        *,
        source: str,
    ) -> FileIndexResult: ...


@dataclass(frozen=True, slots=True)
class LocalCurrentNoteContentFreshener:
    """Converge directly-edited local markdown before accepted-note mutations."""

    entity_repository: LocalCurrentNoteEntityRepository
    file_service: LocalCurrentNoteFileService
    file_indexer: LocalCurrentNoteFileIndexer
    session_maker: async_sessionmaker[AsyncSession]

    async def freshen_note_content(
        self,
        *,
        project_external_id: str,
        entity_external_id: str,
    ) -> ReadCacheChange:
        del project_external_id

        async with self.session_maker() as session:
            entity = await self.entity_repository.get_by_external_id(
                session,
                entity_external_id,
                load_relations=False,
            )
            if entity is None or not runtime_content_type_is_markdown(entity):
                return ReadCacheChange()
            file_path = entity.file_path

        if not await self.file_service.exists(file_path):
            return ReadCacheChange()

        result = await self.file_indexer.index_file(
            file_path,
            source="note-content-mutation-freshen",
        )
        return result.read_cache_change


# --- Directory-Delete File Cleanup ---


async def check_local_directory_file_locks(
    file_service: FileService,
    files: Sequence[RuntimeDirectoryFileSnapshot],
) -> None:
    """Reject API directory deletion when an existing local note is locked."""
    for snapshot in files:
        if not runtime_file_path_is_markdown_note(snapshot.file_path):
            continue
        # Raw filesystem deletion is outside the lock contract. Missing files do
        # not need protection, and the ordinary index pass reconciles their rows.
        if not await file_service.exists(snapshot.file_path):
            continue
        content = await file_service.read_file_content(snapshot.file_path)
        if note_is_locked(content):
            raise DirectoryDeleteRejected(
                DirectoryDeleteRejection(
                    kind=DirectoryDeleteRejectKind.locked,
                    detail=LOCKED_NOTE_MESSAGE,
                )
            )


@dataclass(frozen=True, slots=True)
class LocalNoteFileDeleteStorage:
    """Adapt local FileService to guarded materialized-note cleanup."""

    file_service: FileService

    async def exists(self, path: RuntimeFilePath) -> bool:
        return await self.file_service.exists(path)

    async def compute_checksum(self, path: RuntimeFilePath) -> RuntimeFileChecksum:
        try:
            return await self.file_service.compute_checksum(path)
        except FileError as exc:
            if isinstance(exc.__context__, FileNotFoundError):
                # Directory cleanup uses the portable guard, where deletion after
                # the existence probe is the same absent-file outcome.
                raise FileNotFoundError(path) from exc
            raise

    async def delete_file_if_matches(
        self,
        path: RuntimeFilePath,
        *,
        expected_checksum: RuntimeFileChecksum,
    ) -> RuntimeGuardedFileDeleteOutcome:
        # The local filesystem has no atomic compare-and-delete, so verify the checksum
        # immediately before deleting. A local race is negligible, but this keeps the runner's
        # guarantee portable: never delete an object that no longer matches (basic-memory-cloud#1618).
        if not await self.file_service.exists(path):
            return RuntimeGuardedFileDeleteOutcome.missing
        try:
            actual_checksum = await self.compute_checksum(path)
        except FileNotFoundError:
            # Disappearance after the existence probe means the object is already gone.
            return RuntimeGuardedFileDeleteOutcome.missing
        # Local files carry a single content checksum, so exact equality identifies the version.
        if actual_checksum != expected_checksum:
            return RuntimeGuardedFileDeleteOutcome.changed
        await self.file_service.delete_file(path)
        return RuntimeGuardedFileDeleteOutcome.deleted


@dataclass(frozen=True, slots=True)
class LocalDirectoryFileDeleteEnqueuer:
    """Run accepted directory-delete file cleanup inline for the local runtime."""

    file_service: FileService

    async def enqueue_directory_file_delete(
        self,
        request: RuntimeNoteFileDeleteJobRequest,
    ) -> RuntimeFileDeleteResult | None:
        # Local cleanup runs inline, so return the guarded delete result; a skipped
        # delete (file changed before cleanup) must not be reported as a success.
        try:
            return await run_note_file_delete(
                request,
                storage=LocalNoteFileDeleteStorage(file_service=self.file_service),
            )
        except (FileError, FileOperationError, OSError) as exc:
            raise DirectoryFileDeleteEnqueueError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class LocalDirectoryDeleteRelationCleanupRefresher:
    """Reindex surviving relation sources after an accepted directory delete.

    Their search_index relation rows went stale when the deleted targets'
    rows cascaded away; reindexing each surviving source drops the danglers.
    """

    session_maker: async_sessionmaker[AsyncSession]
    entity_repository: EntityRepository
    search_service: SearchService

    async def refresh_relation_sources(self, entity_ids: Sequence[int]) -> None:
        unique_entity_ids = sorted(set(entity_ids))
        if not unique_entity_ids:
            return

        async with db.scoped_session(self.session_maker) as session:
            entities = await self.entity_repository.find_by_ids(session, unique_entity_ids)

        # A source deleted between acceptance and this refresh has no search rows
        # left to repair, so missing ids are skipped rather than treated as fatal.
        for entity in entities:
            await self.search_service.index_entity(entity)
