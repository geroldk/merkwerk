"""Guarded identity/path changes and durable old-file retirement permissions."""

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from basic_memory.models import Entity, GitFileRetirement, NoteContent
from basic_memory.models.knowledge import file_path_conflict_columns
from basic_memory.repository.entity_repository import AcceptedPendingEntityWrite
from basic_memory.repository.git_projection_repository import (
    GitProjectionConflict,
    GitProjectionContentBasis,
)
from basic_memory.repository.note_content_repository import NoteContentVersionConflict
from basic_memory.utils import ensure_timezone_aware


@dataclass(frozen=True, slots=True)
class GitLocalNote:
    entity_id: int
    external_id: UUID
    file_path: str
    content: GitProjectionContentBasis | None
    file_checksum: str | None


@dataclass(frozen=True, slots=True)
class GitPreparedNote:
    fields: AcceptedPendingEntityWrite
    content: str
    checksum: str


@dataclass(frozen=True, slots=True)
class GitRetiredFile:
    file_path: str
    retirement_id: str
    expected_checksums: tuple[str, ...]


class GitLayoutRepository:
    def __init__(self, project_id: int) -> None:
        self.project_id = project_id

    async def local_notes(self, session: AsyncSession) -> tuple[GitLocalNote, ...]:
        rows = await session.execute(
            select(Entity, NoteContent)
            .outerjoin(NoteContent, Entity.id == NoteContent.entity_id)
            .where(Entity.project_id == self.project_id)
            .order_by(Entity.id)
        )
        notes = []
        for entity, content in rows:
            if content is not None and (
                content.external_id != entity.external_id
                or content.project_id != self.project_id
                or content.file_path != entity.file_path
            ):
                raise GitProjectionConflict("Note content identity does not match its entity")
            notes.append(
                GitLocalNote(
                    entity.id,
                    UUID(entity.external_id),
                    entity.file_path,
                    GitProjectionContentBasis(
                        entity.id, content.db_version, content.db_checksum, content.markdown_content
                    )
                    if content is not None
                    else None,
                    content.file_checksum if content is not None else None,
                )
            )
        return tuple(notes)

    async def apply(
        self,
        session: AsyncSession,
        *,
        expected: tuple[GitLocalNote, ...],
        target: Mapping[UUID, GitPreparedNote],
    ) -> tuple[int, int]:
        """Relocate proven notes atomically in SQL; files/indexes stay asynchronous.

        One short layout transaction is needed for the unique path/permalink
        permutation. It adds no row locks and performs no filesystem/Git I/O.
        NoteContent CAS operations precede Entity changes, as in normal writes.
        """
        if await self.local_notes(session) != expected:
            raise NoteContentVersionConflict("Local content advanced during Git layout projection")
        moved = [
            note
            for note in expected
            if note.external_id in target
            and target[note.external_id].fields.file_path != note.file_path
        ]
        removed = [note for note in expected if note.external_id not in target]
        for note in (*moved, *removed):
            basis = note.content
            if basis is None:
                raise GitProjectionConflict("Cannot retire a note without proven accepted content")
            await self.retain_old_file(session, note)
            condition = (
                NoteContent.entity_id == note.entity_id,
                NoteContent.project_id == self.project_id,
                NoteContent.db_version == basis.db_version,
                NoteContent.db_checksum == basis.checksum,
            )
            if note.external_id in target:
                prepared = target[note.external_id]
                claimed = await session.scalar(
                    update(NoteContent)
                    .where(*condition)
                    .values(
                        file_path=prepared.fields.file_path,
                        markdown_content=prepared.content,
                        db_checksum=prepared.checksum,
                        db_version=basis.db_version + 1,
                        file_version=None,
                        file_checksum=None,
                        file_write_status="pending",
                        last_materialization_error=None,
                        last_source="git-projection",
                        updated_at=prepared.fields.updated_at,
                    )
                    .returning(NoteContent.entity_id)
                )
            else:
                claimed = await session.scalar(
                    delete(NoteContent).where(*condition).returning(NoteContent.entity_id)
                )
            if claimed is None:
                raise NoteContentVersionConflict("Local content advanced before Git layout claim")
        for note in removed:
            entity = await session.get(Entity, note.entity_id)
            if entity is None:
                raise GitProjectionConflict("Retired entity disappeared during layout projection")
            await session.delete(entity)
        await session.flush()
        # SQL-only placeholders free unique names for swaps. None are ever
        # committed or materialized; every moved note keeps its UUID/numeric ID.
        for note in moved:
            await session.execute(
                update(Entity)
                .where(Entity.id == note.entity_id)
                .values(
                    file_path=f".basic-memory/moving/{note.external_id}",
                    permalink=None,
                )
            )
        for note in moved:
            fields = target[note.external_id].fields
            # A Core UPDATE skips Entity's validators. Set what they would: the conflict
            # keys of the new path (else a later write at the vacated path is refused as
            # "equivalent") and times as local instants (else SQLite keeps an aware UTC
            # value as naive wall time, and the note reads back shifted by the offset).
            await session.execute(
                update(Entity)
                .where(Entity.id == note.entity_id)
                .values(
                    file_path=fields.file_path,
                    **file_path_conflict_columns(fields.file_path),
                    permalink=fields.permalink,
                    title=fields.title,
                    note_type=fields.note_type,
                    entity_metadata=fields.entity_metadata,
                    content_type=fields.content_type,
                    created_at=ensure_timezone_aware(fields.created_at).astimezone(),
                    updated_at=ensure_timezone_aware(fields.updated_at).astimezone(),
                    checksum=None,
                    mtime=None,
                    size=None,
                )
            )
        await session.flush()
        return len(moved), len(removed)

    async def retain_old_file(self, session: AsyncSession, note: GitLocalNote) -> None:
        """Retain cleanup evidence before an accepted move/delete commits in SQL.

        The caller must prove the source UUID/path/content in published Git first.
        This survives deletion of Entity and its cascading NoteContent row.
        """
        basis = note.content
        if basis is None:
            raise GitProjectionConflict("Cannot retire a note without proven accepted content")
        retirement = await session.get(GitFileRetirement, (self.project_id, note.file_path))
        checksums = {basis.checksum}
        if note.file_checksum is not None:
            checksums.add(note.file_checksum)
        if retirement is None:
            session.add(
                GitFileRetirement(
                    project_id=self.project_id,
                    file_path=note.file_path,
                    retirement_id=str(uuid4()),
                    expected_checksums=sorted(checksums),
                )
            )
        else:
            retirement.retirement_id = str(uuid4())
            retirement.expected_checksums = sorted(checksums | set(retirement.expected_checksums))

    async def retirements(self, session: AsyncSession) -> tuple[GitRetiredFile, ...]:
        rows = await session.scalars(
            select(GitFileRetirement)
            .where(GitFileRetirement.project_id == self.project_id)
            .order_by(GitFileRetirement.file_path)
        )
        return tuple(
            GitRetiredFile(row.file_path, row.retirement_id, tuple(row.expected_checksums))
            for row in rows
        )

    async def complete_retirement(self, session: AsyncSession, retirement: GitRetiredFile) -> None:
        await session.execute(
            delete(GitFileRetirement).where(
                GitFileRetirement.project_id == self.project_id,
                GitFileRetirement.file_path == retirement.file_path,
                GitFileRetirement.retirement_id == retirement.retirement_id,
            )
        )
