"""Restore published Git identities/content into the existing local SQL model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from basic_memory.models import Entity, NoteContent, Project
from basic_memory.repository.entity_repository import AcceptedPendingEntityWrite, EntityRepository
from basic_memory.repository.note_content_repository import (
    AcceptedNoteContentWrite,
    NoteContentRepository,
    NoteContentVersionConflict,
)
from basic_memory.schemas.git_project import GitProjectManifest
from basic_memory.utils import generate_permalink


class GitProjectionConflict(ValueError):
    """Local identities or content cannot be safely adopted by this snapshot."""


class GitProjectionSuperseded(GitProjectionConflict):
    """The target is no longer the published head; the work done for it stays valid.

    Nothing is wrong with the project: a newer commit arrived. The worker starts the
    next pass for the newer head at once, without counting a failure or backing off.
    """


@dataclass(frozen=True, slots=True)
class GitProjectionContentBasis:
    entity_id: int
    db_version: int
    checksum: str
    markdown_content: str


async def register_git_project(
    session: AsyncSession, manifest: GitProjectManifest, *, name: str, path: Path
) -> Project:
    """Explicitly register a clone's portable UUID with a local name and root."""
    path = path.resolve()
    existing = await session.scalar(
        select(Project).where(Project.external_id == str(manifest.project_external_id))
    )
    if existing is not None:
        if existing.name != name or Path(existing.path).resolve() != path:
            raise GitProjectionConflict(
                "Project UUID is already bound to another local root or name"
            )
        return existing
    collision = await session.scalar(
        select(Project).where(
            or_(
                Project.name == name,
                Project.path == str(path),
                Project.permalink == generate_permalink(name),
            )
        )
    )
    if collision is not None:
        raise GitProjectionConflict("Local project name or root is already owned by another UUID")
    project = Project(
        external_id=str(manifest.project_external_id),
        name=name,
        path=str(path),
        is_active=True,
        is_default=False,
    )
    session.add(project)
    await session.flush()
    return project


class GitProjectionRepository:
    def __init__(self, project_id: int) -> None:
        self.project_id = project_id
        self.entities = EntityRepository(project_id)
        self.contents = NoteContentRepository(project_id)

    async def validate_project(
        self, session: AsyncSession, manifest: GitProjectManifest
    ) -> Project:
        project = await session.get(Project, self.project_id)
        if project is None or project.external_id != str(manifest.project_external_id):
            raise GitProjectionConflict("Git snapshot belongs to a different project UUID")
        return project

    async def _find_identity(
        self, session: AsyncSession, external_id: UUID, path: str
    ) -> Entity | None:
        entity = await session.scalar(select(Entity).where(Entity.external_id == str(external_id)))
        if entity is not None and (
            entity.project_id != self.project_id or entity.file_path != path
        ):
            raise GitProjectionConflict("Note UUID is already owned by another project or path")
        owner = await session.scalar(
            select(Entity.external_id).where(
                Entity.project_id == self.project_id, Entity.file_path == path
            )
        )
        if owner is not None and owner != str(external_id):
            raise GitProjectionConflict("Note path is already owned by another UUID")
        return entity

    async def read_basis(
        self, session: AsyncSession, external_id: UUID, path: str
    ) -> GitProjectionContentBasis | None:
        entity = await self._find_identity(session, external_id, path)
        if entity is None:
            return None
        content = await self.contents.get_by_entity_id(session, entity.id)
        if content is None:
            return None
        if (
            content.external_id != str(external_id)
            or content.project_id != self.project_id
            or content.file_path != path
        ):
            raise GitProjectionConflict("Note content identity does not match its entity")
        return GitProjectionContentBasis(
            entity.id, content.db_version, content.db_checksum, content.markdown_content
        )

    async def restore_note(
        self,
        session: AsyncSession,
        *,
        external_id: UUID,
        fields: AcceptedPendingEntityWrite,
        markdown_content: str,
        checksum: str,
        expected: GitProjectionContentBasis | None,
    ) -> bool:
        if fields.external_id != str(external_id):
            raise GitProjectionConflict("Prepared fields have a different note UUID")
        current = await self.read_basis(session, external_id, fields.file_path)
        if current != expected:
            raise NoteContentVersionConflict("Local content advanced during Git projection")
        if (
            current is not None
            and current.markdown_content == markdown_content
            and current.checksum == checksum
        ):
            # An unchanged note keeps its file state. The Git files stage checks every
            # note's file on disk itself, so a lost file is rewritten without marking
            # the whole project pending at each commit (F2).
            return False
        entity = await self._find_identity(session, external_id, fields.file_path)
        if entity is None:
            entity = await self.entities.create_pending_accepted_entity(session, fields)
        if current is None:
            # An observed absence is guarded by INSERT's unique key. The normal
            # accept_write(version=1) upsert could otherwise replace a content
            # row that appeared after our read with this older first version.
            await self.contents.create(
                session,
                NoteContent(
                    entity_id=entity.id,
                    markdown_content=markdown_content,
                    db_version=1,
                    db_checksum=checksum,
                    last_source="git-projection",
                    file_write_status="pending",
                    updated_at=fields.updated_at,
                ),
            )
        else:
            await self.contents.accept_write(
                session,
                AcceptedNoteContentWrite(
                    entity_id=entity.id,
                    markdown_content=markdown_content,
                    db_version=current.db_version + 1,
                    db_checksum=checksum,
                    last_source="git-projection",
                    updated_at=fields.updated_at,
                ),
            )
        # Claim NoteContent before touching an existing Entity, as in ordinary
        # accepted writes. Graph/vector work stays outside this short transaction.
        entity.title = fields.title
        entity.note_type = fields.note_type
        entity.entity_metadata = fields.entity_metadata
        entity.content_type = fields.content_type
        entity.permalink = fields.permalink
        entity.created_at = fields.created_at
        entity.updated_at = fields.updated_at
        await session.flush()
        return True
