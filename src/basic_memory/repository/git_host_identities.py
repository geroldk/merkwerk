"""Who owns a note UUID on one Git host: SQL rows, published heads and move proofs (I1).

A note keeps its UUID when it moves to another project (W2: draft in the target, delete
in the source, commit in the target). Between those steps, and on a peer that syncs the
two projects in any order, one UUID can appear in two projects for a while. These rules
tell such a moment from a real duplicate:

(a) the owner's published head no longer holds the note: its SQL lags and its own
    projection will release the UUID -> wait;
(b) the owner's head still holds it, but the commit that last introduced the note into
    the claiming history carries BM's move trailer for it -> wait, the move is still
    travelling; unless the owner itself took the note by a move from the same source
    the claim names, and no move chain leads from the owner to the claim: two hosts
    moved one note to two projects (a double move) -> duplicate;
(c) otherwise -> a real duplicate, refused as before.

Only manifests and commit trailers are read, never note bytes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from basic_memory.models import Entity
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_projection_repository import (
    GitProjectionDeferred,
    GitWaitingNote,
)
from basic_memory.schemas.git_project import GitProjectManifest

# Introductions are fixed for a fixed tip, so they are cached by (repository, tip, note).
_INTRODUCTION_CACHE_LIMIT = 4096


@dataclass(frozen=True, slots=True)
class GitHostProject:
    project_id: int
    external_id: UUID
    name: str
    repository: GitProjectRepository


@dataclass(frozen=True, slots=True)
class ForeignNotes:
    """Note UUIDs of a manifest that SQL places in another project, by owner project id."""

    deferred: Mapping[UUID, int]
    duplicates: Mapping[UUID, int]


class GitNoteOwnedElsewhere(ValueError):
    """A UUID this project would newly publish is still another project's (or reserved)."""

    def __init__(self, external_id: UUID, owner: str) -> None:
        super().__init__(f"Note UUID {external_id} is still owned by {owner}")
        self.external_id = external_id


class GitHostIdentities:
    """The host-wide view of note ownership, shared by every project runtime.

    It also carries the wake-ups between projects: a project whose projection waits
    for an owner is woken when that owner's content stage completes.
    """

    def __init__(self, projects: Iterable[GitHostProject]) -> None:
        self.projects: dict[int, GitHostProject] = {
            project.project_id: project for project in projects
        }
        self._by_external_id = {project.external_id: project for project in self.projects.values()}
        # Note UUID -> the cross-project move that reserves it host-wide (filled in part B).
        self.reservations: dict[UUID, UUID] = {}
        self._wakers: dict[int, Callable[[], None]] = {}
        self._waiting: dict[int, frozenset[int]] = {}
        self._introductions: dict[tuple[str, str, UUID], str | None] = {}

    def describe(self, project_id: int) -> str:
        project = self.projects.get(project_id)
        return f"project '{project.name}' ({project.external_id})" if project else "another project"

    # --- Classification ---

    async def sql_owners(
        self, session: AsyncSession, project_id: int, note_ids: Iterable[UUID]
    ) -> dict[UUID, int]:
        """The other projects whose SQL rows hold these UUIDs."""
        wanted = [str(note) for note in note_ids]
        if not wanted:
            return {}
        rows = await session.execute(
            select(Entity.external_id, Entity.project_id).where(
                Entity.external_id.in_(wanted), Entity.project_id != project_id
            )
        )
        return {UUID(external_id): owner for external_id, owner in rows if external_id}

    async def classify_foreign(
        self,
        session: AsyncSession,
        *,
        project_id: int,
        manifest: GitProjectManifest,
        history: tuple[GitProjectRepository, str],
    ) -> ForeignNotes:
        """Sort the manifest's foreign UUIDs into "wait" and "real duplicate".

        `history` is the claiming repository and the commit whose manifest this is (a
        published head, a validated incoming tip or a prepared merge).
        """
        owners = await self.sql_owners(
            session, project_id, (note.external_id for note in manifest.notes)
        )
        if not owners:
            return ForeignNotes({}, {})
        return await asyncio.to_thread(self._classify, owners, history)

    def _classify(
        self, owners: Mapping[UUID, int], history: tuple[GitProjectRepository, str]
    ) -> ForeignNotes:
        claimant, tip = history
        heads: dict[int, frozenset[UUID]] = {}
        deferred: dict[UUID, int] = {}
        duplicates: dict[UUID, int] = {}
        for note, owner_id in sorted(owners.items(), key=lambda item: str(item[0])):
            owner = self.projects.get(owner_id)
            if owner is None:
                duplicates[note] = owner_id
                continue
            if owner_id not in heads:
                heads[owner_id] = owner.repository.published_note_ids()
            # Rule (a): the owner published the release; its SQL follows on its own.
            if note not in heads[owner_id] or self._move_in_progress(note, owner, claimant, tip):
                deferred[note] = owner_id
            else:
                duplicates[note] = owner_id
        return ForeignNotes(deferred, duplicates)

    def _move_in_progress(
        self, note: UUID, owner: GitHostProject, claimant: GitProjectRepository, tip: str
    ) -> bool:
        """Rule (b): the claim's own introduction proves a move the owner will follow."""
        claim = self._introduced_from(claimant, tip, note)
        if not claim:
            return False
        if owner.external_id in claim or self._custody_reaches(note, claim, owner):
            return True
        # Trigger: the claim names a source other than the owner, and no project on this
        #   host shows a move chain from the owner to that source.
        # Why: either a chain still travelling (the owner will release once it syncs), or
        #   two hosts each moved the note out of one project: the owner then holds a move
        #   from that same source, and neither side would ever release.
        # Outcome: the double move is a duplicate (named invalid_incoming on sync).
        owner_claim = self._introduced_from(owner.repository, owner.repository.head(), note)
        return not (owner_claim & claim)

    def _custody_reaches(self, note: UUID, claim: frozenset[UUID], owner: GitHostProject) -> bool:
        """Whether the projects the claim names took the note, hop by hop, from the owner."""
        seen: set[UUID] = set()
        pending = list(claim)
        while pending:
            source = pending.pop()
            if source in seen or source == owner.external_id:
                continue
            seen.add(source)
            project = self._by_external_id.get(source)
            if project is None:
                continue
            sources = self._introduced_from(project.repository, project.repository.head(), note)
            if owner.external_id in sources:
                return True
            pending.extend(sources)
        return False

    def _introduced_from(
        self, repository: GitProjectRepository, tip: str, note: UUID
    ) -> frozenset[UUID]:
        """The move sources named on the commit that last introduced `note` at `tip`."""
        key = (str(repository.path), tip, note)
        if key not in self._introductions:
            if len(self._introductions) >= _INTRODUCTION_CACHE_LIMIT:
                self._introductions.clear()
            self._introductions[key] = repository.last_introduction(tip, note)
        introduction = self._introductions[key]
        return repository.move_sources(introduction, note) if introduction else frozenset()

    def deferral(self, foreign: ForeignNotes) -> GitProjectionDeferred:
        return GitProjectionDeferred(
            [
                GitWaitingNote(
                    external_id=note,
                    owner_project_id=owner_id,
                    owner_external_id=self.projects[owner_id].external_id,
                    owner_name=self.projects[owner_id].name,
                )
                for note, owner_id in foreign.deferred.items()
            ]
        )

    def duplicates_message(self, prefix: str, foreign: ForeignNotes) -> str:
        return f"{prefix}: " + "; ".join(
            f"{note} (owned by {self.describe(owner_id)})"
            for note, owner_id in foreign.duplicates.items()
        )

    # --- Guards for new UUIDs (transaction commit, PUT upsert create) ---

    def require_unreserved(self, note_ids: Iterable[UUID], *, holder: UUID | None) -> None:
        """Refuse a UUID that an open cross-project move reserves, except for that move.

        Between the source delete and the target commit a moving note is in no project;
        a PUT by UUID or another draft must not take its UUID meanwhile.
        """
        for note in sorted(note_ids, key=str):
            move = self.reservations.get(note)
            if move is not None and move != holder:
                raise GitNoteOwnedElsewhere(note, f"open cross-project move {move}")

    async def require_unowned(
        self,
        session: AsyncSession,
        *,
        project_id: int,
        note_ids: Iterable[UUID],
        holder: UUID | None = None,
    ) -> None:
        """Refuse a UUID that another project holds in SQL or its published head.

        A UUID an open cross-project move reserves is refused too, except for that move
        (`holder`). Unlike projection, nothing waits here: a commit that publishes a UUID
        another project still owns would block that project's twin for good.
        """
        wanted = frozenset(note_ids)
        self.require_unreserved(wanted, holder=holder)
        if not wanted:
            return
        if owners := await self.sql_owners(session, project_id, wanted):
            note = min(owners, key=str)
            raise GitNoteOwnedElsewhere(note, self.describe(owners[note]))
        for other in self.projects.values():
            if other.project_id == project_id:
                continue
            if held := wanted & await asyncio.to_thread(other.repository.published_note_ids):
                raise GitNoteOwnedElsewhere(min(held, key=str), self.describe(other.project_id))

    # --- Wake-ups between projects ---

    def register(self, project_id: int, wake: Callable[[], None]) -> None:
        self._wakers[project_id] = wake

    def wait_for(self, project_id: int, owners: Iterable[int]) -> None:
        self._waiting[project_id] = frozenset(owners)

    def settled(self, project_id: int) -> None:
        """`project_id` completed its content stage: wake the projects waiting for it."""
        self._waiting.pop(project_id, None)
        for waiting, owners in list(self._waiting.items()):
            if project_id in owners and (wake := self._wakers.get(waiting)) is not None:
                wake()

    def nudge_waiting(self) -> None:
        """Wake every waiting project once (after all runtimes entered at startup)."""
        for waiting in list(self._waiting):
            if (wake := self._wakers.get(waiting)) is not None:
                wake()
