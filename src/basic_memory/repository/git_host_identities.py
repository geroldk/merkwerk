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

The double move is also caught while the common source still owns the note: another
project's published head that took the note by a move from a source the claim names too
(and is not on a chain to the claim) makes the claim a duplicate, whatever SQL says.

Only manifests and commit trailers are read, never note bytes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
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

# Introductions are fixed for a fixed tip and trailers for a commit, so both are cached.
_INTRODUCTION_CACHE_LIMIT = 4096
_MISSING: Any = object()


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


class GitMoveIntentsUnreadable(GitNoteOwnedElsewhere):
    """A caller-chosen UUID while some move intent cannot be read: it may reserve it."""

    def __init__(self, external_id: UUID) -> None:
        ValueError.__init__(
            self,
            f"Note UUID {external_id} cannot be taken now: a cross-project move intent "
            "cannot be read and may reserve it (GET /v2/git/moves lists the file)",
        )
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
        # Note UUID -> the cross-project move that reserves it host-wide.
        self.reservations: dict[UUID, UUID] = {}
        # Intent files that cannot be read (by name), as the last read of bm-moves/ found.
        # Their moves' UUIDs are unknown, so every caller-chosen UUID is refused meanwhile.
        self.unreadable_intents: tuple[str, ...] = ()
        self._wakers: dict[int, Callable[[], None]] = {}
        self._waiting: dict[int, frozenset[int]] = {}
        self._introductions: dict[tuple[str, str, UUID], str | None] = {}
        self._trailers: dict[tuple[str, str], dict[UUID, frozenset[UUID]]] = {}

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
        note_ids = [note.external_id for note in manifest.notes]
        owners = await self.sql_owners(session, project_id, note_ids)
        return await asyncio.to_thread(self._classify, project_id, note_ids, owners, history)

    def _classify(
        self,
        project_id: int,
        note_ids: list[UUID],
        owners: Mapping[UUID, int],
        history: tuple[GitProjectRepository, str],
    ) -> ForeignNotes:
        claimant, tip = history
        # The other projects' published heads: a manifest read each, no note bytes.
        heads = {
            other.project_id: other.repository.published_note_ids()
            for other in self.projects.values()
            if other.project_id != project_id
        }
        deferred: dict[UUID, int] = {}
        duplicates: dict[UUID, int] = {}
        held: dict[UUID, GitHostProject] = {}
        for note, owner_id in sorted(owners.items(), key=lambda item: str(item[0])):
            owner = self.projects.get(owner_id)
            if owner is None:
                duplicates[note] = owner_id
                continue
            # Rule (a): the owner published the release; its SQL follows on its own.
            if note not in heads[owner_id]:
                deferred[note] = owner_id
            else:
                held[note] = owner
        # Trigger: another project's head newly holds a note of this manifest, though its
        #   SQL does not (yet): it synced a peer's move, and its projection waits or lags.
        # Why: two hosts that each moved one note out of one project into two targets
        #   leave two heads that both claim a move from the same source; while the source
        #   still owns the note in SQL, rule (b) alone would let both through, and the
        #   second projection would hit SQL's unique index (and refuse the next start).
        # Outcome: such a rival head makes the claim a duplicate (invalid_incoming on sync).
        rivals: dict[UUID, list[GitHostProject]] = {}
        for note in note_ids:
            if note in duplicates:
                continue
            others = [
                self.projects[other_id]
                for other_id, ids in heads.items()
                if note in ids and other_id != owners.get(note)
            ]
            if others:
                rivals[note] = others
        # Rules (b) and the rival check need each claim's introduction: one walk of the
        # claiming history for all of them (a folder that arrived by 600 single moves is
        # 600 such notes).
        claims = self._claims(claimant, tip, held.keys() | rivals.keys())
        for note, owner in held.items():
            if self._move_in_progress(note, owner, claims[note]):
                deferred[note] = owner.project_id
            else:
                duplicates[note] = owner.project_id
        for note, others in sorted(rivals.items(), key=lambda item: str(item[0])):
            if note in duplicates:
                continue
            if rival := next(
                (other for other in others if self._double_move(note, other, claims[note])),
                None,
            ):
                deferred.pop(note, None)
                duplicates[note] = rival.project_id
        return ForeignNotes(deferred, duplicates)

    def _move_in_progress(self, note: UUID, owner: GitHostProject, claim: frozenset[UUID]) -> bool:
        """Rule (b): the claim's own introduction proves a move the owner will follow."""
        return bool(claim) and not self._double_move(note, owner, claim)

    def _double_move(self, note: UUID, holder: GitHostProject, claim: frozenset[UUID]) -> bool:
        """Whether `holder` took the note by a move from a source the claim names too.

        A holder the claim names, or one a move chain leads from to the claim, will
        release the note once its own move arrives: no double move.
        """
        if not claim or holder.external_id in claim or self._custody_reaches(note, claim, holder):
            return False
        # Trigger: the claim names a source other than the holder, and no project on this
        #   host shows a move chain from the holder to that source.
        # Why: either a chain still travelling (the holder will release once it syncs), or
        #   two hosts each moved the note out of one project: the holder then holds a move
        #   from that same source, and neither side would ever release.
        # Outcome: the double move is a duplicate (named invalid_incoming on sync).
        holder_claim = self._claims(holder.repository, holder.repository.head(), [note])[note]
        return bool(holder_claim & claim)

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
            repository = project.repository
            sources = self._claims(repository, repository.head(), [note])[note]
            if owner.external_id in sources:
                return True
            pending.extend(sources)
        return False

    def _claims(
        self, repository: GitProjectRepository, tip: str, notes: Iterable[UUID]
    ) -> dict[UUID, frozenset[UUID]]:
        """For each note, the move sources on the commit that last introduced it at `tip`.

        Introductions and trailers never change for a fixed tip or commit, so both are
        cached; a waiting project re-classifies on every pass.
        """
        notes = list(notes)
        path = str(repository.path)
        # Read into locals with one lookup each: another thread may clear a full cache
        # between an `in` test and an index, and a cached introduction may be None.
        introductions = {
            note: found
            for note in notes
            if (found := self._introductions.get((path, tip, note), _MISSING)) is not _MISSING
        }
        if missing := [note for note in notes if note not in introductions]:
            introductions |= repository.last_introductions(tip, missing)
            self._remember(
                self._introductions, {(path, tip, note): introductions[note] for note in missing}
            )
        commits = {commit for commit in introductions.values() if commit is not None}
        trailers = {
            commit: found
            for commit in commits
            if (found := self._trailers.get((path, commit))) is not None
        }
        if unread := [commit for commit in commits if commit not in trailers]:
            trailers |= repository.moved_notes_of(unread)
            self._remember(self._trailers, {(path, commit): trailers[commit] for commit in unread})
        return {
            note: trailers[commit].get(note, frozenset()) if commit is not None else frozenset()
            for note, commit in introductions.items()
        }

    @staticmethod
    def _remember[K, V](cache: dict[K, V], entries: Mapping[K, V]) -> None:
        if len(cache) + len(entries) > _INTRODUCTION_CACHE_LIMIT:
            cache.clear()
        cache.update(entries)

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

    def require_intents_readable(self, note_ids: Iterable[UUID]) -> None:
        """Fail closed for a caller-chosen UUID while an intent file cannot be read.

        Random UUIDs of new notes cannot collide with a move's; a chosen one (a PUT by
        UUID, a new cross-project move) might be exactly the unreadable move's note.
        """
        if self.unreadable_intents and (wanted := sorted(note_ids, key=str)):
            raise GitMoveIntentsUnreadable(wanted[0])

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
