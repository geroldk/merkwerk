"""Opt-in Git acceptance for a lifecycle-owned, externally controlled local project.

Configured API/MCP hosts select this executor and its explicit transaction service.
The composition root suppresses legacy writers/materializers/imports before admission.
"""

import asyncio
import hashlib
import subprocess
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from typing import assert_never
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.file_utils import compute_checksum
from basic_memory.index.git_projection import GitProjectionPipeline
from basic_memory.index.git_projection_worker import GitProjectionWorker
from basic_memory.indexing.accepted_note_mutation_runner import (
    AcceptedNoteCreateMutation,
    AcceptedNoteDeleteMutation,
    AcceptedNoteEditMutation,
    AcceptedNoteMoveMutation,
    AcceptedNoteMutationDependencies,
    AcceptedNoteUpdateMutation,
    run_accepted_note_create,
    run_accepted_note_delete,
    run_accepted_note_edit,
    run_accepted_note_move,
    run_accepted_note_update,
)
from basic_memory.models import Entity, NoteContent
from basic_memory.repository.git_layout_repository import GitLayoutRepository
from basic_memory.repository.git_graph_inputs import git_graph_identities_unchanged
from basic_memory.repository.git_progress_repository import (
    GitProgressRepository,
    invalidate_git_graphs,
)
from basic_memory.repository.git_project_repository import (
    GitPublication,
    GitPublicationConflict,
    GitPublicationError,
    GitSnapshot,
    settle_git_publication,
)
from basic_memory.repository.git_projection_repository import (
    GitProjectionConflict,
    GitProjectionDeferred,
)
from basic_memory.repository.git_transaction_repository import GitTransactionError
from basic_memory.runtime.project_partition import RuntimeProjectNoteOperation
from basic_memory.services.note_content_writes import (
    AcceptedNoteChange,
    AcceptedNoteMutationRequest,
    NoteContentMutationServiceError,
    NoteContentProjectWaiting,
    NoteContentPublicationConflict,
)
from basic_memory.services.git_directory_moves import (
    GitDirectoryMoveConflict,
    GitDirectoryMoveDisk,
    GitDirectoryMoveOutcome,
    GitDirectoryMovePlan,
    GitDirectoryMoveRefused,
    disk_conflicts,
    normalize_note_directory,
    plan_git_directory_move,
    unpublished_source_files,
)
from basic_memory.services.git_transactions import GitTransactionService, validate_git_draft
from basic_memory.services.note_times import stamp_markdown_times, wanted_note_times
from basic_memory.schemas.git_sync import GitSyncRemote
from basic_memory.services.git_sync import GitSyncService


class GitProjectRuntime:
    """Capture Git basis before real BM preparation; publish before SQL commit.

    Only short implicit mutation/preparation work holds the local admission gate.
    Agent draft thinking must never run inside accept(). Git CAS still arbitrates
    external publications. The worker and this executor share only the SQL content
    gate; vector work never holds it.
    """

    def __init__(
        self,
        pipeline: GitProjectionPipeline,
        dependencies: AcceptedNoteMutationDependencies,
        *,
        poll_interval: float = 5,
        transaction_idle_seconds: int = 1800,
        transaction_retention_seconds: int = 604800,
        sync_remote: GitSyncRemote | None = None,
        sync_merge: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.dependencies = dependencies
        self.gate = asyncio.Lock()
        self.worker = GitProjectionWorker(
            pipeline, poll_interval=poll_interval, content_gate=self.gate
        )
        self._running = False
        self._sql_commit: str | None = None
        self.last_publication: GitPublication | None = None
        self.transactions = GitTransactionService(
            self,
            idle_seconds=transaction_idle_seconds,
            retention_seconds=transaction_retention_seconds,
        )
        self.sync = GitSyncService(self, sync_remote, merge=sync_merge)

    @asynccontextmanager
    async def run(self) -> AsyncIterator["GitProjectRuntime"]:
        async with self.worker.run(recover_content_before_yield=True):
            self._sql_commit = self.worker.startup_content_commit
            self._running = True
            try:
                await self.transactions.maintain()
                await self.sync.recover()
                yield self
            finally:
                self._running = False
                # Bounded transfers may outlive their caller. Drain them before
                # another process can acquire this project's repository ownership.
                async with self.sync.lock:
                    pass
                # An ASGI host must drain request tasks before exiting its lifespan.
                # Also wait for any admitted mutation before releasing ownership.
                async with self.gate:
                    pass

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[None]:
        """Short call ownership; explicit drafts release it between agent calls."""
        if not self._running:
            raise GitProjectionConflict("Git project is not accepting mutations")
        async with self.gate:
            if not self._running:
                raise GitProjectionConflict("Git project stopped before mutation admission")
            await self.pipeline.require_registration()
            yield

    async def accept(self, request: AcceptedNoteMutationRequest) -> AcceptedNoteChange:
        if request.project_external_id != self.pipeline.project_external_id:
            raise GitProjectionConflict("Mutation belongs to a different Git project")
        async with self.admission():
            repository = self.pipeline.repository
            # This immutable basis is captured before any runner/preparer reads SQL.
            base = await asyncio.to_thread(repository.snapshot)
            if self._sql_commit != base.commit_id:
                try:
                    await self.pipeline.apply("content", base.commit_id)
                except GitProjectionDeferred as deferral:
                    # Trigger: notes of this head wait for another project to release them.
                    # Why: SQL must match the basis exactly before any write here, and it
                    #   lacks the waiting notes; this is a wait, not "SQL pending" (503).
                    # Outcome: 409 kind=deferred naming the notes and their owners.
                    raise NoteContentProjectWaiting(
                        await deferral_detail(deferral, self.pipeline.session_maker)
                    ) from deferral
                self._sql_commit = base.commit_id
            layout = GitLayoutRepository(self.pipeline.project_id)
            publication: GitPublication | None = None
            try:
                async with db.scoped_session(self.pipeline.session_maker) as session:
                    before = await layout.local_notes(session)
                    if {
                        note.file_path: note.external_id for note in before
                    } != base.manifest.note_ids:
                        raise GitProjectionConflict("SQL content is outside the captured Git basis")
                    for note in before:
                        if (
                            note.content is None
                            or note.content.markdown_content != base.files[note.file_path]
                            or note.content.checksum
                            != await compute_checksum(note.content.markdown_content)
                        ):
                            raise GitProjectionConflict(
                                "SQL content is outside the captured Git basis"
                            )
                    match request:
                        case AcceptedNoteCreateMutation():
                            result = await run_accepted_note_create(
                                session, request=request, dependencies=self.dependencies
                            )
                        case AcceptedNoteUpdateMutation():
                            result = await run_accepted_note_update(
                                session, request=request, dependencies=self.dependencies
                            )
                        case AcceptedNoteEditMutation():
                            result = await run_accepted_note_edit(
                                session, request=request, dependencies=self.dependencies
                            )
                        case AcceptedNoteMoveMutation():
                            result = await run_accepted_note_move(
                                session, request=request, dependencies=self.dependencies
                            )
                        case AcceptedNoteDeleteMutation():
                            result = await run_accepted_note_delete(
                                session, request=request, dependencies=self.dependencies
                            )
                        case _:  # pragma: no cover - exhaustiveness is checked by ty
                            assert_never(request)
                    change = result.change.project_change
                    if change is None:
                        # Already absent DELETE has no effective content mutation.
                        return result.change
                    files, note_ids = dict(base.files), base.manifest.note_ids
                    note_id = UUID(change.note_external_id)
                    if change.operation in {
                        RuntimeProjectNoteOperation.moved,
                        RuntimeProjectNoteOperation.deleted,
                    }:
                        source_path = change.previous_file_path or change.file_path
                        source = next(
                            (note for note in before if note.external_id == note_id), None
                        )
                        if source is None or note_ids.get(source_path) != note_id:
                            raise GitProjectionConflict("Mutation source is outside its Git basis")
                        await layout.retain_old_file(session, source)
                        del files[source_path]
                        del note_ids[source_path]
                    if change.operation is not RuntimeProjectNoteOperation.deleted:
                        content = await session.get(NoteContent, change.entity_id)
                        if (
                            content is None
                            or content.external_id != str(note_id)
                            or content.db_checksum
                            != await compute_checksum(content.markdown_content)
                        ):
                            raise GitProjectionConflict(
                                "Accepted mutation has inconsistent content"
                            )
                        files[change.file_path] = content.markdown_content
                        note_ids[change.file_path] = note_id
                    if not git_graph_identities_unchanged(
                        base.files, base.manifest.note_ids, files, note_ids
                    ):
                        await invalidate_git_graphs(session)
                    prepared = await asyncio.to_thread(
                        repository.prepare,
                        transaction_id=uuid4(),
                        base_commit=base.commit_id,
                        files=files,
                        note_ids=note_ids,
                    )
                    publication = await settle_git_publication(repository, prepared)
                    self.last_publication = publication
                self._sql_commit = publication.commit_id
                # Legacy route materialization must not run after this executor.
                # The lifecycle worker alone owns files/graph/FTS/vectors.
                return replace(
                    result.change,
                    materialization=None,
                    file_delete=None,
                    relation_cleanup_entity_ids=frozenset(),
                )
            except GitPublicationConflict as error:
                raise NoteContentPublicationConflict(
                    {
                        "message": "Git project advanced during preparation",
                        "base_commit": error.expected,
                        "source_commit": error.actual,
                    },
                ) from error
            except Exception as error:
                if publication is not None:
                    raise NoteContentMutationServiceError(
                        503,
                        {
                            "message": "Git published; SQL pending. Check status before retry.",
                            "published_commit": publication.commit_id,
                        },
                    ) from error
                raise
            finally:
                # Even a cancelled/lost publication response or failed SQL commit
                # must wake durable Git recovery. Polling covers process death.
                self.worker.notify()

    # --- Folder moves (V1) ---

    async def move_directory(self, source: str, destination: str) -> GitDirectoryMoveOutcome:
        """Move every note under `source` to `destination` as ONE published commit.

        This follows accept(): the immutable basis is captured inside the admission, every
        check runs before anything changes, and SQL follows the publication before the gate
        opens. There is no transaction record (a departure from the ENTWURF sketch): no
        caller could resume this one-shot, and Git's compare-and-swap already makes the
        publication all or nothing. Permalinks, ids and created/modified stay as they were.
        """
        # Trigger: the host is configured to give a moved note a new permalink.
        # Why: a folder move keeps every permalink, so single and folder moves would
        #   silently disagree.
        # Outcome: refused before anything is read, until the setting is off.
        if self.dependencies.move_policy.update_permalinks_on_move:
            raise GitDirectoryMoveRefused(
                409,
                "unsupported_config",
                "This host gives moved notes new permalinks (update_permalinks_on_move); "
                "a folder move in one commit keeps them, so it is refused.",
            )
        try:
            async with self.admission():
                return await self._move_directory_admitted(source, destination)
        except GitProjectionConflict as error:
            # Every projection conflict after publication is answered as 503 below, so
            # one here (admission, or catching SQL up to the basis) means nothing moved.
            raise GitDirectoryMoveRefused(
                409, "not_accepting", f"{error}. Nothing was moved."
            ) from error
        finally:
            # A cancelled or lost answer may still have published; wake durable recovery.
            self.worker.notify()

    async def _move_directory_admitted(
        self, source: str, destination: str
    ) -> GitDirectoryMoveOutcome:
        repository = self.pipeline.repository
        base = await asyncio.to_thread(repository.snapshot)
        if self._sql_commit != base.commit_id:
            await self.pipeline.apply("content", base.commit_id)
            self._sql_commit = base.commit_id
        disk = await self._directory_move_disk(base)

        # --- Plan and refusals: nothing has changed yet ---
        try:
            plan = plan_git_directory_move(base.files, base.manifest.note_ids, source, destination)
        except GitDirectoryMoveRefused as refused:
            # Trigger: no published note is under the source.
            # Why: a folder holding only attachments must be refused by name, not "missing".
            # Outcome: files found there are listed; otherwise the folder is not found.
            if refused.reason == "not_found":
                directory = normalize_note_directory(source, role="source")
                if stray := await _scan_disk(unpublished_source_files, disk, directory):
                    raise _unclean_source(stray) from refused
            raise
        unclean, taken = await _scan_disk(disk_conflicts, disk, plan)
        if unclean:
            raise _unclean_source(unclean)
        if taken:
            raise GitDirectoryMoveRefused(
                409,
                "occupied",
                f"{len(taken)} destination path(s) are taken on disk: "
                + "; ".join(conflict.describe() for conflict in taken),
                taken,
            )
        files, note_ids = await self._moved_tree(base, plan)
        try:
            await validate_git_draft(
                {move.file_path: files[move.file_path] for move in plan.moves},
                self.pipeline.file_service.base_path,
            )
        except GitTransactionError as error:
            raise GitDirectoryMoveRefused(
                422, "invalid_draft", f"{error}. Nothing was moved."
            ) from error

        # --- Publication: one commit, then SQL before the gate opens ---
        transaction_id = uuid4()
        try:
            try:
                prepared = await asyncio.to_thread(
                    repository.prepare,
                    transaction_id=transaction_id,
                    base_commit=base.commit_id,
                    files=files,
                    note_ids=note_ids,
                    message=plan.message,
                )
            except ValueError as error:
                raise GitDirectoryMoveRefused(
                    422, "invalid_draft", f"{error}. Nothing was moved."
                ) from error
            try:
                publication = await settle_git_publication(repository, prepared)
            except GitPublicationConflict as error:
                raise GitDirectoryMoveRefused(
                    409,
                    "base_advanced",
                    "The Git project advanced during the move; nothing was moved. "
                    "Repeat the same move.",
                ) from error
            self.last_publication = publication
            try:
                # The basis is the parent of this commit: its notes prove themselves.
                await self.pipeline.recover_content(publication.commit_id, known_ancestor=base)
            except Exception as error:
                raise NoteContentMutationServiceError(
                    503,
                    {
                        "message": "Git published; SQL pending. Check status before retry.",
                        "published_commit": publication.commit_id,
                    },
                ) from error
            # A deferral cannot reach this point: the basis above projected without one
            # (accept-style catch-up), and the move only renames notes it already holds. A
            # deferred project refuses before publication, as "not_accepting" (I1).
            self._sql_commit = publication.commit_id
        finally:
            await self._drop_candidate(transaction_id)
        return GitDirectoryMoveOutcome(plan, publication.commit_id)

    async def _directory_move_disk(self, base: GitSnapshot) -> GitDirectoryMoveDisk:
        """The bytes the files stage may find on disk for each published path.

        Like accept(), first prove that SQL holds exactly the basis: a commit SQL could not
        follow must not be published.
        """
        layout = GitLayoutRepository(self.pipeline.project_id)
        async with self.pipeline.session_maker() as session:
            local = {note.file_path: note for note in await layout.local_notes(session)}
            retired = {
                retirement.file_path: frozenset(retirement.expected_checksums)
                for retirement in await layout.retirements(session)
            }
        if {path: note.external_id for path, note in local.items()} != base.manifest.note_ids:
            raise GitProjectionConflict("SQL content is outside the captured Git basis")
        for path, note in local.items():
            if (
                note.content is None
                or note.content.markdown_content != base.files[path]
                or note.content.checksum != await compute_checksum(note.content.markdown_content)
            ):
                raise GitProjectionConflict("SQL content is outside the captured Git basis")
        published: dict[str, frozenset[str]] = {}
        for path, text in base.files.items():
            # The same set retain_old_file stores, so the later guarded delete succeeds.
            accepted = {hashlib.sha256(text.encode()).hexdigest()}
            if (note := local.get(path)) is not None:
                if note.content is not None:
                    accepted.add(note.content.checksum)
                if note.file_checksum is not None:
                    accepted.add(note.file_checksum)
            published[path] = frozenset(accepted) | retired.get(path, frozenset())
        return GitDirectoryMoveDisk(self.pipeline.file_service.base_path, published, retired)

    async def _moved_tree(
        self, base: GitSnapshot, plan: GitDirectoryMovePlan
    ) -> tuple[dict[str, str], dict[str, UUID]]:
        """The whole project after the move: same bytes, only missing times stamped."""
        async with self.pipeline.session_maker() as session:
            rows = await session.execute(
                select(Entity.external_id, Entity.created_at, Entity.updated_at).where(
                    Entity.project_id == self.pipeline.project_id
                )
            )
            times = {
                UUID(external_id): (created, updated) for external_id, created, updated in rows
            }
        files, note_ids = dict(base.files), dict(base.manifest.note_ids)
        # Remove every source before adding any target: a target may be another note's
        # source when the folder moves up into its parent ('a/b' -> 'a').
        texts = {move.file_path: files.pop(move.previous_file_path) for move in plan.moves}
        for move in plan.moves:
            del note_ids[move.previous_file_path]
        now = datetime.now().astimezone()
        for move in plan.moves:
            text = texts[move.file_path]
            created, modified = times.get(move.external_id, (None, None))
            # A plain move keeps both times; a note without them gets its indexed ones,
            # as a single move and a transaction move do (BM-010).
            wanted = wanted_note_times(
                text,
                now=now,
                current_markdown=text,
                fallback_created=created,
                fallback_modified=modified,
            )
            files[move.file_path] = text if wanted is None else stamp_markdown_times(text, wanted)
            note_ids[move.file_path] = move.external_id
        return files, note_ids

    async def _drop_candidate(self, transaction_id: UUID) -> None:
        """Remove the private candidate ref prepare() leaves, published or not."""
        # Constraint: this runs in a finally block, also when the caller was cancelled.
        #   A cancelled await of to_thread revokes the job while it still waits for a
        #   worker thread, so the ref would stay; shielded, the cleanup always runs.
        await asyncio.shield(asyncio.to_thread(self._remove_candidates, transaction_id))

    def _remove_candidates(self, transaction_id: UUID) -> None:
        try:
            self.pipeline.repository.remove_candidates(transaction_id)
        except (GitPublicationError, subprocess.SubprocessError, OSError) as error:
            # Trigger: git fails, times out (30 s) or cannot start (EMFILE, ENOMEM).
            # Why: raising here would replace the outcome the move already reached (a
            #   published commit, or its refusal) with a 500.
            # Outcome: a leftover candidate ref is harmless residue; the outcome stands.
            logger.warning(
                "Git candidate cleanup failed", transaction_id=str(transaction_id), error=str(error)
            )


async def _scan_disk[**P, T](scan: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a disk check off the event loop; a disk that cannot be read refuses the move.

    The checks run before anything is published, so an OSError there (permission, I/O)
    means nothing moved. Answered as "not accepting", a caller is not told the outcome is
    unknown. A file that vanishes during the scan is not an error (see _disk_entry).
    """
    try:
        return await asyncio.to_thread(scan, *args, **kwargs)
    except OSError as error:
        raise GitDirectoryMoveRefused(
            409,
            "not_accepting",
            f"The project folder could not be read: {error}. Nothing was moved.",
        ) from error


def _unclean_source(conflicts: Sequence[GitDirectoryMoveConflict]) -> GitDirectoryMoveRefused:
    """Refuse a source folder that holds anything but published notes, naming each file."""
    return GitDirectoryMoveRefused(
        422,
        "unpublished_file",
        f"{len(conflicts)} file(s) in the folder are not clean published notes: "
        + "; ".join(conflict.describe() for conflict in conflicts),
        conflicts,
    )


async def owner_content_errors(
    deferral: GitProjectionDeferred, session_maker: async_sessionmaker[AsyncSession]
) -> dict[int, str]:
    """Each owner's own content-stage error, by project id.

    An owner whose own projection fails never releases what it holds; that error is
    the real cause of the wait, so it is named next to it.
    """
    errors: dict[int, str] = {}
    for owner_id in sorted(deferral.owner_project_ids):
        stages = await GitProgressRepository(owner_id, session_maker).read()
        content = next((stage for stage in stages if stage.stage == "content"), None)
        if content is not None and content.last_error:
            errors[owner_id] = content.last_error
    return errors


async def deferral_detail(
    deferral: GitProjectionDeferred, session_maker: async_sessionmaker[AsyncSession]
) -> dict[str, str | None]:
    """The 409 body for a write into a waiting project: notes, owners, owners' errors."""
    owners = sorted(
        {(n.owner_project_id, n.owner_name, str(n.owner_external_id)) for n in deferral.waiting}
    )
    errors = await owner_content_errors(deferral, session_maker)
    return {
        "kind": "deferred",
        "message": f"{deferral}. Nothing was changed.",
        "waiting_note_ids": ",".join(str(note.external_id) for note in deferral.waiting),
        "owner_projects": ",".join(name for _, name, _ in owners),
        "owner_project_ids": ",".join(external for _, _, external in owners),
        "owner_last_error": "; ".join(
            f"{name}: {errors[owner_id]}" for owner_id, name, _ in owners if owner_id in errors
        )
        or None,
    }
