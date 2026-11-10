"""Explicit opportunistic sync. Network waits never hold local mutation admission."""

import asyncio
import re
import subprocess
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from loguru import logger

from basic_memory.repository.git_project_repository import (
    GitPublicationConflict,
    GitPublicationError,
    GitSnapshot,
    settle_git_operation,
)
from basic_memory.repository.git_sync_repository import GitSyncRepository
from basic_memory.repository.git_transaction_repository import GitTransactionError
from basic_memory.schemas.git_sync import (
    GitSyncRecord,
    GitSyncRemote,
    GitSyncStatus,
    SyncCompleted,
    SyncFailed,
    SyncRunning,
)
from basic_memory.schemas.git_project import GitNoteIdentity, GitProjectManifest
from basic_memory.services.git_merge import plan_merge
from basic_memory.services.git_merge_resolution import (
    Resolved,
    ReviewIncomplete,
    require_decided,
    resolve_package,
)
from basic_memory.services.git_merge_review import (
    REVIEW_DIRECTORY,
    read_review,
    review_id,
    write_review_package,
)
from basic_memory.services.git_transactions import validate_git_draft

if TYPE_CHECKING:  # pragma: no cover
    from basic_memory.services.git_project_runtime import GitProjectRuntime


_REVIEW_ID = re.compile(r"[0-9a-f]{12}-[0-9a-f]{12}\Z")


class GitSyncBusy(RuntimeError):
    """A second call must not compete with this project's current sync attempt."""


class GitSyncService:
    def __init__(
        self, runtime: "GitProjectRuntime", remote: GitSyncRemote | None, *, merge: bool = False
    ) -> None:
        self.runtime = runtime
        self.remote = remote
        self.merge = merge
        self.lock = asyncio.Lock()
        self.records = GitSyncRepository(
            runtime.pipeline.repository, UUID(runtime.pipeline.project_external_id), remote
        )
        # The failed attempt's id and its reason in words (not persisted, see GitSyncStatus).
        self._failure: tuple[UUID, str] | None = None

    async def recover(self) -> None:
        record = await asyncio.to_thread(self.records.load)
        if isinstance(record.attempt, SyncRunning):
            await self.save(
                record.model_copy(
                    update={
                        "attempt": SyncFailed(
                            attempt_id=record.attempt.attempt_id,
                            started_at=record.attempt.started_at,
                            finished_at=datetime.now(UTC),
                            reason="interrupted",
                        )
                    }
                )
            )

    async def save(self, record: GitSyncRecord) -> None:
        await settle_git_operation(partial(self.records.save, record))

    async def status(self) -> GitSyncStatus:
        record = await asyncio.to_thread(self.records.load)
        repository = self.runtime.pipeline.repository
        local = await asyncio.to_thread(repository.head)
        relationship = await asyncio.to_thread(
            repository.relationship, local, record.observed_remote_commit
        )
        attempt = record.attempt
        review = (
            await asyncio.to_thread(
                read_review,
                repository.path / REVIEW_DIRECTORY,
                review_id(local, record.observed_remote_commit),
            )
            if isinstance(attempt, SyncCompleted)
            and attempt.outcome == "needs_review"
            and record.observed_remote_commit is not None
            else None
        )
        return GitSyncStatus(
            open_review=review,
            configured=self.remote is not None,
            local_commit=local,
            observed_remote_commit=record.observed_remote_commit,
            relationship=relationship,
            last_attempt=record.attempt,
            failure_detail=(
                self._failure[1]
                if self._failure is not None
                and isinstance(attempt, SyncFailed)
                and attempt.attempt_id == self._failure[0]
                else None
            ),
        )

    async def _merge(
        self,
        attempt_id: UUID,
        local: str,
        remote: GitSnapshot,
        project_id: UUID,
        review: str | None = None,
    ) -> tuple[Literal["merged", "needs_review", "diverged"], str | None]:
        """Merge note by note; returns the outcome and the locally published merge, if any.

        Runs outside the local mutation gate: only the compare-and-swap in publish_merge
        needs it, and it fails (local_changed) if the local tip moved meanwhile.
        """
        repository = self.runtime.pipeline.repository
        bases = await asyncio.to_thread(repository.merge_bases, local, remote.commit_id)
        if len(bases) != 1:
            return "diverged", None  # criss-cross history: a person decides how to proceed
        base = await asyncio.to_thread(repository.read_snapshot, bases[0])
        left = await asyncio.to_thread(repository.read_snapshot, local)
        root = repository.path / REVIEW_DIRECTORY
        resolved: Resolved | None = None
        plan = plan_merge(base, left, remote)
        if review is not None:
            package = root / review
            document = await asyncio.to_thread(require_decided, package)
            if (document["local"], document["remote"]) != (local, remote.commit_id):
                # A tip moved since the decisions were made: carry over what still applies
                # and ask again, never apply a decision to versions its author did not see.
                await asyncio.to_thread(
                    write_review_package,
                    root,
                    project_id=project_id,
                    base=base,
                    local=left,
                    remote=remote,
                    plan=plan,
                    previous=package,
                )
                return "needs_review", None
            resolved = await asyncio.to_thread(resolve_package, package, base, left, remote)
            plan = plan_merge(base, left, remote, decided=resolved.decided)
        if plan.conflicts:
            await asyncio.to_thread(
                write_review_package,
                root,
                project_id=project_id,
                base=base,
                local=left,
                remote=remote,
                plan=plan,
            )
            return "needs_review", None
        await validate_git_draft(plan.files, self.runtime.pipeline.file_service.base_path)
        merge = await asyncio.to_thread(
            partial(
                repository.prepare_merge,
                attempt_id=attempt_id,
                base=base,
                local=left,
                remote=remote,
                files=plan.files,
                note_ids=plan.note_ids,
                reviewed=resolved.reviewed if resolved else frozenset(),
                review_id=resolved.review_id if resolved else None,
            )
        )
        manifest = GitProjectManifest(
            project_external_id=project_id,
            notes=tuple(GitNoteIdentity(path=p, external_id=i) for p, i in plan.note_ids.items()),
        )
        async with self.runtime.admission():
            async with self.runtime.pipeline.session_maker() as session:
                # The prepared merge is the history to judge: a note it keeps although
                # the local side deleted it is introduced by the merge itself (I1).
                await self.records.require_note_ownership(
                    session,
                    self.runtime.pipeline.project_id,
                    manifest,
                    identities=self.runtime.pipeline.identities,
                    history=(repository, merge.commit_id),
                )
            await settle_git_operation(partial(repository.publish_merge, merge))
        return "merged", merge.commit_id

    async def synchronize(self, *, review: str | None = None) -> GitSyncStatus:
        if self.remote is None:
            raise ValueError("No Git peer configured for this project")
        if review is not None:
            if not self.merge:
                raise ValueError("Merge is not enabled for this project")
            if _REVIEW_ID.fullmatch(review) is None:
                raise ValueError("Not a review package id")
            await asyncio.to_thread(
                require_decided,
                self.runtime.pipeline.repository.path / REVIEW_DIRECTORY / review,
            )
        if self.lock.locked():
            raise GitSyncBusy("A sync attempt is already running for this project")
        async with self.lock:
            # Only the lifecycle owner may start a transfer; never hold this
            # gate across network I/O or full incoming-history validation.
            async with self.runtime.admission():
                record = await asyncio.to_thread(self.records.load)
                attempt = SyncRunning(attempt_id=uuid4(), started_at=datetime.now(UTC))
                record = record.model_copy(update={"attempt": attempt})
                await self.save(record)
            repository = self.runtime.pipeline.repository
            failure_reason = "transport"
            refused: ReviewIncomplete | None = None
            try:
                incoming = await settle_git_operation(
                    partial(repository.fetch_published, self.remote.url, attempt.attempt_id)
                )
                previous = record.observed_remote_commit
                rewritten = previous is not None and (
                    incoming is None
                    or not await asyncio.to_thread(repository.is_ancestor, previous, incoming)
                )
                if rewritten:
                    outcome = "remote_rewritten"
                else:
                    validated_tip: GitSnapshot | None = None
                    if incoming is not None:
                        failure_reason = "invalid_incoming"
                        snapshots = await asyncio.to_thread(
                            repository.incoming_snapshots, attempt.attempt_id, incoming
                        )
                        validated_tip = snapshots[0]
                        for snapshot in snapshots:
                            if snapshot.manifest.project_external_id != record.project_id:
                                raise ValueError("Incoming history belongs to another project")
                            await validate_git_draft(
                                snapshot.files, self.runtime.pipeline.file_service.base_path
                            )
                    failure_reason = "local_changed"
                    async with self.runtime.admission():
                        local = await asyncio.to_thread(repository.head)
                        relationship = await asyncio.to_thread(
                            repository.relationship, local, incoming
                        )
                        if relationship == "behind":
                            assert incoming is not None and validated_tip is not None
                            failure_reason = "invalid_incoming"
                            async with self.runtime.pipeline.session_maker() as session:
                                await self.records.require_note_ownership(
                                    session,
                                    self.runtime.pipeline.project_id,
                                    validated_tip.manifest,
                                    identities=self.runtime.pipeline.identities,
                                    history=(repository, incoming),
                                )
                            failure_reason = "local_changed"
                            await settle_git_operation(
                                partial(
                                    repository.publish_incoming, attempt.attempt_id, incoming, local
                                )
                            )
                    if relationship in {"ahead", "unobserved"}:
                        failure_reason = "transport"
                        await settle_git_operation(
                            partial(repository.push_published, self.remote.url, local)
                        )
                        incoming = local
                        outcome = "pushed"
                    elif relationship == "behind":
                        outcome = "pulled"
                    elif relationship == "diverged" and self.merge:
                        assert incoming is not None and validated_tip is not None
                        failure_reason = "invalid_incoming"
                        outcome, merged = await self._merge(
                            attempt.attempt_id, local, validated_tip, record.project_id, review
                        )
                        if merged is not None:
                            # Published locally; if the push fails the next run sees `ahead`,
                            # and the observed remote must stay the tip that was merged.
                            failure_reason = "transport"
                            await settle_git_operation(
                                partial(repository.push_published, self.remote.url, merged)
                            )
                            incoming = merged
                    else:
                        outcome = relationship
                record = record.model_copy(
                    update={
                        "observed_remote_commit": previous if rewritten else incoming,
                        "attempt": SyncCompleted(
                            attempt_id=attempt.attempt_id,
                            started_at=attempt.started_at,
                            finished_at=datetime.now(UTC),
                            outcome=outcome,
                        ),
                    }
                )
            except asyncio.CancelledError:
                record = record.model_copy(
                    update={
                        "attempt": SyncFailed(
                            attempt_id=attempt.attempt_id,
                            started_at=attempt.started_at,
                            finished_at=datetime.now(UTC),
                            reason="interrupted",
                        )
                    }
                )
                await self.save(record)
                raise
            except (
                GitPublicationError,
                subprocess.TimeoutExpired,
                OSError,
                ValueError,
                GitTransactionError,
            ) as error:
                if isinstance(error, GitPublicationConflict):
                    failure_reason = "local_changed"
                if isinstance(error, ReviewIncomplete):
                    refused = error  # the caller must see why its decisions were not applied
                if failure_reason == "invalid_incoming":
                    # The record keeps only the reason; the words (which UUID, which
                    # projects) go to the log and to the status of this attempt.
                    self._failure = (attempt.attempt_id, str(error))
                    logger.warning(
                        "Git sync refused incoming history",
                        project_id=self.runtime.pipeline.project_id,
                        reason=str(error),
                    )
                record = record.model_copy(
                    update={
                        "attempt": SyncFailed(
                            attempt_id=attempt.attempt_id,
                            started_at=attempt.started_at,
                            finished_at=datetime.now(UTC),
                            reason=failure_reason,
                        )
                    }
                )
            finally:
                # Git may have changed before a lost response or process failure.
                # The ordinary worker re-reads canonical main, never incoming refs.
                self.runtime.worker.notify()
            await self.save(record)
            if refused is not None:
                raise refused
            return await self.status()
