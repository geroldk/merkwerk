"""Local durable draft pointers, capability ownership, leases and terminal evidence.

One short file lock serializes metadata operations, publication and cleanup. It is
never held between agent calls, and no SQLite transaction participates. Markdown
lives in private candidate refs, never in an indexed root or published branch.
"""

import hashlib
import os
import secrets
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Literal
from uuid import UUID, uuid4

from filelock import FileLock

from basic_memory.repository.git_project_repository import (
    GitProjectRepository,
    GitPublicationConflict,
    GitSnapshot,
    PreparedGitPublication,
)
from basic_memory.schemas.git_transactions import (
    GitCleanedState,
    GitCommittedState,
    GitDraftState,
    GitRolledBackState,
    GitTransactionRecord,
    GitTransactionStart,
    GitTransactionStarted,
    GitTransactionStatus,
)


class GitTransactionError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def utc_now() -> datetime:
    return datetime.now(UTC)


def draft_candidate(record: GitTransactionRecord) -> PreparedGitPublication:
    if not isinstance(record.state, GitDraftState):
        raise GitTransactionError(409, "Transaction no longer has a private draft")
    return PreparedGitPublication(
        record.transaction_id,
        record.base_commit,
        record.state.candidate_commit,
        f"refs/bm/candidates/{record.transaction_id}/{record.state.candidate_commit}",
        record.state.changed,
    )


class GitTransactionRepository:
    def __init__(
        self,
        repository: GitProjectRepository,
        project_id: UUID,
        *,
        idle_seconds: int = 1800,
        retention_seconds: int = 604800,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if idle_seconds <= 0 or retention_seconds < 0:
            raise ValueError("Transaction lease must be positive and retention nonnegative")
        self.repository = repository
        self.project_id = project_id
        self.idle = timedelta(seconds=idle_seconds)
        self.retention = timedelta(seconds=retention_seconds)
        self.clock = clock
        self.path = repository.path / "bm-transactions"
        self.lock_path = repository.path / "bm-transactions.lock"

    @contextmanager
    def locked(self) -> Iterator[None]:
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise GitTransactionError(409, "Transaction metadata must not use symlinks")
        self.path.mkdir(mode=0o700, exist_ok=True)
        with FileLock(self.lock_path, timeout=30):
            yield

    def load(self, transaction_id: UUID) -> GitTransactionRecord:
        path = self.path / f"{transaction_id}.json"
        if path.is_symlink():
            raise GitTransactionError(409, "Transaction record must not use a symlink")
        if not path.is_file():
            raise GitTransactionError(404, "Transaction not found or key is invalid")
        record = GitTransactionRecord.model_validate_json(path.read_bytes())
        if record.transaction_id != transaction_id or record.project_id != self.project_id:
            raise GitTransactionError(409, "Transaction record identity does not match its project")
        return record

    def save(self, record: GitTransactionRecord) -> None:
        """Publish the pointer only after Git has durably retained its candidate.

        Same-directory replacement is process-crash atomic. fsync is explicit;
        this does not claim tested power-loss behavior for the Git object store.
        """
        with NamedTemporaryFile(mode="wb", dir=self.path, prefix=".draft-", delete=False) as out:
            temporary = Path(out.name)
            try:
                out.write(record.model_dump_json(indent=2).encode("utf-8"))
                out.flush()
                os.fsync(out.fileno())
                os.replace(temporary, self.path / f"{record.transaction_id}.json")
                directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)

    def reconcile(self, record: GitTransactionRecord) -> GitTransactionRecord:
        # Resolve publication BEFORE expiry/cleanup. Git may have published while
        # the process died before replacing this small metadata record.
        if isinstance(record.state, GitDraftState):
            publication = self.repository.transaction_status(record.transaction_id)
            if publication is not None:
                record = record.model_copy(
                    update={"state": GitCommittedState(commit_id=publication.commit_id)}
                )
                self.save(record)
            elif record.state.kind != "expired" and self.clock() >= record.expires_at:
                record = record.model_copy(
                    update={"state": record.state.model_copy(update={"kind": "expired"})}
                )
                self.save(record)
        return record

    def owned(self, transaction_id: UUID, key: str) -> GitTransactionRecord:
        record = self.load(transaction_id)
        if not secrets.compare_digest(record.owner_hash, hashlib.sha256(key.encode()).hexdigest()):
            raise GitTransactionError(404, "Transaction not found or key is invalid")
        return self.reconcile(record)

    def status_value(
        self, record: GitTransactionRecord, *, already_committed: bool = False
    ) -> GitTransactionStatus:
        return GitTransactionStatus(
            transaction_id=record.transaction_id,
            project_id=record.project_id,
            base_commit=record.base_commit,
            revision=record.revision,
            state=record.state.kind,
            expires_at=record.expires_at,
            retained_until=record.expires_at + self.retention,
            draft_available=isinstance(record.state, GitDraftState),
            published_commit=record.state.commit_id
            if isinstance(record.state, GitCommittedState)
            else None,
            conflict_commit=record.state.conflict_commit
            if isinstance(record.state, GitDraftState)
            else None,
            source_transaction_id=record.source_transaction_id,
            source_revision=record.source_revision,
            already_committed=already_committed,
        )

    def renew(self, record: GitTransactionRecord) -> GitTransactionRecord:
        if not isinstance(record.state, GitDraftState) or record.state.kind == "expired":
            raise GitTransactionError(409, "Transaction lease has ended; start a new transaction")
        now = self.clock()
        renewed = record.model_copy(update={"last_activity_at": now, "expires_at": now + self.idle})
        self.save(renewed)
        return renewed

    def require_revision(self, record: GitTransactionRecord, revision: int) -> None:
        if record.revision != revision:
            raise GitTransactionError(
                409, f"Draft revision is {record.revision}; inspect it before retrying"
            )

    def require_writable(self, record: GitTransactionRecord, revision: int) -> None:
        self.require_revision(record, revision)
        if not isinstance(record.state, GitDraftState) or record.state.kind != "open":
            raise GitTransactionError(
                409, f"Transaction is {record.state.kind}; draft changes are refused"
            )

    def maintain(self) -> None:
        with self.locked():
            self.maintain_locked()

    def maintain_locked(self) -> set[UUID]:
        records = {
            UUID(p.stem): self.reconcile(self.load(UUID(p.stem))) for p in self.path.glob("*.json")
        }
        # A retained successor pins its fixed source, transitively. Terminal
        # successors no longer need the source; their own data is independent.
        protected = {
            identity
            for identity, record in records.items()
            if isinstance(record.state, GitDraftState)
            and self.clock() < record.expires_at + self.retention
        }
        pending = list(protected)
        while pending:
            record = records[pending.pop()]
            parent = record.source_transaction_id
            if parent is not None and parent not in protected:
                if parent not in records:
                    raise GitTransactionError(409, "Retained draft source is missing")
                protected.add(parent)
                pending.append(parent)
        # Constraint: terminal records are kept for good, and every transaction call runs
        #   this; one ref listing replaces a Git call per record, which made each call
        #   slower with every committed move (a folder of 600 single moves).
        retained = self.repository.candidate_transaction_ids()
        for identity, record in records.items():
            if isinstance(record.state, GitDraftState) and identity not in protected:
                self.save(record.model_copy(update={"state": GitCleanedState()}))
                if identity in retained:
                    self.repository.remove_candidates(identity)
            elif not isinstance(record.state, GitDraftState) and identity in retained:
                self.repository.remove_candidates(identity)
        return {
            record.source_transaction_id
            for identity, record in records.items()
            if identity in protected and record.source_transaction_id is not None
        }

    def start(self, request: GitTransactionStart) -> GitTransactionStarted:
        with self.locked():
            self.maintain_locked()
            source = None
            if request.source_transaction_id is not None:
                assert request.source_key is not None and request.source_revision is not None
                source = self.owned(request.source_transaction_id, request.source_key)
                self.require_revision(source, request.source_revision)
                if not isinstance(source.state, GitDraftState):
                    raise GitTransactionError(
                        409, "Source draft is unavailable or already committed"
                    )
                if source.state.kind != "expired":
                    source = self.renew(source)
                if source.state.kind == "open":
                    source = source.model_copy(
                        update={"state": source.state.model_copy(update={"kind": "frozen"})}
                    )
                    self.save(source)
            base = self.repository.snapshot()
            identity, key, now = uuid4(), secrets.token_urlsafe(32), self.clock()
            candidate = self.repository.prepare(
                transaction_id=identity,
                base_commit=base.commit_id,
                files=base.files,
                note_ids=base.manifest.note_ids,
            )
            record = GitTransactionRecord(
                transaction_id=identity,
                project_id=self.project_id,
                base_commit=base.commit_id,
                owner_hash=hashlib.sha256(key.encode()).hexdigest(),
                created_at=now,
                last_activity_at=now,
                expires_at=now + self.idle,
                state=GitDraftState(
                    candidate_commit=candidate.commit_id, changed=candidate.changed
                ),
                source_transaction_id=source.transaction_id if source is not None else None,
                source_revision=source.revision if source is not None else None,
            )
            self.save(record)
            return GitTransactionStarted.model_validate(
                {**self.status_value(record).model_dump(), "transaction_key": key}
            )

    def status(
        self, transaction_id: UUID, key: str, *, keep_alive: bool = False
    ) -> GitTransactionStatus:
        with self.locked():
            self.maintain_locked()
            record = self.owned(transaction_id, key)
            if keep_alive:
                record = self.renew(record)
            return self.status_value(record)

    def read(
        self, transaction_id: UUID, key: str, *, view: Literal["base", "draft"] = "draft"
    ) -> tuple[GitTransactionRecord, GitSnapshot]:
        with self.locked():
            self.maintain_locked()
            record = self.owned(transaction_id, key)
            if not isinstance(record.state, GitDraftState):
                raise GitTransactionError(409, "Private draft is no longer available")
            if record.state.kind != "expired":
                record = self.renew(record)
            snapshot = (
                self.repository.snapshot(record.base_commit)
                if view == "base"
                else self.repository.candidate_snapshot(draft_candidate(record))
            )
            return record, snapshot

    def replace(
        self,
        transaction_id: UUID,
        key: str,
        *,
        expected_revision: int,
        files: dict[str, str],
        note_ids: dict[str, UUID],
    ) -> GitTransactionStatus:
        with self.locked():
            self.maintain_locked()
            record = self.owned(transaction_id, key)
            self.require_writable(record, expected_revision)
            candidate = self.repository.prepare(
                transaction_id=transaction_id,
                base_commit=record.base_commit,
                files=files,
                note_ids=note_ids,
            )
            record = record.model_copy(
                update={
                    "revision": record.revision + 1,
                    "state": GitDraftState(
                        candidate_commit=candidate.commit_id, changed=candidate.changed
                    ),
                }
            )
            return self.status_value(self.renew(record))

    def commit(
        self,
        transaction_id: UUID,
        key: str,
        *,
        expected_revision: int,
        message: str,
        moved_notes: Mapping[UUID, UUID] | None = None,
    ) -> GitTransactionStatus:
        with self.locked():
            self.maintain_locked()
            record = self.owned(transaction_id, key)
            if isinstance(record.state, GitCommittedState):
                return self.status_value(record, already_committed=True)
            self.require_writable(record, expected_revision)
            snapshot = self.repository.candidate_snapshot(draft_candidate(record))
            candidate = self.repository.prepare(
                transaction_id=transaction_id,
                base_commit=record.base_commit,
                files=snapshot.files,
                note_ids=snapshot.manifest.note_ids,
                message=message,
                moved_notes=moved_notes,
            )
            try:
                publication = self.repository.publish(candidate)
            except GitPublicationConflict as error:
                self.save(
                    record.model_copy(
                        update={
                            "state": record.state.model_copy(
                                update={"kind": "conflicted", "conflict_commit": error.actual}
                            )
                        }
                    )
                )
                raise GitTransactionError(
                    409,
                    f"Project advanced from {error.expected} to {error.actual}; draft preserved",
                ) from error
            record = record.model_copy(
                update={"state": GitCommittedState(commit_id=publication.commit_id)}
            )
            self.save(record)
            return self.status_value(record, already_committed=publication.already_committed)

    def rollback(
        self, transaction_id: UUID, key: str, *, expected_revision: int
    ) -> GitTransactionStatus:
        with self.locked():
            pinned = self.maintain_locked()
            record = self.owned(transaction_id, key)
            self.require_revision(record, expected_revision)
            if isinstance(record.state, GitCommittedState):
                raise GitTransactionError(
                    409, "Transaction is published; rollback cannot rewind Git"
                )
            if transaction_id in pinned:
                raise GitTransactionError(409, "Retained successor still needs this source draft")
            record = record.model_copy(update={"state": GitRolledBackState()})
            self.save(record)
            self.repository.remove_candidates(transaction_id)
            return self.status_value(record)
