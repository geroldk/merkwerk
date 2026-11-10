"""Configured peer and durable observations, never a claim about live remote state."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from basic_memory.config_git import GitSyncRemote as GitSyncRemote


class SyncAttemptBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attempt_id: UUID
    started_at: datetime


class SyncRunning(SyncAttemptBase):
    kind: Literal["running"] = "running"


class SyncCompleted(SyncAttemptBase):
    kind: Literal["completed"] = "completed"
    finished_at: datetime
    outcome: Literal[
        "equal", "pushed", "pulled", "diverged", "remote_rewritten", "merged", "needs_review"
    ]


class SyncFailed(SyncAttemptBase):
    kind: Literal["failed"] = "failed"
    finished_at: datetime
    reason: Literal["transport", "invalid_incoming", "local_changed", "interrupted"]


type GitSyncAttempt = Annotated[
    SyncRunning | SyncCompleted | SyncFailed, Field(discriminator="kind")
]


class GitSyncRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format_version: Literal[1] = 1
    project_id: UUID
    peer_fingerprint: str
    observed_remote_commit: str | None = None
    attempt: GitSyncAttempt | None = None


class GitSyncReview(BaseModel):
    """An open merge review: paths and counts only, never note contents."""

    id: str
    conflicts: dict[str, int]
    paths: list[str]


class GitSyncStatus(BaseModel):
    configured: bool
    local_commit: str
    observed_remote_commit: str | None
    relationship: Literal["unobserved", "equal", "ahead", "behind", "diverged"]
    last_attempt: GitSyncAttempt | None
    open_review: GitSyncReview | None = None
    # Why the last attempt failed, in words (e.g. which note UUID two projects claim).
    # Held in memory only: bm-sync.json keeps its format, so an older version reads it.
    failure_detail: str | None = None
