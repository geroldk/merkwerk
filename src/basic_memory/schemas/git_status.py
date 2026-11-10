"""Published content and historical index progress; no strict snapshot guarantee."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from basic_memory.schemas.git_sync import GitSyncStatus

type GitProjectionStage = Literal["content", "files", "graph", "vectors"]


class GitWaitingNote(BaseModel):
    """A note the content stage waits for: another project still owns its UUID (I1)."""

    external_id: UUID
    owner_project_id: str
    owner_project: str
    # The owner's own content error, when it fails: then it never releases by itself.
    owner_last_error: str | None = None


class GitStageStatus(BaseModel):
    stage: GitProjectionStage
    target_commit: str
    completed_commit: str | None
    running_commit: str | None
    attempts: int
    last_error: str | None
    # Set on the content stage while notes wait for other projects (I1); `last_error`
    # then holds the same as text. Waiting is not a failure, but caught_up stays false.
    deferred: list[GitWaitingNote] | None = None


class GitProjectStatus(BaseModel):
    project_id: str
    source_commit: str
    caught_up: bool
    read_consistency: Literal["relaxed"] = "relaxed"
    stages: list[GitStageStatus]
    synchronization: GitSyncStatus | None = None
