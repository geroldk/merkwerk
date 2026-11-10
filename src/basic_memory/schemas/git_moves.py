"""Cross-project moves on a Git host (I1): the durable intent and the API contracts.

A move keeps the note's UUID: a draft in the target holds the note under its old UUID, the
source deletes it, then the target commits (W2). Between the source delete and the target
commit the note is in no project's current state; the intent, kept next to the target's
bare repository, holds what is needed to finish the move after a crash.
"""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from basic_memory.schemas.git_project import validate_note_path
from basic_memory.schemas.git_transactions import GitCommitId

# drafted: the target draft exists; the source may or may not have let go yet.
# source_deleted: the source published the delete (its commit is recorded).
# stuck: the source let go, but the move cannot finish as planned; `stuck_reason` names
#   why, and a person chooses where it goes (POST /v2/git/moves/{move_id}/resolve).
#   Nothing is lost.
type GitMoveState = Literal["drafted", "source_deleted", "stuck"]


class GitMoveIntent(BaseModel):
    """What a cross-project move needs to finish, written before the source lets go."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    format_version: Literal[1] = 1
    move_id: UUID
    created_at: AwareDatetime

    # --- Source: the note as it was read and the head it was read at ---
    source_project_id: UUID
    source_path: str
    source_checksum: str
    source_base_commit: GitCommitId
    source_permalink: str | None = None
    # The source entity's times: a note without `created`/`modified` keeps them (BM-010).
    source_created: AwareDatetime | None = None
    source_modified: AwareDatetime | None = None

    # --- Target: where the note goes, with the exact bytes the draft holds ---
    target_project_id: UUID
    external_id: UUID
    target_path: str
    title: str
    target_content: str
    target_permalink: str

    # --- The target draft (the key is a secret; never listed) ---
    transaction_id: UUID
    transaction_key: str

    # --- Outcome so far ---
    state: GitMoveState = "drafted"
    source_commit: GitCommitId | None = None
    stuck_reason: str | None = None

    @field_validator("source_path", "target_path")
    @classmethod
    def require_note_path(cls, value: str) -> str:
        validate_note_path(value)
        return value


class GitCrossProjectMoveRequest(BaseModel):
    """POST /v2/projects/{source}/git/moves: move one note into another project."""

    model_config = ConfigDict(extra="forbid")
    external_id: UUID
    target_project_id: UUID
    destination_path: str
    expected_checksum: str | None = Field(default=None, min_length=1, max_length=128)
    # A folder moved note by note skips the catch-up after each commit and catches the
    # target up once at the end (POST .../git/moves/catch-up); the target's SQL still
    # follows its head before each note, for the title and permalink checks.
    catch_up: bool = True

    @field_validator("destination_path")
    @classmethod
    def require_note_path(cls, value: str) -> str:
        validate_note_path(value)
        return value


# returned: a person chose to put the note of a stuck move back into the source; then
#   `file_path` is its path there.
type GitMoveOutcomeState = Literal["completed", "pending", "stuck", "returned"]


class GitCrossProjectMoveResult(BaseModel):
    """A move that reached the target (completed) or keeps its intent (pending, stuck)."""

    state: GitMoveOutcomeState
    move_id: UUID
    external_id: UUID
    source_project_id: UUID
    target_project_id: UUID
    previous_file_path: str
    file_path: str
    previous_permalink: str | None
    permalink: str
    source_commit: GitCommitId | None = None
    target_commit: GitCommitId | None = None
    # The target's SQL had not caught up when the answer left (completed only): reads by
    # UUID or path may lag until its worker has projected the commit.
    index_pending: bool = False
    message: str | None = None


class GitMoveIntentView(BaseModel):
    """An open intent as GET /v2/git/moves lists it: no note bytes, no transaction key."""

    move_id: UUID
    created_at: AwareDatetime
    state: GitMoveState
    external_id: UUID
    title: str
    source_project_id: UUID
    source_path: str
    source_base_commit: GitCommitId
    source_commit: GitCommitId | None
    target_project_id: UUID
    target_path: str
    target_permalink: str
    transaction_id: UUID
    stuck_reason: str | None

    @classmethod
    def of(cls, intent: GitMoveIntent) -> "GitMoveIntentView":
        # The view ignores every other key; the bytes and the key never leave the host.
        return cls.model_validate(intent.model_dump(exclude={"target_content", "transaction_key"}))


class GitMoveListing(BaseModel):
    moves: list[GitMoveIntentView]
    # Intent files that could not be used, by project and file name with a fixed reason
    # (no host path, no raw error). Each needs a person; while any is listed, every
    # caller-chosen UUID is refused host-wide, since such a file's move is unknown.
    unreadable: list[str] = []


type GitMoveResumeKind = Literal["completed", "abandoned", "pending", "stuck", "returned", "failed"]


class GitMoveResumeOutcome(BaseModel):
    move_id: UUID
    external_id: UUID
    outcome: GitMoveResumeKind
    detail: str
    target_commit: GitCommitId | None = None


class GitMoveResumeResult(BaseModel):
    outcomes: list[GitMoveResumeOutcome]


class GitMoveResolve(BaseModel):
    """A person's choice for a stuck move; both keep the note and its UUID.

    target: publish it in the target project, at `destination_path` (default: the planned
      path), with its permalink formed again there.
    source: put it back into the source project, at `destination_path` (default: where it
      was), with the bytes it had there; at another path the permalink is formed again
      there (a taken one gets '-1'), as for any new note.
    """

    model_config = ConfigDict(extra="forbid")
    choice: Literal["target", "source"]
    destination_path: str | None = None

    @field_validator("destination_path")
    @classmethod
    def require_note_path(cls, value: str | None) -> str | None:
        if value is not None:
            validate_note_path(value)
        return value


class GitMoveCatchUp(BaseModel):
    """The answer of POST .../git/moves/catch-up: SQL of the project follows its head."""

    project_id: UUID
    commit_id: GitCommitId
    index_pending: bool
    message: str | None = None
