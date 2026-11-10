"""Private transaction records and explicit API contracts; no general V2 replay."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from basic_memory.schemas.git_project import validate_note_path

type GitCommitId = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
type GitDraftKind = Literal["open", "frozen", "conflicted", "expired"]


class GitDraftState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: GitDraftKind = "open"
    candidate_commit: GitCommitId
    changed: bool
    conflict_commit: GitCommitId | None = None


class GitCommittedState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["committed"] = "committed"
    commit_id: GitCommitId


class GitRolledBackState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["rolled_back"] = "rolled_back"


class GitCleanedState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["expired_cleaned"] = "expired_cleaned"


type GitTransactionState = Annotated[
    GitDraftState | GitCommittedState | GitRolledBackState | GitCleanedState,
    Field(discriminator="kind"),
]


class GitTransactionRecord(BaseModel):
    """Small durable pointer; draft bytes are retained exclusively in private Git refs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    format_version: Literal[1] = 1
    transaction_id: UUID
    project_id: UUID
    base_commit: GitCommitId
    owner_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    revision: int = Field(default=0, ge=0)
    created_at: AwareDatetime
    last_activity_at: AwareDatetime
    expires_at: AwareDatetime
    state: GitTransactionState
    source_transaction_id: UUID | None = None
    source_revision: int | None = None


class GitTransactionStatus(BaseModel):
    transaction_id: UUID
    project_id: UUID
    base_commit: GitCommitId
    revision: int
    state: Literal[
        "open", "frozen", "conflicted", "expired", "expired_cleaned", "committed", "rolled_back"
    ]
    expires_at: AwareDatetime
    retained_until: AwareDatetime
    draft_available: bool
    published_commit: GitCommitId | None = None
    conflict_commit: GitCommitId | None = None
    source_transaction_id: UUID | None = None
    source_revision: int | None = None
    already_committed: bool = False


class GitTransactionStarted(GitTransactionStatus):
    transaction_key: str


class GitTransactionStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_transaction_id: UUID | None = None
    source_key: str | None = None
    source_revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_complete_source(self) -> Self:
        supplied = (self.source_transaction_id, self.source_key, self.source_revision)
        if any(value is not None for value in supplied) and any(
            value is None for value in supplied
        ):
            raise ValueError("Source transfer requires transaction ID, key and revision together")
        return self


class GitTransactionControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)


class GitTransactionCommit(GitTransactionControl):
    message: str = Field(default="Publish Basic Memory transaction", min_length=1, max_length=10000)

    @field_validator("message")
    @classmethod
    def require_safe_message(cls, value: str) -> str:
        # Basic-Memory-* trailers are BM's own evidence (transaction, moved note): the
        # prefix is refused anywhere and in any case, as GitProjectRepository.prepare does.
        if not value.strip() or "basic-memory-" in value.casefold() or "\0" in value:
            raise ValueError("Commit message must omit reserved transaction evidence")
        return value


class GitTransactionWrite(GitTransactionControl):
    path: str
    content: str
    operation: Literal["replace", "append", "prepend"] = "replace"

    @field_validator("path")
    @classmethod
    def require_note_path(cls, value: str) -> str:
        validate_note_path(value)
        return value


class GitTransactionMove(GitTransactionControl):
    path: str
    destination: str

    @field_validator("path", "destination")
    @classmethod
    def require_note_path(cls, value: str) -> str:
        validate_note_path(value)
        return value


class GitTransactionDelete(GitTransactionControl):
    path: str

    @field_validator("path")
    @classmethod
    def require_note_path(cls, value: str) -> str:
        validate_note_path(value)
        return value


class GitTransactionNoteInfo(BaseModel):
    external_id: UUID
    path: str
    checksum: str


class GitTransactionNote(GitTransactionNoteInfo):
    content: str


class GitTransactionRead(BaseModel):
    transaction: GitTransactionStatus
    view: Literal["base", "draft"]
    note: GitTransactionNote


class GitTransactionListing(BaseModel):
    transaction: GitTransactionStatus
    view: Literal["base", "draft"]
    notes: list[GitTransactionNoteInfo]
