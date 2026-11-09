"""Explicit private draft operations attached to an already owned Git runtime."""

import hashlib
from collections.abc import Mapping
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from filelock import Timeout
from pydantic import ValidationError
from sqlalchemy import select

from basic_memory.file_utils import ParseError, parse_frontmatter
from basic_memory.markdown import EntityParser
from basic_memory.repository.git_project_repository import (
    GitPublicationError,
    settle_git_operation,
)
from basic_memory.repository.git_transaction_repository import (
    GitTransactionError,
    GitTransactionRepository,
)
from basic_memory.schemas.git_transactions import (
    GitTransactionCommit,
    GitTransactionControl,
    GitTransactionDelete,
    GitTransactionListing,
    GitTransactionMove,
    GitTransactionNote,
    GitTransactionNoteInfo,
    GitTransactionRead,
    GitTransactionStart,
    GitTransactionStarted,
    GitTransactionStatus,
    GitTransactionWrite,
)
from basic_memory.models import Entity
from basic_memory.services.note_preparation import (
    PermalinkChangeError,
    apply_edit_operation,
    require_permalink_unchanged,
)
from basic_memory.services.note_times import stamp_markdown_times, wanted_note_times

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from basic_memory.services.git_project_runtime import GitProjectRuntime


async def validate_git_draft(files: Mapping[str, str], root: Path) -> None:
    """Parse the complete proposed tree without consulting mutable SQL/disk notes."""
    parser = EntityParser(root)
    permalinks: set[str] = set()
    for path, content in files.items():
        try:
            metadata = parse_frontmatter(content)
            if any(
                not isinstance(metadata.get(field), str) or not metadata[field].strip()
                for field in ("title", "type", "permalink")
            ):
                raise ValueError("Canonical title/type/permalink frontmatter is required")
            permalink = metadata["permalink"]
            if permalink in permalinks:
                raise ValueError("Draft notes must have unique permalinks")
            permalinks.add(permalink)
            await parser.parse_markdown_content(
                file_path=Path(path), content=content, mtime=0, ctime=0
            )
        except (ParseError, ValueError) as error:
            raise GitTransactionError(422, f"Invalid draft note {path}: {error}") from error


def _draft_permalink(text: str) -> str | None:
    """The `permalink:` of a draft text; None when there is none or the frontmatter is unreadable."""
    try:
        value = parse_frontmatter(text).get("permalink")
    except ParseError:  # draft validation reports unreadable frontmatter on its own
        return None
    return value if isinstance(value, str) and value else None


class GitTransactionService:
    def __init__(
        self,
        runtime: "GitProjectRuntime",
        *,
        idle_seconds: int = 1800,
        retention_seconds: int = 604800,
    ) -> None:
        self.runtime = runtime
        self.repository = GitTransactionRepository(
            runtime.pipeline.repository,
            UUID(runtime.pipeline.project_external_id),
            idle_seconds=idle_seconds,
            retention_seconds=retention_seconds,
        )

    async def call[T](self, operation: "Callable[[], T]") -> T:
        try:
            return await settle_git_operation(operation)
        except GitTransactionError:
            raise
        except (OSError, Timeout, GitPublicationError, ValidationError) as error:
            raise GitTransactionError(
                503,
                "Transaction operation interrupted; inspect its status and contents before retrying",
            ) from error
        except ValueError as error:
            raise GitTransactionError(422, str(error)) from error

    async def maintain(self) -> None:
        await self.call(self.repository.maintain)

    async def start(self, request: GitTransactionStart) -> GitTransactionStarted:
        async with self.runtime.admission():
            return await self.call(partial(self.repository.start, request))

    async def status(
        self, identity: UUID, key: str, *, keep_alive: bool = False
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            return await self.call(
                partial(self.repository.status, identity, key, keep_alive=keep_alive)
            )

    async def listing(
        self, identity: UUID, key: str, *, view: Literal["base", "draft"] = "draft"
    ) -> GitTransactionListing:
        async with self.runtime.admission():
            record, snapshot = await self.call(
                partial(self.repository.read, identity, key, view=view)
            )
            return GitTransactionListing(
                transaction=self.repository.status_value(record),
                view=view,
                notes=[
                    GitTransactionNoteInfo(
                        external_id=snapshot.manifest.note_ids[path],
                        path=path,
                        checksum=hashlib.sha256(content.encode()).hexdigest(),
                    )
                    for path, content in sorted(snapshot.files.items())
                ],
            )

    async def read(
        self, identity: UUID, key: str, identifier: str, *, view: Literal["base", "draft"] = "draft"
    ) -> GitTransactionRead:
        async with self.runtime.admission():
            record, snapshot = await self.call(
                partial(self.repository.read, identity, key, view=view)
            )
            # Resolve only inside the fixed tree. Titles are not unique addresses.
            paths = [
                path
                for path, content in snapshot.files.items()
                if identifier
                in {
                    path,
                    str(snapshot.manifest.note_ids[path]),
                    parse_frontmatter(content)["permalink"],
                }
            ]
            if len(paths) != 1:
                raise GitTransactionError(
                    404 if not paths else 409,
                    "Draft note is missing or its identifier is ambiguous",
                )
            path = paths[0]
            content = snapshot.files[path]
            return GitTransactionRead(
                transaction=self.repository.status_value(record),
                view=view,
                note=GitTransactionNote(
                    external_id=snapshot.manifest.note_ids[path],
                    path=path,
                    checksum=hashlib.sha256(content.encode()).hexdigest(),
                    content=content,
                ),
            )

    async def _entity_times(self, note_id: UUID) -> tuple[datetime | None, datetime | None]:
        """The indexed times of a note, for one that has no `created`/`modified` yet (BM-010)."""
        async with self.runtime.pipeline.session_maker() as session:
            entity = await session.scalar(
                select(Entity).where(
                    Entity.external_id == str(note_id),
                    Entity.project_id == self.runtime.pipeline.project_id,
                )
            )
        return (entity.created_at, entity.updated_at) if entity is not None else (None, None)

    async def _stamp_times(
        self, content: str, *, current: str | None, note_id: UUID | None, now: datetime
    ) -> str:
        created, modified = await self._entity_times(note_id) if note_id else (None, None)
        wanted = wanted_note_times(
            content,
            now=now,
            current_markdown=current,
            fallback_created=created,
            fallback_modified=modified,
        )
        return content if wanted is None else stamp_markdown_times(content, wanted)

    async def write(
        self,
        identity: UUID,
        key: str,
        request: GitTransactionWrite | GitTransactionMove | GitTransactionDelete,
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            record, snapshot = await self.call(partial(self.repository.read, identity, key))
            self.repository.require_writable(record, request.expected_revision)
            files, note_ids = dict(snapshot.files), snapshot.manifest.note_ids
            if isinstance(request, GitTransactionWrite):
                content = request.content
                if request.operation != "replace":
                    if request.path not in files:
                        raise GitTransactionError(404, "Draft note does not exist")
                    content = apply_edit_operation(files[request.path], request.operation, content)
                # Trigger: the text replaces a draft note that already has a permalink.
                # Why: a permalink changes only by design, as for `bm_write` and `bm_edit`.
                # Outcome: a different `permalink:` line is refused; a missing one is not.
                if request.path in files:
                    try:
                        require_permalink_unchanged(
                            _draft_permalink(files[request.path]), _draft_permalink(content)
                        )
                    except PermalinkChangeError as error:
                        raise GitTransactionError(400, str(error)) from error
                # Trigger: the draft text is the only durable copy once committed.
                # Why: without `created`/`modified` in it, a rebuild dates the note anew.
                # Outcome: both keys are stamped by the same rules as an ordinary write.
                files[request.path] = await self._stamp_times(
                    content,
                    current=files.get(request.path),
                    note_id=note_ids.get(request.path),
                    now=datetime.now().astimezone(),
                )
                note_ids.setdefault(request.path, uuid4())
            else:
                if request.path not in files:
                    raise GitTransactionError(404, "Draft note does not exist")
                if isinstance(request, GitTransactionMove):
                    if request.destination in files:
                        raise GitTransactionError(409, "Draft destination is already occupied")
                    # A plain move keeps both times; a note without them gets its indexed ones.
                    moved = files[request.path]
                    files[request.destination] = await self._stamp_times(
                        moved,
                        current=moved,
                        note_id=note_ids[request.path],
                        now=datetime.now().astimezone(),
                    )
                    note_ids[request.destination] = note_ids[request.path]
                del files[request.path]
                del note_ids[request.path]
            await validate_git_draft(files, self.runtime.pipeline.file_service.base_path)
            return await self.call(
                partial(
                    self.repository.replace,
                    identity,
                    key,
                    expected_revision=request.expected_revision,
                    files=files,
                    note_ids=note_ids,
                )
            )

    async def commit(
        self, identity: UUID, key: str, request: GitTransactionCommit
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            try:
                current = await self.call(partial(self.repository.status, identity, key))
                if current.state != "committed":
                    record, snapshot = await self.call(partial(self.repository.read, identity, key))
                    self.repository.require_writable(record, request.expected_revision)
                    await validate_git_draft(
                        snapshot.files, self.runtime.pipeline.file_service.base_path
                    )
                return await self.call(
                    partial(
                        self.repository.commit,
                        identity,
                        key,
                        expected_revision=request.expected_revision,
                        message=request.message,
                    )
                )
            finally:
                # A lost/cancelled response may have published. Only Git is input
                # to the durable worker, so waking it cannot index private drafts.
                self.runtime.worker.notify()

    async def rollback(
        self, identity: UUID, key: str, request: GitTransactionControl
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            return await self.call(
                partial(
                    self.repository.rollback,
                    identity,
                    key,
                    expected_revision=request.expected_revision,
                )
            )
