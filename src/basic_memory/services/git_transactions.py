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
from basic_memory.repository.git_host_identities import GitNoteOwnedElsewhere
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
from basic_memory.schemas.git_project import validate_note_path
from basic_memory.services.note_preparation import (
    PermalinkChangeError,
    apply_edit_operation,
    require_permalink_unchanged,
)
from basic_memory.services.note_times import stamp_markdown_times, wanted_note_times

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from basic_memory.services.git_project_runtime import GitProjectRuntime


# Notes that passed validation, by (root, path, sha256 of the text) -> their permalink.
# Every draft write and commit validates the whole tree; in a project of a thousand notes
# the parse cost a second each time, and a folder moved note by note paid it twice per
# note. A note is only skipped once its exact bytes passed; uniqueness is checked always.
_VALIDATED: dict[tuple[str, str, str], str] = {}
_VALIDATED_LIMIT = 16384


async def validate_git_draft(files: Mapping[str, str], root: Path) -> None:
    """Parse the complete proposed tree without consulting mutable SQL/disk notes."""
    parser = EntityParser(root)
    permalinks: set[str] = set()
    passed: dict[tuple[str, str, str], str] = {}
    for path, content in files.items():
        key = (str(root), path, hashlib.sha256(content.encode("utf-8")).hexdigest())
        try:
            permalink = _VALIDATED.get(key)
            if permalink is None:
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
            if key not in _VALIDATED:
                await parser.parse_markdown_content(
                    file_path=Path(path), content=content, mtime=0, ctime=0
                )
                passed[key] = permalink
        except (ParseError, ValueError) as error:
            raise GitTransactionError(422, f"Invalid draft note {path}: {error}") from error
    if len(_VALIDATED) + len(passed) > _VALIDATED_LIMIT:
        _VALIDATED.clear()
    _VALIDATED.update(passed)


def _draft_permalink(text: str) -> str | None:
    """The `permalink:` of a draft text; None when there is none or the frontmatter is unreadable."""
    try:
        value = parse_frontmatter(text).get("permalink")
    except ParseError:  # draft validation reports unreadable frontmatter on its own
        return None
    return value if isinstance(value, str) and value else None


class GitNoteStillOwned(GitTransactionError):
    """A commit would publish a note UUID another project (or open move) still holds.

    A 409 like any refused commit; a cross-project move reads it as "the source has not
    let go yet" and waits, instead of rebasing its draft.
    """


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
            return await self.start_admitted(request)

    async def start_admitted(self, request: GitTransactionStart) -> GitTransactionStarted:
        """`start` for a caller that already holds this runtime's admission gate."""
        return await self.call(partial(self.repository.start, request))

    async def status(
        self, identity: UUID, key: str, *, keep_alive: bool = False
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            return await self.status_admitted(identity, key, keep_alive=keep_alive)

    async def status_admitted(
        self, identity: UUID, key: str, *, keep_alive: bool = False
    ) -> GitTransactionStatus:
        """`status` for a caller that already holds this runtime's admission gate."""
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

    # --- Internal: a draft note with a chosen UUID (I1), never routed ---

    async def write_identified(
        self,
        identity: UUID,
        key: str,
        *,
        path: str,
        content: str,
        note_id: UUID,
        expected_revision: int,
        created: datetime | None = None,
        modified: datetime | None = None,
        reservation: UUID | None = None,
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            return await self.write_identified_admitted(
                identity,
                key,
                path=path,
                content=content,
                note_id=note_id,
                expected_revision=expected_revision,
                created=created,
                modified=modified,
                reservation=reservation,
            )

    async def write_identified_admitted(
        self,
        identity: UUID,
        key: str,
        *,
        path: str,
        content: str,
        note_id: UUID,
        expected_revision: int,
        created: datetime | None = None,
        modified: datetime | None = None,
        reservation: UUID | None = None,
    ) -> GitTransactionStatus:
        """Add a new draft note that keeps an existing UUID (a cross-project move).

        The public write API never takes a UUID (GitTransactionWrite has none), so agents
        cannot choose one. The note keeps its times as a plain move does: `created` and
        `modified` stay as written, and a missing key is filled from `created`/`modified`
        here (the source entity's times), never from the moment of the move.
        The ownership check runs at commit, not here: at draft time the note still lives
        in its source project. Only a reservation by another move refuses it now.
        """
        try:
            validate_note_path(path)
        except ValueError as error:
            raise GitTransactionError(422, f"Invalid draft note path {path}: {error}") from error
        if (identities := self.runtime.pipeline.identities) is not None:
            try:
                identities.require_unreserved({note_id}, holder=reservation)
            except GitNoteOwnedElsewhere as error:
                raise GitTransactionError(409, str(error)) from error
        record, snapshot = await self.call(partial(self.repository.read, identity, key))
        self.repository.require_writable(record, expected_revision)
        files, note_ids = dict(snapshot.files), snapshot.manifest.note_ids
        if path in files:
            raise GitTransactionError(409, "Draft path is already occupied")
        if note_id in note_ids.values():
            raise GitTransactionError(409, "Draft already holds this note UUID")
        wanted = wanted_note_times(
            content,
            now=datetime.now().astimezone(),
            current_markdown=content,
            fallback_created=created,
            fallback_modified=modified,
        )
        files[path] = content if wanted is None else stamp_markdown_times(content, wanted)
        note_ids[path] = note_id
        await validate_git_draft(files, self.runtime.pipeline.file_service.base_path)
        return await self.call(
            partial(
                self.repository.replace,
                identity,
                key,
                expected_revision=expected_revision,
                files=files,
                note_ids=note_ids,
            )
        )

    async def commit(
        self,
        identity: UUID,
        key: str,
        request: GitTransactionCommit,
        *,
        moved_notes: Mapping[UUID, UUID] | None = None,
        reservation: UUID | None = None,
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            return await self.commit_admitted(
                identity, key, request, moved_notes=moved_notes, reservation=reservation
            )

    async def commit_admitted(
        self,
        identity: UUID,
        key: str,
        request: GitTransactionCommit,
        *,
        moved_notes: Mapping[UUID, UUID] | None = None,
        reservation: UUID | None = None,
    ) -> GitTransactionStatus:
        """`commit` for a caller that already holds this runtime's admission gate.

        `moved_notes` (note -> source project) adds BM's move trailers; `reservation` is
        the cross-project move allowed to publish the UUIDs it reserves.
        """
        try:
            current = await self.call(partial(self.repository.status, identity, key))
            if current.state != "committed":
                record, snapshot = await self.call(partial(self.repository.read, identity, key))
                self.repository.require_writable(record, request.expected_revision)
                await validate_git_draft(
                    snapshot.files, self.runtime.pipeline.file_service.base_path
                )
                base = await self.call(
                    partial(self.repository.repository.note_ids_at, record.base_commit)
                )
                await self._require_owned_here(
                    set(snapshot.manifest.note_ids.values()) - base, reservation
                )
            return await self.call(
                partial(
                    self.repository.commit,
                    identity,
                    key,
                    expected_revision=request.expected_revision,
                    message=request.message,
                    moved_notes=moved_notes,
                )
            )
        finally:
            # A lost/cancelled response may have published. Only Git is input
            # to the durable worker, so waking it cannot index private drafts.
            self.runtime.worker.notify()

    async def _require_owned_here(self, new_ids: set[UUID], reservation: UUID | None) -> None:
        """Ownership guard: a draft may not publish a UUID another project still holds.

        Agent drafts get fresh UUIDs, so this refuses only a chosen one (a cross-project
        move whose source has not let go yet). Publishing it anyway would leave the
        twin in this project waiting for the other for good.
        """
        identities = self.runtime.pipeline.identities
        project_id = self.runtime.pipeline.project_id
        async with self.runtime.pipeline.session_maker() as session:
            try:
                if identities is not None:
                    await identities.require_unowned(
                        session, project_id=project_id, note_ids=new_ids, holder=reservation
                    )
                    return
                owned = await session.scalar(
                    select(Entity.external_id)
                    .where(
                        Entity.external_id.in_([str(note) for note in new_ids]),
                        Entity.project_id != project_id,
                    )
                    .limit(1)
                )
                if owned is not None:
                    raise GitNoteOwnedElsewhere(UUID(owned), "another project")
            except GitNoteOwnedElsewhere as error:
                raise GitNoteStillOwned(409, str(error)) from error

    async def rollback(
        self, identity: UUID, key: str, request: GitTransactionControl
    ) -> GitTransactionStatus:
        async with self.runtime.admission():
            return await self.rollback_admitted(identity, key, request)

    async def rollback_admitted(
        self, identity: UUID, key: str, request: GitTransactionControl
    ) -> GitTransactionStatus:
        """`rollback` for a caller that already holds this runtime's admission gate."""
        return await self.call(
            partial(
                self.repository.rollback,
                identity,
                key,
                expected_revision=request.expected_revision,
            )
        )
