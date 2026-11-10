"""Move one note into another project on a Git host and keep its UUID (I1, W2).

A move across projects publishes in two repositories, so it cannot be one commit. The
order keeps the note's UUID unique host-wide at every moment:

0. a draft in the target holds the note under its old UUID (nothing is published);
   the intent, written next to the target repository, records how to finish;
b. the source deletes the note (an ordinary guarded delete, with the checksum read);
c. the target commits the draft, with BM's move trailer as proof for peers.

The target's admission gate is held from (0) through (c), and moves are serialized by
the host's move lock, so no local write, transaction or peer sync of the target can take
the path, the permalink or the UUID in between: the window in which the note is in no
project can only stay open after a crash. Then the intent finishes the move at the next
host start (or POST /v2/git/moves/resume), deciding from the source's published history
whether (b) happened, never from an answer or from the source head alone.

A move that cannot finish as planned is kept: "pending" (the target could not commit
yet; resume retries) or "stuck" (after a crash the planned path was taken, or the UUID
came back into the source). Its bytes stay in the intent and in the source history, and
a person chooses where the note goes (POST /v2/git/moves/{move_id}/resolve).
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import select

from basic_memory.file_utils import drop_frontmatter_key, parse_frontmatter
from basic_memory.indexing.accepted_note_mutation_runner import (
    AcceptedNoteMutationRejected,
    load_accepted_note_mutation_project,
    prepare_create_or_reject,
)
from basic_memory.markdown.note_lock import LOCKED_NOTE_MESSAGE, note_is_locked
from basic_memory.models import Entity, NoteContent
from basic_memory.repository.entity_repository import EntityRepository
from basic_memory.repository.git_host_identities import GitHostIdentities
from basic_memory.repository.git_move_intent_repository import (
    GitMoveIntentError,
    GitMoveIntentRepository,
)
from basic_memory.repository.git_project_repository import GitSnapshot, quote_in_message
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.repository.git_transaction_repository import GitTransactionError
from basic_memory.schemas.base import Entity as EntitySchema
from basic_memory.schemas.git_moves import (
    GitCrossProjectMoveRequest,
    GitCrossProjectMoveResult,
    GitMoveIntent,
    GitMoveIntentView,
    GitMoveListing,
    GitMoveOutcomeState,
    GitMoveResolve,
    GitMoveResumeKind,
    GitMoveResumeOutcome,
)
from basic_memory.schemas.git_transactions import (
    GitTransactionCommit,
    GitTransactionControl,
    GitTransactionStart,
    GitTransactionStatus,
)
from basic_memory.services.git_transactions import GitNoteStillOwned
from basic_memory.services.note_content_writes import (
    NoteContentMutationService,
    NoteContentMutationServiceError,
)
from basic_memory.services.note_times import stamp_markdown_times, wanted_note_times
from basic_memory.utils import portable_path_key

if TYPE_CHECKING:  # pragma: no cover
    from basic_memory.index.git_host import GitHost
    from basic_memory.services.git_project_runtime import GitProjectRuntime

# The journal's source label of the source delete.
MOVE_SOURCE = "api-cross-project-move"
UNCHANGED = "Nothing was moved."
# A draft rebuilt this often without a commit means the target keeps changing.
COMMIT_ATTEMPTS = 3


class GitMoveRefused(Exception):
    """Nothing was moved: the note is where it was (an HTTP 4xx or 503)."""

    def __init__(self, status_code: int, kind: str, message: str, **detail: str | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.message = message
        self.detail = detail

    def body(self) -> dict[str, str | None]:
        return {"message": self.message, "kind": self.kind, **self.detail}


class _Stuck(Exception):
    """The target cannot take the note as planned; a person must choose."""


@dataclass(frozen=True, slots=True)
class _SourceNote:
    external_id: UUID
    title: str
    path: str
    checksum: str
    markdown: str
    permalink: str | None
    created: datetime | None
    modified: datetime | None
    base_commit: str


@dataclass(frozen=True, slots=True)
class _Canonical:
    path: str
    content: str
    permalink: str


# --- Helpers shared by move, resume and resolve ---


def _runtime(host: GitHost, project_id: UUID) -> GitProjectRuntime | None:
    return host.runtimes.get(str(project_id))


def _identities(host: GitHost) -> GitHostIdentities:
    if host.identities is None:  # pragma: no cover - run_git_host always builds them
        raise RuntimeError("Git host has no ownership view")
    return host.identities


def _intents(runtime: GitProjectRuntime) -> GitMoveIntentRepository:
    return GitMoveIntentRepository(
        runtime.pipeline.repository, UUID(runtime.pipeline.project_external_id)
    )


def _aware(value: datetime | None) -> datetime | None:
    # SQLite hands back naive local times; note times treat them as local (BM-010).
    return value if value is None or value.tzinfo is not None else value.astimezone()


async def _settled[T](operation: Coroutine[Any, Any, T]) -> T:
    """Run past a cancelled caller: the draft, delete and commit belong together.

    A cancelled request would otherwise release the target gate between the source
    delete and the target commit, the window the gate is held to close. The caller
    still receives its cancellation once the move has reached its outcome.
    """
    task = asyncio.ensure_future(operation)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
    if cancelled is not None:
        if not task.cancelled():
            task.exception()
        raise cancelled
    return task.result()


def _project_name(runtime: GitProjectRuntime) -> str:
    identities = runtime.pipeline.identities
    project = identities.projects.get(runtime.pipeline.project_id) if identities else None
    return project.name if project is not None else runtime.pipeline.project_external_id


def _commit_message(intent: GitMoveIntent, source: GitProjectRuntime | None) -> str:
    """One line a person reading `git log` understands; caller text is quoted."""
    project = _project_name(source) if source is not None else str(intent.source_project_id)
    return (
        f"Move {quote_in_message(intent.source_path)} from project {quote_in_message(project)} "
        f"to {quote_in_message(intent.target_path)}"
    )


def _result(
    intent: GitMoveIntent,
    state: GitMoveOutcomeState,
    *,
    target_commit: str | None = None,
    index_pending: bool = False,
    message: str | None = None,
) -> GitCrossProjectMoveResult:
    return GitCrossProjectMoveResult(
        state=state,
        move_id=intent.move_id,
        external_id=intent.external_id,
        source_project_id=intent.source_project_id,
        target_project_id=intent.target_project_id,
        previous_file_path=intent.source_path,
        file_path=intent.target_path,
        previous_permalink=intent.source_permalink,
        permalink=intent.target_permalink,
        source_commit=intent.source_commit,
        target_commit=target_commit,
        index_pending=index_pending,
        message=message,
    )


# --- One move ---


async def move_note_across_projects(
    host: GitHost,
    source: GitProjectRuntime,
    request: GitCrossProjectMoveRequest,
    mutations: NoteContentMutationService,
) -> GitCrossProjectMoveResult:
    """Move one note of `source` into `request.target_project_id`, keeping its UUID.

    Raises GitMoveRefused when nothing was moved. A result is "completed", or "pending" /
    "stuck" with its intent kept (the note is then in no project's current state until
    resume or a person finishes it).
    """
    target = _runtime(host, request.target_project_id)
    if target is None:
        raise GitMoveRefused(
            404,
            "not_found",
            f"Project {request.target_project_id} is not on this Git host. {UNCHANGED}",
        )
    if target is source:
        raise GitMoveRefused(
            422,
            "same_project",
            "The target is the source project; a move inside one project keeps the id "
            f"anyway. {UNCHANGED}",
        )
    async with host.move_lock:
        try:
            async with target.admission():
                return await _move_admitted(host, source, target, request, mutations)
        except GitProjectionConflict as error:
            # Only the admission and the checks raise this: every later step answers with a
            # result or a refusal of its own.
            raise GitMoveRefused(409, "not_accepting", f"{error}. {UNCHANGED}") from error


async def _move_admitted(
    host: GitHost,
    source: GitProjectRuntime,
    target: GitProjectRuntime,
    request: GitCrossProjectMoveRequest,
    mutations: NoteContentMutationService,
) -> GitCrossProjectMoveResult:
    # --- Checks: nothing is written yet ---
    # Target SQL first follows its head, so the title and permalink checks see every
    # note; a target that waits for other projects' notes still takes this one.
    await target.catch_up_admitted()
    base = await asyncio.to_thread(target.pipeline.repository.snapshot)
    note = await _source_note(source, request)
    identities = _identities(host)
    if (holder := identities.reservations.get(note.external_id)) is not None:
        raise GitMoveRefused(
            409,
            "reserved",
            f"Note {note.external_id} is part of the open cross-project move {holder}; "
            f"finish or resolve it first (GET /v2/git/moves). {UNCHANGED}",
        )
    await _require_unique_title(target, note.title)
    canonical = await _canonical(
        target,
        title=note.title,
        content=note.markdown,
        path=request.destination_path,
        created=note.created,
        modified=note.modified,
    )
    _require_free(base, canonical.path, note.external_id)

    # --- Draft (0) and intent: still nothing published ---
    move_id = uuid4()
    return await _settled(
        _publish(host, source, target, note, canonical, move_id, mutations, request.catch_up, base)
    )


async def _publish(
    host: GitHost,
    source: GitProjectRuntime,
    target: GitProjectRuntime,
    note: _SourceNote,
    canonical: _Canonical,
    move_id: UUID,
    mutations: NoteContentMutationService,
    catch_up: bool,
    base: GitSnapshot,
) -> GitCrossProjectMoveResult:
    identities = _identities(host)
    intents = _intents(target)
    transactions = target.transactions
    identities.reservations[note.external_id] = move_id
    started = None
    intent: GitMoveIntent | None = None
    try:
        started = await transactions.start_admitted(GitTransactionStart())
        await transactions.write_identified_admitted(
            started.transaction_id,
            started.transaction_key,
            path=canonical.path,
            content=canonical.content,
            note_id=note.external_id,
            expected_revision=0,
            created=note.created,
            modified=note.modified,
            reservation=move_id,
        )
        intent = GitMoveIntent(
            move_id=move_id,
            created_at=datetime.now().astimezone(),
            source_project_id=UUID(source.pipeline.project_external_id),
            source_path=note.path,
            source_checksum=note.checksum,
            source_base_commit=note.base_commit,
            source_permalink=note.permalink,
            source_created=note.created,
            source_modified=note.modified,
            target_project_id=UUID(target.pipeline.project_external_id),
            external_id=note.external_id,
            target_path=canonical.path,
            title=note.title,
            target_content=canonical.content,
            target_permalink=canonical.permalink,
            transaction_id=started.transaction_id,
            transaction_key=started.transaction_key,
        )
        await asyncio.to_thread(intents.save, intent)
    except BaseException as error:
        # Before the intent exists the source is untouched: undo the draft, refuse.
        identities.reservations.pop(note.external_id, None)
        if intent is not None:
            await _forget(intents, intent)
        if started is not None:
            await _drop_draft(target, started.transaction_id, started.transaction_key)
        if isinstance(error, GitTransactionError):
            raise GitMoveRefused(
                error.status_code, "invalid_draft", f"{error}. {UNCHANGED}"
            ) from error
        raise

    # --- Source delete (b) ---
    try:
        removal = await _delete_source(source, note, mutations)
    except GitMoveRefused:
        identities.reservations.pop(note.external_id, None)
        await _drop_draft(target, intent.transaction_id, intent.transaction_key)
        await _forget(intents, intent)
        raise
    # --- Target commit (c) ---
    # From here on the source has let go: no error may read as "nothing was moved".
    try:
        intent = intent.model_copy(update={"state": "source_deleted", "source_commit": removal})
        await asyncio.to_thread(intents.save, intent)
        # Nothing could publish into the target meanwhile (its gate is held): the draft's
        # base is the head the checks read, and its notes prove themselves to the catch-up.
        return await _commit(host, source, target, intent, catch_up=catch_up, base=base)
    except Exception as error:
        logger.exception("Cross-project move stopped after the source delete", error=str(error))
        return _result(
            intent,
            "pending",
            message=(
                f"The move stopped after the source deleted the note ({error}); it finishes "
                "at the next host start or with POST /v2/git/moves/resume"
            ),
        )


async def _source_note(
    source: GitProjectRuntime, request: GitCrossProjectMoveRequest
) -> _SourceNote:
    """The note as SQL and the source head both hold it; refuse anything else.

    A Git project's notes are Markdown by its manifest's path rule, so no other file
    kind can be addressed here.
    """
    async with source.pipeline.session_maker() as session:
        entity = await session.scalar(
            select(Entity).where(
                Entity.project_id == source.pipeline.project_id,
                Entity.external_id == str(request.external_id),
            )
        )
        content = await session.get(NoteContent, entity.id) if entity is not None else None
    if entity is None or content is None:
        raise GitMoveRefused(
            404,
            "not_found",
            f"Note {request.external_id} is not in project '{_project_name(source)}'. {UNCHANGED}",
        )
    if request.expected_checksum is not None and content.db_checksum != request.expected_checksum:
        raise GitMoveRefused(
            409,
            "revision_conflict",
            f"'{entity.file_path}' changed since it was read. {UNCHANGED}",
            db_checksum=content.db_checksum,
        )
    # Trigger: the note carries `locked: true`.
    # Why: the source refuses to delete it, so the move would stop half-way.
    # Outcome: refused before anything is written, as a move within a project is.
    if note_is_locked(content.markdown_content):
        raise GitMoveRefused(423, "locked", f"{LOCKED_NOTE_MESSAGE} {UNCHANGED}")
    head = await asyncio.to_thread(source.pipeline.repository.snapshot)
    if (
        head.manifest.note_ids.get(entity.file_path) != request.external_id
        or head.files.get(entity.file_path) != content.markdown_content
    ):
        # The draft must hold exactly the bytes the delete removes.
        raise GitMoveRefused(
            409,
            "not_accepting",
            f"Project '{_project_name(source)}' has not caught up with its published "
            f"notes yet; repeat the move in a moment. {UNCHANGED}",
        )
    return _SourceNote(
        external_id=request.external_id,
        title=entity.title,
        path=entity.file_path,
        checksum=content.db_checksum,
        markdown=content.markdown_content,
        permalink=entity.permalink,
        created=_aware(entity.created_at),
        modified=_aware(entity.updated_at),
        base_commit=head.commit_id,
    )


async def _require_unique_title(target: GitProjectRuntime, title: str) -> None:
    """P2: a second note with the title would make `[[Title]]` ambiguous in the target."""
    async with target.pipeline.session_maker() as session:
        twins = await EntityRepository(
            project_id=target.pipeline.project_id
        ).get_by_equivalent_title(session, title)
    if twins:
        twin = twins[0]
        raise GitMoveRefused(
            409,
            "duplicate_title",
            f"A note titled '{twin.title}' already exists at '{twin.file_path}' in project "
            f"'{_project_name(target)}'. {UNCHANGED}",
            title=twin.title,
            file_path=twin.file_path,
            external_id=twin.external_id,
        )


async def _canonical(
    target: GitProjectRuntime,
    *,
    title: str,
    content: str,
    path: str,
    created: datetime | None,
    modified: datetime | None,
) -> _Canonical:
    """The bytes an ordinary create of this note in the target would accept.

    The permalink is formed again from the target path (one of the source project
    would point into the wrong project), with the '-1' suffix a taken one gets. The
    source entity's times go into a note that lacks them before the preparer runs, which
    would otherwise date the note with the move (BM-010).
    """
    text = drop_frontmatter_key(content, "permalink")
    wanted = wanted_note_times(
        text,
        now=datetime.now().astimezone(),
        current_markdown=text,
        fallback_created=created,
        fallback_modified=modified,
    )
    if wanted is not None:
        text = stamp_markdown_times(text, wanted)
    location = PurePosixPath(path)
    data = EntitySchema(
        title=title,
        directory="" if str(location.parent) == "." else str(location.parent),
        file_name=location.name,
        content_type="text/markdown",
        content=text,
    )
    async with target.pipeline.session_maker() as session:
        try:
            project = await load_accepted_note_mutation_project(
                session,
                project_external_id=target.pipeline.project_external_id,
                dependencies=target.dependencies,
            )
            preparer = target.dependencies.preparer_factory.create_note_preparer(project)
            prepared = await prepare_create_or_reject(
                preparer, data, check_storage_exists=True, session=session, authorship=None
            )
        except AcceptedNoteMutationRejected as rejected:
            status = rejected.rejection.kind.http_status_code
            raise GitMoveRefused(
                status,
                "occupied" if status == 409 else "invalid_path",
                f"{rejected.rejection.detail} {UNCHANGED}",
            ) from rejected
        finally:
            await session.rollback()
    final = prepared.prepared.file_path.as_posix()
    if final != path:
        raise GitMoveRefused(
            422,
            "invalid_path",
            f"'{path}' would be stored as '{final}'; name that path. {UNCHANGED}",
        )
    markdown = prepared.prepared.markdown_content
    return _Canonical(final, markdown, str(parse_frontmatter(markdown)["permalink"]))


def _require_free(base: GitSnapshot, path: str, external_id: UUID) -> None:
    key = portable_path_key(path)
    if any(portable_path_key(other) == key for other in base.files):
        raise GitMoveRefused(
            409, "occupied", f"A note already exists at '{path}' in the target. {UNCHANGED}"
        )
    if external_id in base.manifest.note_ids.values():
        raise GitMoveRefused(
            409, "already_in_target", f"Note {external_id} is already in the target. {UNCHANGED}"
        )


async def _delete_source(
    source: GitProjectRuntime, note: _SourceNote, mutations: NoteContentMutationService
) -> str:
    """Delete the note in the source; the commit that removed it, or GitMoveRefused.

    The answer alone never decides: a refused or failed delete may still have published
    (a lost answer, "Git published; SQL pending"), and only the source's history knows.
    """
    failure: Exception | None = None
    try:
        await mutations.delete_note(
            project_external_id=source.pipeline.project_external_id,
            entity_external_id=str(note.external_id),
            source=MOVE_SOURCE,
            base_checksum=note.checksum,
        )
    except Exception as error:  # decided from Git below
        failure = error
    repository = source.pipeline.repository
    head = await asyncio.to_thread(repository.head)
    removals = await asyncio.to_thread(
        repository.removals_since, note.base_commit, head, note.external_id
    )
    if not removals:
        # The source still holds the note as it was read: nothing was moved.
        match failure:
            case NoteContentMutationServiceError(status_code=status, detail=dict() as detail):
                message = str(detail.get("message") or "The source refused the delete")
                extra = {k: v for k, v in detail.items() if k not in {"message", "kind"}}
                raise GitMoveRefused(
                    status,
                    str(detail.get("kind") or "refused_by_source"),
                    f"{message.rstrip('.')}. {UNCHANGED}",
                    **extra,
                ) from failure
            case NoteContentMutationServiceError(status_code=status, detail=str() as detail):
                raise GitMoveRefused(
                    status, "refused_by_source", f"{detail.rstrip('.')}. {UNCHANGED}"
                ) from failure
            case None:
                raise GitMoveRefused(
                    409, "not_accepting", f"The source did not delete the note. {UNCHANGED}"
                )
            case _:
                raise GitMoveRefused(
                    503, "source_failed", f"The source delete failed: {failure}. {UNCHANGED}"
                ) from failure
    if failure is not None:
        # Trigger: the delete published, but its SQL did not follow (or the answer was lost).
        # Why: the target commit refuses a UUID the source's SQL still holds.
        # Outcome: the source catches up now, under its own gate (target -> source).
        logger.warning(
            "Cross-project move: source delete published but answered with an error",
            note=str(note.external_id),
            error=str(failure),
        )
        await _catch_up_quietly(source)
    return removals[0]


async def _catch_up_quietly(runtime: GitProjectRuntime) -> None:
    try:
        await runtime.catch_up()
    except Exception as error:  # the commit guard reports what is still missing
        logger.warning(
            "Cross-project move: catch-up failed",
            project=runtime.pipeline.project_external_id,
            error=str(error),
        )


# --- The target commit, with rebuilds; shared by move, resume and resolve ---


async def _commit(
    host: GitHost,
    source: GitProjectRuntime | None,
    target: GitProjectRuntime,
    intent: GitMoveIntent,
    *,
    catch_up: bool,
    base: GitSnapshot | None = None,
) -> GitCrossProjectMoveResult:
    """Commit the target draft (the target's gate is held); keep the intent if it cannot.

    `base` is the published snapshot the draft was built on, when the caller read it.
    """
    intents = _intents(target)
    caught_up_source = False
    status: GitTransactionStatus | None = None
    for _attempt in range(COMMIT_ATTEMPTS):
        try:
            status = await target.transactions.commit_admitted(
                intent.transaction_id,
                intent.transaction_key,
                GitTransactionCommit(expected_revision=1, message=_commit_message(intent, source)),
                moved_notes={intent.external_id: intent.source_project_id},
                reservation=intent.move_id,
            )
            break
        except GitNoteStillOwned as error:
            # Trigger: the source's SQL (or head) still holds the UUID.
            # Why: its catch-up after the delete failed or has not run yet.
            # Outcome: catch it up once and retry; otherwise pending, resume retries.
            if source is None or caught_up_source:
                return await _keep(intents, intent, "pending", f"{error}; the move resumes later")
            caught_up_source = True
            await _catch_up_quietly(source)
        except GitTransactionError as error:
            current = await _draft_status(target, intent)
            if current is not None and current.state == "committed":
                status = current
                break
            if current is not None and current.state not in {
                "conflicted",
                "expired",
                "expired_cleaned",
                "rolled_back",
                "frozen",
            }:
                return await _keep(intents, intent, "pending", f"{error}; the move resumes later")
            # The draft can no longer commit (the target advanced, or the draft expired or
            # was cleaned while the host was down): rebuild it from the intent's bytes.
            try:
                intent, base = await _rebuild(host, target, intent)
            except _Stuck as stuck:
                return await _keep(intents, intent, "stuck", str(stuck))
    else:
        return await _keep(
            intents, intent, "pending", "The target kept changing; the move resumes later"
        )
    assert status is not None and status.published_commit is not None
    return await _completed(host, target, intent, status.published_commit, base, catch_up)


async def _completed(
    host: GitHost,
    target: GitProjectRuntime,
    intent: GitMoveIntent,
    target_commit: str,
    base: GitSnapshot | None,
    catch_up: bool,
) -> GitCrossProjectMoveResult:
    """The note is published in the target: the intent goes, SQL follows if asked."""
    await _forget(_intents(target), intent)
    _identities(host).reservations.pop(intent.external_id, None)
    if not catch_up:
        return _result(intent, "completed", target_commit=target_commit, index_pending=True)
    # Trigger: the move is published; only the target's SQL may still lag.
    # Why: a failure here is no failed move; the worker catches up on its own.
    # Outcome: "completed" with index_pending, never an error.
    try:
        deferral = await target.catch_up_admitted(known_ancestor=base)
    except Exception as error:
        logger.warning("Cross-project move: target catch-up failed", error=str(error))
        return _result(
            intent,
            "completed",
            target_commit=target_commit,
            index_pending=True,
            message=f"The target's index has not caught up yet: {error}",
        )
    if deferral is not None:
        return _result(
            intent,
            "completed",
            target_commit=target_commit,
            index_pending=True,
            message=f"The target project waits for other notes: {deferral}",
        )
    return _result(intent, "completed", target_commit=target_commit)


async def _rebuild(
    host: GitHost,
    target: GitProjectRuntime,
    intent: GitMoveIntent,
    *,
    path: str | None = None,
) -> tuple[GitMoveIntent, GitSnapshot]:
    """A new draft from the intent's bytes, its pointer saved before the old one goes.

    The canonical bytes are formed again (a permalink taken meanwhile gets the '-1'
    suffix, as for any new note). A planned path taken meanwhile cannot be: _Stuck.
    """
    await target.catch_up_admitted()
    path = path or intent.target_path
    base = await asyncio.to_thread(target.pipeline.repository.snapshot)
    try:
        _require_free(base, path, intent.external_id)
        canonical = await _canonical(
            target,
            title=intent.title,
            content=intent.target_content,
            path=path,
            created=intent.source_created,
            modified=intent.source_modified,
        )
    except GitMoveRefused as refused:
        reason = refused.message.removesuffix(UNCHANGED).strip()
        raise _Stuck(f"The target cannot take the note at '{path}': {reason}") from refused
    transactions = target.transactions
    started = await transactions.start_admitted(GitTransactionStart())
    try:
        await transactions.write_identified_admitted(
            started.transaction_id,
            started.transaction_key,
            path=canonical.path,
            content=canonical.content,
            note_id=intent.external_id,
            expected_revision=0,
            created=intent.source_created,
            modified=intent.source_modified,
            reservation=intent.move_id,
        )
    except GitTransactionError as error:
        await _drop_draft(target, started.transaction_id, started.transaction_key)
        raise _Stuck(f"The target cannot take the note at '{path}': {error}") from error
    rebuilt = intent.model_copy(
        update={
            "transaction_id": started.transaction_id,
            "transaction_key": started.transaction_key,
            "target_path": canonical.path,
            "target_content": canonical.content,
            "target_permalink": canonical.permalink,
        }
    )
    # The new pointer first: a crash before the old draft is dropped still finds it.
    await asyncio.to_thread(_intents(target).save, rebuilt)
    await _drop_draft(target, intent.transaction_id, intent.transaction_key)
    return rebuilt, base


async def _draft_status(
    target: GitProjectRuntime, intent: GitMoveIntent
) -> GitTransactionStatus | None:
    try:
        return await target.transactions.status_admitted(
            intent.transaction_id, intent.transaction_key
        )
    except GitTransactionError:
        return None  # the record is gone or unreadable; a rebuild replaces it


async def _drop_draft(target: GitProjectRuntime, transaction_id: UUID, key: str) -> None:
    """Roll a draft back if it still is one; it expires on its own otherwise."""
    try:
        current = await target.transactions.status_admitted(transaction_id, key)
        if current.draft_available:
            await target.transactions.rollback_admitted(
                transaction_id, key, GitTransactionControl(expected_revision=current.revision)
            )
    except GitTransactionError as error:
        logger.warning(
            "Cross-project move: draft rollback failed",
            transaction=str(transaction_id),
            error=str(error),
        )


async def _keep(
    intents: GitMoveIntentRepository,
    intent: GitMoveIntent,
    state: GitMoveOutcomeState,
    reason: str,
) -> GitCrossProjectMoveResult:
    """Keep the intent (and the reservation) for resume or a person's choice."""
    if state == "stuck":
        intent = intent.model_copy(update={"state": "stuck", "stuck_reason": reason})
        await asyncio.to_thread(intents.save, intent)
    logger.warning(
        "Cross-project move kept open",
        move=str(intent.move_id),
        note=str(intent.external_id),
        state=state,
        reason=reason,
    )
    return _result(intent, state, message=reason)


async def _forget(intents: GitMoveIntentRepository, intent: GitMoveIntent) -> None:
    await asyncio.to_thread(intents.remove, intent.move_id)


# --- Open intents: listing, startup reservations, resume, resolve ---


def _open_intents(
    host: GitHost,
) -> tuple[list[tuple[GitMoveIntent, GitProjectRuntime]], list[str]]:
    """Every readable intent with its target runtime (oldest first), and the unreadable."""
    found: list[tuple[GitMoveIntent, GitProjectRuntime]] = []
    unreadable: list[str] = []
    for runtime in host.runtimes.values():
        intents = _intents(runtime)
        try:
            found.extend((intent, runtime) for intent in intents.load_all())
        except (GitMoveIntentError, ValueError, OSError) as error:
            unreadable.append(f"{intents.path}: {error}")
    found.sort(key=lambda item: (item[0].created_at, str(item[0].move_id)))
    return found, unreadable


def list_cross_project_moves(host: GitHost) -> GitMoveListing:
    found, unreadable = _open_intents(host)
    return GitMoveListing(
        moves=[GitMoveIntentView.of(intent) for intent, _ in found], unreadable=unreadable
    )


def reserve_open_moves(host: GitHost) -> None:
    """Before any request: an open move's UUID may not be created anywhere else."""
    found, unreadable = _open_intents(host)
    identities = _identities(host)
    for intent, _ in found:
        identities.reservations[intent.external_id] = intent.move_id
    for problem in unreadable:
        logger.error("Cross-project move intents cannot be read", problem=problem)


async def resume_cross_project_moves(host: GitHost) -> list[GitMoveResumeOutcome]:
    """Finish or undo every open move; never raises (it runs at host start)."""
    async with host.move_lock:
        # Unreadable intent files are collected there, not raised (they stay reserved).
        found, _ = await asyncio.to_thread(_open_intents, host)
        outcomes = []
        for intent, target in found:
            try:
                outcomes.append(await _settled(_resume_one(host, intent, target)))
            except Exception as error:
                # One broken intent must not keep the others (or the host) from running.
                logger.exception(
                    "Cross-project move resume failed", move=str(intent.move_id), error=str(error)
                )
                outcomes.append(
                    GitMoveResumeOutcome(
                        move_id=intent.move_id,
                        external_id=intent.external_id,
                        outcome="failed",
                        detail=str(error),
                    )
                )
        return outcomes


async def _resume_one(
    host: GitHost, intent: GitMoveIntent, target: GitProjectRuntime
) -> GitMoveResumeOutcome:
    source = _runtime(host, intent.source_project_id)
    async with target.admission():
        result = await _resume_admitted(host, source, target, intent)
    return GitMoveResumeOutcome(
        move_id=intent.move_id,
        external_id=intent.external_id,
        outcome=result.state,
        detail=result.message or result.state,
        target_commit=result.target_commit,
    )


async def _resume_admitted(
    host: GitHost,
    source: GitProjectRuntime | None,
    target: GitProjectRuntime,
    intent: GitMoveIntent,
) -> _ResumeResult:
    intents = _intents(target)
    # (c) already happened: the draft published (its record or trailer says so), or the
    # target head holds the note.
    current = await _draft_status(target, intent)
    held = await asyncio.to_thread(target.pipeline.repository.published_note_ids)
    if (current is not None and current.state == "committed") or intent.external_id in held:
        moved = await _completed(
            host,
            target,
            intent,
            current.published_commit
            if current is not None and current.published_commit is not None
            else await asyncio.to_thread(target.pipeline.repository.head),
            None,
            True,
        )
        return _ResumeResult(
            "completed", moved.message or "The move had completed", moved.target_commit
        )
    if source is None:
        kept = await _keep(
            intents, intent, "stuck", "The source project is not on this host any more"
        )
        return _ResumeResult("stuck", kept.message or "", None)
    # Did (b) happen? Only the source's published history after the base can say.
    repository = source.pipeline.repository
    head = await asyncio.to_thread(repository.head)
    removals = await asyncio.to_thread(
        repository.removals_since, intent.source_base_commit, head, intent.external_id
    )
    in_source = intent.external_id in await asyncio.to_thread(repository.published_note_ids)
    if removals is None:
        kept = await _keep(
            intents,
            intent,
            "stuck",
            "The source's published history no longer contains the commit the move read",
        )
        return _ResumeResult("stuck", kept.message or "", None)
    if not removals and in_source:
        # (b) never happened: the note is still in the source, as it was.
        await _drop_draft(target, intent.transaction_id, intent.transaction_key)
        await _forget(intents, intent)
        _identities(host).reservations.pop(intent.external_id, None)
        return _ResumeResult(
            "abandoned", "The source never deleted the note; it stays in the source", None
        )
    if in_source:
        # Trigger: the source deleted the note, and its UUID came back there later (a
        #   peer, or a write that came before the reservation existed).
        # Why: committing would make a duplicate; dropping the intent would lose the
        #   moved bytes from every current state.
        # Outcome: stuck, with the bytes kept; a person decides (resolve).
        kept = await _keep(
            intents,
            intent,
            "stuck",
            f"Note {intent.external_id} is in the source project again after the move "
            "deleted it there",
        )
        return _ResumeResult("stuck", kept.message or "", None)
    if intent.state == "drafted" and removals:
        intent = intent.model_copy(update={"state": "source_deleted", "source_commit": removals[0]})
        await asyncio.to_thread(intents.save, intent)
    await _catch_up_quietly(source)
    moved = await _commit(host, source, target, intent, catch_up=True)
    return _ResumeResult(
        moved.state, moved.message or f"The move {moved.state}", moved.target_commit
    )


@dataclass(frozen=True, slots=True)
class _ResumeResult:
    state: GitMoveResumeKind
    message: str
    target_commit: str | None


async def resolve_cross_project_move(
    host: GitHost, move_id: UUID, choice: GitMoveResolve
) -> GitCrossProjectMoveResult:
    """A person's choice for an open move: publish in the target, or return to the source."""
    async with host.move_lock:
        found, _ = await asyncio.to_thread(_open_intents, host)
        match = next(((i, t) for i, t in found if i.move_id == move_id), None)
        if match is None:
            raise GitMoveRefused(404, "not_found", f"No open cross-project move {move_id}.")
        intent, target = match
        source = _runtime(host, intent.source_project_id)
        return await _settled(_resolve(host, intent, target, source, choice))


async def _resolve(
    host: GitHost,
    intent: GitMoveIntent,
    target: GitProjectRuntime,
    source: GitProjectRuntime | None,
    choice: GitMoveResolve,
) -> GitCrossProjectMoveResult:
    async with target.admission():
        if choice.choice == "target":
            path = choice.destination_path or intent.target_path
            current = await _draft_status(target, intent)
            if current is not None and current.state == "committed":
                return await _commit(host, source, target, intent, catch_up=True)
            if source is not None:
                in_source = intent.external_id in await asyncio.to_thread(
                    source.pipeline.repository.published_note_ids
                )
                if in_source:
                    raise GitMoveRefused(
                        409,
                        "in_source",
                        f"Note {intent.external_id} is in the source project; resolve with "
                        "choice 'source'.",
                    )
            try:
                intent, _ = await _rebuild(host, target, intent, path=path)
            except _Stuck as stuck:
                raise GitMoveRefused(409, "occupied", str(stuck)) from stuck
            intent = intent.model_copy(update={"state": "source_deleted", "stuck_reason": None})
            await asyncio.to_thread(_intents(target).save, intent)
            return await _commit(host, source, target, intent, catch_up=True)
        if source is None:
            raise GitMoveRefused(
                409, "no_source", "The source project is not on this host; choose 'target'."
            )
        return await _return_to_source(host, intent, target, source, choice.destination_path)


async def _return_to_source(
    host: GitHost,
    intent: GitMoveIntent,
    target: GitProjectRuntime,
    source: GitProjectRuntime,
    destination: str | None,
) -> GitCrossProjectMoveResult:
    """Put the note back where it came from, with its UUID and its old bytes."""
    repository = source.pipeline.repository
    held = await asyncio.to_thread(repository.published_note_ids)
    current = await _draft_status(target, intent)
    if current is not None and current.state == "committed":
        raise GitMoveRefused(
            409, "completed", "The move already published in the target; nothing to return."
        )
    await _drop_draft(target, intent.transaction_id, intent.transaction_key)
    if intent.external_id in held:
        # The note is in the source already (it never left, or came back): close the move.
        await _forget(_intents(target), intent)
        _identities(host).reservations.pop(intent.external_id, None)
        return _result(
            intent,
            "returned",
            message=(
                f"Note {intent.external_id} is in the source project; the moved bytes stay "
                f"in its history at commit {intent.source_base_commit}"
            ),
        )
    path = destination or intent.source_path
    original = await asyncio.to_thread(repository.snapshot, intent.source_base_commit)
    content = original.files[intent.source_path]
    async with source.admission():
        transactions = source.transactions
        started = await transactions.start_admitted(GitTransactionStart())
        try:
            await transactions.write_identified_admitted(
                started.transaction_id,
                started.transaction_key,
                path=path,
                content=content,
                note_id=intent.external_id,
                expected_revision=0,
                created=intent.source_created,
                modified=intent.source_modified,
                reservation=intent.move_id,
            )
            status = await transactions.commit_admitted(
                started.transaction_id,
                started.transaction_key,
                GitTransactionCommit(
                    expected_revision=1,
                    message=f"Return {quote_in_message(path)} from an interrupted move",
                ),
                reservation=intent.move_id,
            )
        except GitTransactionError as error:
            await _drop_draft(source, started.transaction_id, started.transaction_key)
            raise GitMoveRefused(
                error.status_code, "occupied", f"The source cannot take the note back: {error}"
            ) from error
        await source.catch_up_admitted()
    await _forget(_intents(target), intent)
    _identities(host).reservations.pop(intent.external_id, None)
    return _result(
        intent.model_copy(update={"target_path": path}),
        "returned",
        target_commit=status.published_commit,
        message=f"The note is back in the source project at '{path}'",
    )
