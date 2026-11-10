"""Moving notes into another project (#1842).

On a Git host the note keeps its id (I1): the server drafts it in the target under its
old UUID, deletes it in the source, then publishes the draft (POST
/v2/projects/{source}/git/moves). An interrupted move keeps an intent and finishes at the
next host start; until then the call answers PARTIAL_FAILURE with the move's id and state.

On any other host a move across projects is a write in the target project followed by a
delete in the source project. The moved note is a new note there: it gets a new id and its
own version history. On both, links in either project are not rewritten.

Its permalink is formed again from its new path, as for a note created there: a permalink
that still carried the source project's prefix would point into the wrong project, and a
gateway refuses addresses that start with another project's name. The result names the
old and the new permalink and the identity of the new note. Every other frontmatter key keeps
its value; the copy is written like any new note (frontmatter re-serialised, body trimmed).

The source is deleted only after the copy is written, and only while it is still
the revision that was copied. If that delete is refused (the note changed, or the
caller cannot delete in the source project), the copy is removed again, so a
refused move leaves both projects as they were.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, assert_never
from uuid import UUID

from fastmcp.exceptions import ToolError
from httpx import HTTPStatusError

from basic_memory.file_utils import drop_frontmatter_key
from basic_memory.markdown.note_lock import LOCKED_NOTE_MESSAGE, note_is_locked
from basic_memory.mcp.clients import DirectoryClient, KnowledgeClient
from basic_memory.mcp.clients.git_moves import GitMoveClient
from basic_memory.mcp.note_targets import ProjectTarget, resolve_note
from basic_memory.mcp.refusals import (
    Refusal,
    RefusalCode,
    api_refusal,
    note_change_failure,
    refusal,
    revision_conflict,
)
from basic_memory.mcp.tool_params import OutputFormat
from basic_memory.runtime.git_routing import git_local_routing_active
from basic_memory.runtime.storage import runtime_file_path_is_markdown_note
from basic_memory.schemas.base import Entity
from basic_memory.schemas.directory import DirectoryNode
from basic_memory.schemas.git_moves import GitCrossProjectMoveRequest
from basic_memory.schemas.memory import memory_url_path
from basic_memory.schemas.v2.entity import EntityResponseV2
from basic_memory.schemas.v2.note_write import (
    NoteAlreadyExists,
    NoteCreated,
    NoteLocked,
    NoteRevisionConflict,
    NoteDuplicateTitle,
    NoteTargetMoved,
    NoteUpdated,
)

UNCHANGED = "Nothing was moved."
LISTING_PAGE_SIZE = 200


@dataclass(frozen=True, slots=True)
class MovedNote:
    """One note that now lives in the target project."""

    previous_file_path: str
    file_path: str
    previous_permalink: str | None
    permalink: str | None
    previous_external_id: str
    external_id: str


# --- One note ---


async def move_note_across(
    source: ProjectTarget,
    target: ProjectTarget,
    identifier: str,
    destination: str,
    expected_checksum: str | None,
    output_format: OutputFormat,
) -> str | dict[str, Any]:
    """Move one note into the target project; refuse with both projects unchanged."""
    note = await resolve_note(
        source,
        identifier,
        output_format=output_format,
        expected_checksum=expected_checksum,
        unchanged=UNCHANGED,
    )
    # Trigger: the destination ends with '/', or is the project root.
    # Outcome: the note keeps its file name under that folder, as within a project.
    if destination.endswith("/") or not destination:
        destination_path = str(
            PurePosixPath(destination.rstrip("/")) / PurePosixPath(note.file_path).name
        )
    else:
        destination_path = destination
    moved = await _move_one(
        source,
        target,
        note.external_id,
        note.file_path,
        destination_path,
        expected_checksum,
        output_format,
    )
    result = {
        "action": "moved",
        "previous_project": source.project.name,
        "previous_file_path": source.paths.path(moved.previous_file_path),
        "project": target.project.name,
        "file_path": target.paths.path(moved.file_path),
        "previous_permalink": moved.previous_permalink,
        "permalink": moved.permalink,
        "previous_external_id": moved.previous_external_id,
        "external_id": moved.external_id,
    }
    if output_format == "json":
        return result
    text = (
        f"Moved {result['previous_file_path']} (project {result['previous_project']}) to "
        f"{result['file_path']} (project {result['project']})"
    )
    if moved.external_id == moved.previous_external_id:
        text += f", id kept ({moved.external_id})"
    if moved.previous_permalink and moved.permalink:
        text += f"; permalink {moved.previous_permalink} -> {moved.permalink}"
    return text


async def _move_one(
    source: ProjectTarget,
    target: ProjectTarget,
    external_id: str,
    file_path: str,
    destination_path: str,
    expected_checksum: str | None,
    output_format: OutputFormat,
    *,
    catch_up: bool = True,
) -> MovedNote:
    """Copy one note into the target project, then delete it from the source.

    On a Git host the server moves it instead and the note keeps its id; `catch_up`
    False leaves the target's index to one catch-up after a whole folder.
    """
    address = source.paths.path(file_path)
    if not runtime_file_path_is_markdown_note(file_path):
        raise refusal(
            "UNSUPPORTED",
            f"'{address}' is not a Markdown note; only notes move between projects.",
            output_format=output_format,
        )
    if not runtime_file_path_is_markdown_note(destination_path):
        raise refusal(
            "INVALID_ARGUMENTS",
            f"'{target.paths.path(destination_path)}' is not a Markdown note path.",
            try_instead="a destination ending in '.md', or a folder ending in '/'",
            output_format=output_format,
        )
    # Trigger: this host publishes its projects through Git.
    # Why: the server can move the note under its own id (draft, delete, commit) and
    #   finish an interrupted move; a copy here would give it a new id.
    # Outcome: one call to the source project's move route; W0 below stays for other hosts.
    if git_local_routing_active():
        return await _move_keeping_id(
            source,
            target,
            external_id,
            file_path,
            destination_path,
            expected_checksum,
            output_format,
            catch_up=catch_up,
        )

    # --- Read the revision being moved ---
    # The checksum is taken before the content, so a change landing in between
    # makes the guarded delete below refuse instead of dropping that change.
    source_client = KnowledgeClient(source.client, source.project.external_id)
    entity = await source_client.get_entity(external_id)
    if expected_checksum is not None and entity.db_checksum != expected_checksum:
        raise revision_conflict(
            address,
            entity.db_checksum,
            unchanged=UNCHANGED,
            output_format=output_format,
            project=source.project_ref,
        )
    revision = entity.db_checksum
    content = (await source_client.accepted_note_contents([external_id]))[external_id]
    # Trigger: the note carries `locked: true`.
    # Why: the source refuses the delete, and the copy carries the lock too, so it
    #      could not be removed again; the note would end up in both projects.
    # Outcome: refused before anything is written, as a move within a project is.
    if note_is_locked(content):
        raise refusal("LOCKED", LOCKED_NOTE_MESSAGE, output_format=output_format)

    # --- Write the copy ---
    # Trigger: the note carries a permalink formed in the source project.
    # Why: a permalink in the frontmatter is kept as written, so it would keep the source
    #      project's prefix; without it the target forms one from the new path.
    # Outcome: the copy has the target's own permalink; every other key keeps its value. (The
    #          note is written again as for any new note: frontmatter re-serialised, body trimmed.)
    content = drop_frontmatter_key(content, "permalink")
    copy = await _write_copy(target, destination_path, entity.title, content, output_format)

    # --- Delete the source ---
    try:
        await source_client.delete_entity(external_id, base_checksum=revision)
    except ToolError as error:
        copy_address = target.paths.path(copy.file_path)
        # Trigger: no answer said whether the source delete landed.
        # Why: the source may already be gone; removing the copy could lose the note.
        # Outcome: the copy stays, and the caller is told to check the source.
        if api_refusal(error) is None:
            raise refusal(
                "OUTCOME_UNKNOWN",
                f"'{address}' was copied to '{copy_address}' in project "
                f"'{target.project.name}', and its delete from project "
                f"'{source.project.name}' may or may not have happened: {error}",
                try_instead=(
                    f"bm_read(identifier={address!r}, project={source.project_ref!r}), and "
                    "bm_delete it if it is still there"
                ),
                output_format=output_format,
            ) from error
        # Trigger: the source refused the delete (the note changed since it was
        #   read, or the caller may not delete in the source project).
        # Why: the move must not leave the note in both projects.
        # Outcome: the copy is removed while it is still unchanged, and the move refuses.
        await _remove_copy(target, copy, copy_address, output_format)
        raise note_change_failure(
            error,
            action="move",
            address=address,
            unchanged=UNCHANGED,
            output_format=output_format,
            project=source.project.external_id,
        ) from error
    return MovedNote(
        previous_file_path=file_path,
        file_path=copy.file_path,
        previous_permalink=entity.permalink,
        permalink=copy.permalink,
        previous_external_id=external_id,
        external_id=copy.external_id,
    )


async def _write_copy(
    target: ProjectTarget,
    destination_path: str,
    title: str,
    content: str,
    output_format: OutputFormat,
) -> EntityResponseV2:
    """Create the note in the target project; never replace a note already there.

    The note keeps its title; `file_name` alone sets where it lands, as a move
    within a project does.
    """
    path = PurePosixPath(destination_path)
    directory = "" if str(path.parent) == "." else str(path.parent)
    note = Entity(
        title=title,
        directory=directory,
        file_name=path.name,
        content_type="text/markdown",
        content=content,
    )
    shown = target.paths.path(destination_path)
    try:
        outcome = await KnowledgeClient(target.client, target.project.external_id).write_note(
            note, overwrite=False
        )
    except ToolError as error:
        raise note_change_failure(
            error,
            action="move",
            address=shown,
            unchanged=UNCHANGED,
            output_format=output_format,
            project=target.project.external_id,
        ) from error
    match outcome:
        case NoteCreated(entity=written):
            return written
        case NoteDuplicateTitle(title=twin_title, file_path=twin_path):
            raise refusal(
                "ALREADY_EXISTS",
                f"A note titled '{twin_title}' already exists at "
                f"'{target.paths.path(twin_path)}' in project '{target.project.name}'. {UNCHANGED}",
                try_instead="renaming one of the notes first",
                output_format=output_format,
            )
        case NoteAlreadyExists() | NoteTargetMoved():
            raise refusal(
                "ALREADY_EXISTS",
                f"A note already exists at '{shown}' in project '{target.project.name}'. "
                f"{UNCHANGED}",
                try_instead="a different destination",
                output_format=output_format,
            )
        case NoteUpdated() | NoteRevisionConflict() | NoteLocked():  # pragma: no cover
            # overwrite=False without expected_checksum creates a note or finds the
            # path taken; there is no predecessor to update, conflict with or lock.
            raise AssertionError(f"unexpected write outcome for a new note: {outcome!r}")
        case _:  # pragma: no cover
            assert_never(outcome)


async def _remove_copy(
    target: ProjectTarget, copy: EntityResponseV2, copy_address: str, output_format: OutputFormat
) -> None:
    """Delete the copy just written, unless someone has changed it since."""
    try:
        await KnowledgeClient(target.client, target.project.external_id).delete_entity(
            copy.external_id, base_checksum=copy.db_checksum
        )
    except ToolError as error:
        raise refusal(
            "PARTIAL_FAILURE",
            f"The source note could not be deleted, and its copy at '{copy_address}' in "
            f"project '{target.project.name}' could not be removed either: {error}. The note "
            "is now in both projects.",
            try_instead=f"bm_delete(identifier={copy_address!r}, project={target.project_ref!r})",
            output_format=output_format,
        ) from error


# --- On a Git host: the note keeps its id (I1) ---

# A refusal's `detail.kind` decides the code; nothing was moved for any of them.
_KEPT_ID_CODES: dict[str, RefusalCode] = {
    "not_found": "NOT_FOUND",
    "locked": "LOCKED",
    "unsupported": "UNSUPPORTED",
    "invalid_path": "INVALID_ARGUMENTS",
    "invalid_draft": "INVALID_ARGUMENTS",
    "same_project": "INVALID_ARGUMENTS",
    "occupied": "ALREADY_EXISTS",
    "already_in_target": "ALREADY_EXISTS",
    "deferred": "UNAVAILABLE",
    "not_accepting": "UNAVAILABLE",
    "reserved": "UNAVAILABLE",
    "source_failed": "UNAVAILABLE",
}


async def _move_keeping_id(
    source: ProjectTarget,
    target: ProjectTarget,
    external_id: str,
    file_path: str,
    destination_path: str,
    expected_checksum: str | None,
    output_format: OutputFormat,
    *,
    catch_up: bool,
) -> MovedNote:
    """Ask the Git host to move the note; refuse, or report a move that is still open."""
    address = source.paths.path(file_path)
    try:
        request = GitCrossProjectMoveRequest(
            external_id=UUID(external_id),
            target_project_id=UUID(target.project.external_id),
            destination_path=destination_path,
            expected_checksum=expected_checksum,
            catch_up=catch_up,
        )
    except ValueError as error:
        raise refusal(
            "INVALID_ARGUMENTS",
            f"'{target.paths.path(destination_path)}' is not a note path there: {error}",
            output_format=output_format,
        ) from error
    try:
        moved = await GitMoveClient(source.client, source.project.external_id).move(request)
    except ToolError as error:
        raise _kept_id_failure(
            error, source, target, address, destination_path, output_format
        ) from error
    shown = target.paths.path(moved.file_path)
    if moved.state != "completed":
        # Trigger: the source let go, but the target has not published the note (yet).
        # Why: the note is in neither project's current state; its bytes are kept in the
        #   move's intent and in the source's history, never lost.
        # Outcome: PARTIAL_FAILURE naming the move, so the caller can follow it up.
        follow_up = (
            "it finishes at the next host start or with POST /v2/git/moves/resume"
            if moved.state == "pending"
            else (
                "a person must choose where it goes: POST "
                f"/v2/git/moves/{moved.move_id}/resolve with choice 'target' (optionally a "
                "free destination_path) or 'source'"
            )
        )
        raise refusal(
            "PARTIAL_FAILURE",
            f"'{address}' left project '{source.project.name}' but is not in project "
            f"'{target.project.name}' yet (move {moved.move_id}, {moved.state}: "
            f"{moved.message}); {follow_up}. Its content is kept and its id "
            f"{moved.external_id} stays reserved.",
            try_instead="GET /v2/git/moves to see the open move",
            output_format=output_format,
            details={
                "move_id": str(moved.move_id),
                "state": moved.state,
                "external_id": str(moved.external_id),
                "target_path": shown,
                "reason": moved.message,
            },
        )
    return MovedNote(
        previous_file_path=moved.previous_file_path,
        file_path=moved.file_path,
        previous_permalink=moved.previous_permalink,
        permalink=moved.permalink,
        previous_external_id=external_id,
        external_id=str(moved.external_id),
    )


def _kept_id_failure(
    error: ToolError,
    source: ProjectTarget,
    target: ProjectTarget,
    address: str,
    destination_path: str,
    output_format: OutputFormat,
) -> ToolError:
    """Word a refused move by its `detail.kind`; every one of them moved nothing."""
    cause = error.__cause__
    answer = cause.response if isinstance(cause, HTTPStatusError) else None
    detail = (
        answer.json().get("detail")
        if answer is not None and "application/json" in answer.headers.get("content-type", "")
        else None
    )
    shown = target.paths.path(destination_path)
    kind = detail.get("kind") if isinstance(detail, dict) else None
    if kind == "revision_conflict" and isinstance(detail, dict):
        return revision_conflict(
            address,
            detail.get("db_checksum"),
            unchanged=UNCHANGED,
            output_format=output_format,
            project=source.project_ref,
        )
    if kind == "duplicate_title" and isinstance(detail, dict):
        return refusal(
            "ALREADY_EXISTS",
            f"A note titled '{detail.get('title')}' already exists at "
            f"'{target.paths.path(str(detail.get('file_path')))}' in project "
            f"'{target.project.name}'. {UNCHANGED}",
            try_instead="renaming one of the notes first",
            output_format=output_format,
        )
    if isinstance(detail, dict) and kind in _KEPT_ID_CODES:
        code = _KEPT_ID_CODES[kind]
        try_instead = {
            "ALREADY_EXISTS": "a different destination",
            "UNAVAILABLE": (
                f"bm_project(command='status', project={source.project_ref!r}) and "
                f"bm_project(command='status', project={target.project_ref!r}), then repeat"
            ),
        }.get(code, f"bm_read(identifier={address!r}, project={source.project_ref!r})")
        return refusal(
            code,
            str(detail.get("message") or error),
            try_instead=try_instead,
            output_format=output_format,
        )
    return note_change_failure(
        error,
        action="move",
        address=address,
        unchanged=UNCHANGED,
        occupied=shown,
        output_format=output_format,
        project=source.project.external_id,
    )


# --- A directory ---


async def move_directory_across(
    source: ProjectTarget,
    target: ProjectTarget,
    destination: str,
    output_format: OutputFormat,
) -> str | dict[str, Any]:
    """Move every note under a directory into the target project, one note at a time."""
    directory = memory_url_path(source.route.path).strip("/")
    if not directory:
        raise refusal(
            "INVALID_ARGUMENTS",
            "The project root cannot be moved.",
            output_format=output_format,
        )
    # A trailing '/' moves the directory into that folder, as it does within a project.
    if destination.endswith("/"):
        destination = f"{destination}{PurePosixPath(directory).name}"
    destination = destination.strip("/")

    files = await _files_under(source, directory)
    if not files:
        raise refusal(
            "NOT_FOUND",
            f"No files under '{source.paths.path(directory)}' in project '{source.project.name}'.",
            output_format=output_format,
        )

    moved: list[MovedNote] = []
    failed: list[dict[str, str]] = []
    # Moves the server keeps open (Git host only): the note left the source, not yet here.
    pending: list[dict[str, Any]] = []
    on_git = git_local_routing_active()
    for node in files:
        assert node.file_path is not None and node.external_id is not None
        relative = PurePosixPath(node.file_path).relative_to(directory)
        destination_path = (
            str(PurePosixPath(destination) / relative) if destination else str(relative)
        )
        try:
            moved.append(
                await _move_one(
                    source,
                    target,
                    node.external_id,
                    node.file_path,
                    destination_path,
                    None,
                    output_format,
                    # One catch-up of the target after the folder, not one per note.
                    catch_up=not on_git,
                )
            )
        except Refusal as error:
            failed.append({"path": source.paths.path(node.file_path), "error": str(error)})
            if error.code == "PARTIAL_FAILURE" and error.details and "move_id" in error.details:
                failed[-1]["error"] = f"pending (move_id {error.details['move_id']})"
                pending.append({"path": source.paths.path(node.file_path), **error.details})
        except ToolError as error:
            failed.append({"path": source.paths.path(node.file_path), "error": str(error)})
    index_note = ""
    if on_git and moved:
        # Trigger: notes moved in with catch_up off.
        # Why: reads by id or path in the target must work when the answer arrives.
        # Outcome: one catch-up; a failed one leaves only the index behind (the worker
        #   follows), never the move.
        try:
            caught = await GitMoveClient(target.client, target.project.external_id).catch_up()
            if caught.index_pending:
                index_note = f"; the target's index is still catching up ({caught.message})"
        except ToolError as error:
            index_note = f"; the target's index is still catching up ({error})"

    result: dict[str, Any] = {
        "action": "moved",
        "previous_project": source.project.name,
        "path": source.paths.path(directory),
        "project": target.project.name,
        "destination": target.paths.path(destination),
        "total_files": len(files),
        "moved_files": [target.paths.path(note.file_path) for note in moved],
        "moved_notes": [
            {
                "previous_file_path": source.paths.path(note.previous_file_path),
                "file_path": target.paths.path(note.file_path),
                "previous_permalink": note.previous_permalink,
                "permalink": note.permalink,
                "previous_external_id": note.previous_external_id,
                "external_id": note.external_id,
            }
            for note in moved
        ],
        "failed": failed,
    }
    if on_git:
        result["pending"] = pending
    # Trigger: some notes moved and some did not (a file that is not a note, a name
    #   already taken in the target project, a note that changed meanwhile).
    # Why: each note moved or stayed whole, but the directory did neither.
    # Outcome: an error listing what stayed, so the caller can decide what to retry.
    if failed:
        # What did move is part of the answer: the caller has to log it and cannot ask again,
        # because each moved note is already gone from the source. The JSON form carries it as
        # data (details.moved_notes); the text form lists it in the message.
        done = "; ".join(
            f"{note['previous_file_path']} -> {note['file_path']} "
            f"(permalink {note['previous_permalink']} -> {note['permalink']}, "
            + (
                f"id {note['external_id']} kept)"
                if note["previous_external_id"] == note["external_id"]
                else f"id {note['previous_external_id']} -> {note['external_id']})"
            )
            for note in result["moved_notes"]
        )
        listed = f". Moved: {done}" if done and output_format != "json" else ""
        raise refusal(
            "PARTIAL_FAILURE",
            f"Moved {len(moved)} of {len(files)} files from '{result['path']}' "
            f"(project {source.project.name}) to '{result['destination']}' "
            f"(project {target.project.name}); {len(failed)} stayed: "
            + "; ".join(f"{failure['path']}: {failure['error']}" for failure in failed)
            + listed,
            try_instead=(
                f"bm_list(path={result['path']!r}, project={source.project_ref!r}) "
                "to see what stayed"
            ),
            output_format=output_format,
            details={
                "moved_notes": result["moved_notes"],
                "failed": failed,
                "total_files": len(files),
                **({"pending": pending} if on_git else {}),
            },
        )
    if output_format == "json":
        return result
    summary = (
        f"Moved directory {result['path']} (project {source.project.name}) to "
        f"{result['destination']} (project {target.project.name}) ({len(files)} files)"
    )
    if not on_git:
        return summary
    # Every note, one line each and no upper limit, as a folder move inside a project.
    return "\n".join(
        [summary + ", ids kept" + index_note]
        + [
            f"- {note['previous_file_path']} -> {note['file_path']} "
            f"(id {note['external_id']}, permalink {note['permalink']})"
            for note in result["moved_notes"]
        ]
    )


async def _files_under(
    source: ProjectTarget, directory: str, page_size: int = LISTING_PAGE_SIZE
) -> list[DirectoryNode]:
    """Every file under `directory`, at any depth, in listing order."""
    client = DirectoryClient(source.client, source.project.external_id)
    files: list[DirectoryNode] = []
    pending = [f"/{directory}"]
    while pending:
        folder = pending.pop(0)
        page = 1
        while True:
            listing = await client.list(folder, depth=1, page=page, page_size=page_size)
            for node in listing.nodes:
                if node.type == "directory":
                    pending.append(node.directory_path)
                else:
                    files.append(node)
            if not listing.has_more:
                break
            page += 1
    return files
