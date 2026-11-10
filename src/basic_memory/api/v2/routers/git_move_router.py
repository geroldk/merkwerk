"""Cross-project moves that keep the note's UUID, on a local Git host (I1).

POST /v2/projects/{source}/git/moves moves one note; the open moves of the whole host
(pending or stuck after a crash) are listed, resumed and resolved under /v2/git/moves.
A host without Git publication has none of this: there a move across projects copies
the note with a new id (the MCP tool does that itself), and these routes answer 404.
"""

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from basic_memory.deps.git import GitHostDep, GitRuntimeDep
from basic_memory.deps.services import NoteContentMutationServiceDep
from basic_memory.index.git_host import GitHost
from basic_memory.repository.git_projection_repository import GitProjectionConflict
from basic_memory.schemas.git_moves import (
    GitCrossProjectMoveRequest,
    GitCrossProjectMoveResult,
    GitMoveCatchUp,
    GitMoveListing,
    GitMoveResolve,
    GitMoveResumeResult,
)
from basic_memory.services.git_cross_project_move import (
    GitMoveRefused,
    list_cross_project_moves,
    move_note_across_projects,
    resolve_cross_project_move,
    resume_cross_project_moves,
)

NOT_A_GIT_HOST = {
    "message": "Moves that keep the note id need a local Git host; this host has none.",
    "kind": "not_git_host",
}

# The source project's routes (prefix /v2/projects/{project_id}).
router = APIRouter(prefix="/git/moves", tags=["git-moves"])
# The host's open moves (prefix /v2).
host_router = APIRouter(prefix="/git/moves", tags=["git-moves"])


def _require_host(host: GitHost | None) -> GitHost:
    if host is None:
        raise HTTPException(404, NOT_A_GIT_HOST)
    return host


@router.post("", response_model=GitCrossProjectMoveResult)
async def move_git_note_across_projects(
    request: GitCrossProjectMoveRequest,
    host: GitHostDep,
    source: GitRuntimeDep,
    mutations: NoteContentMutationServiceDep,
    response: Response,
) -> GitCrossProjectMoveResult:
    """Move one note into another project; it keeps its UUID (external_id).

    200 with state "completed" (index_pending when the target's SQL lags), or "pending" /
    "stuck" when the move keeps its intent: the note is then in no project's current state
    until resume or a person's choice finishes it. A refusal (4xx) moved nothing; its
    detail carries `kind`.
    """
    host = _require_host(host)
    if source is None:  # pragma: no cover - a Git host always resolves the runtime
        raise HTTPException(404, NOT_A_GIT_HOST)
    response.headers["Cache-Control"] = "no-store"
    try:
        return await move_note_across_projects(host, source, request, mutations)
    except GitMoveRefused as refused:
        raise HTTPException(refused.status_code, refused.body()) from refused


@router.post("/catch-up", response_model=GitMoveCatchUp)
async def catch_up_git_project(source: GitRuntimeDep, host: GitHostDep) -> GitMoveCatchUp:
    """Bring this project's SQL to its published head now (after a folder moved in)."""
    _require_host(host)
    if source is None:  # pragma: no cover - a Git host always resolves the runtime
        raise HTTPException(404, NOT_A_GIT_HOST)
    try:
        deferral = await source.catch_up()
    except GitProjectionConflict as error:
        raise HTTPException(409, {"message": str(error), "kind": "not_accepting"}) from error
    head = await asyncio.to_thread(source.pipeline.repository.head)
    return GitMoveCatchUp(
        project_id=UUID(source.pipeline.project_external_id),
        commit_id=head,
        index_pending=deferral is not None,
        message=str(deferral) if deferral is not None else None,
    )


@host_router.get("", response_model=GitMoveListing)
async def list_git_moves(host: GitHostDep, response: Response) -> GitMoveListing:
    """Open moves (pending or stuck), without note bytes or transaction keys."""
    response.headers["Cache-Control"] = "no-store"
    # Reads intent files of every project: off the event loop.
    return await asyncio.to_thread(list_cross_project_moves, _require_host(host))


@host_router.post("/resume", response_model=GitMoveResumeResult)
async def resume_git_moves(host: GitHostDep, response: Response) -> GitMoveResumeResult:
    """Finish or undo every open move now, as the host does at start."""
    response.headers["Cache-Control"] = "no-store"
    return GitMoveResumeResult(outcomes=await resume_cross_project_moves(_require_host(host)))


@host_router.post("/{move_id}/resolve", response_model=GitCrossProjectMoveResult)
async def resolve_git_move(
    move_id: UUID, choice: GitMoveResolve, host: GitHostDep, response: Response
) -> GitCrossProjectMoveResult:
    """A person's choice for a stuck move: publish in the target, or return to the source."""
    response.headers["Cache-Control"] = "no-store"
    try:
        return await resolve_cross_project_move(_require_host(host), move_id, choice)
    except GitMoveRefused as refused:
        raise HTTPException(refused.status_code, refused.body()) from refused
