"""Observe Git publication and eventual local projection without starting work."""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from basic_memory.deps.git import GitHostDep
from basic_memory.schemas.git_status import GitProjectStatus, GitStageStatus, GitWaitingNote
from basic_memory.schemas.git_sync import GitSyncStatus
from basic_memory.services.git_merge_resolution import ReviewIncomplete
from basic_memory.services.git_project_runtime import GitProjectRuntime, owner_content_errors
from basic_memory.services.git_sync import GitSyncBusy

router = APIRouter(prefix="/git", tags=["git"])


@router.get("/status", response_model=GitProjectStatus)
async def get_git_project_status(project_id: str, host: GitHostDep) -> GitProjectStatus:
    if host is None or project_id not in host.runtimes:
        raise HTTPException(404, "Project is not owned by this Git host")
    runtime = host.runtime_for(project_id)
    progress = await runtime.worker.status()
    waiting = await waiting_notes(runtime)
    return GitProjectStatus(
        project_id=project_id,
        source_commit=progress.source_commit,
        caught_up=progress.caught_up,
        stages=[
            GitStageStatus(
                stage=row.stage,
                target_commit=row.target_commit,
                completed_commit=row.completed_commit,
                running_commit=row.running_commit,
                attempts=row.attempts,
                last_error=row.last_error,
                # The wait belongs to the published head it was found at; a newer head
                # gets a new pass, whose outcome replaces it.
                deferred=(
                    waiting
                    if row.stage == "content"
                    and runtime.worker.deferral_commit == progress.source_commit
                    else None
                ),
            )
            for row in progress.stages
        ],
        synchronization=await runtime.sync.status(),
    )


async def waiting_notes(runtime: GitProjectRuntime) -> list[GitWaitingNote] | None:
    """The notes the worker's last content pass left waiting, with each owner's error."""
    deferral = runtime.worker.deferral
    if deferral is None:
        return None
    errors = await owner_content_errors(deferral, runtime.pipeline.session_maker)
    return [
        GitWaitingNote(
            external_id=note.external_id,
            owner_project_id=str(note.owner_external_id),
            owner_project=note.owner_name,
            owner_last_error=errors.get(note.owner_project_id),
        )
        for note in deferral.waiting
    ]


@router.post("/sync", response_model=GitSyncStatus)
async def sync_git_project(project_id: str, host: GitHostDep, response: Response) -> GitSyncStatus:
    if host is None or project_id not in host.runtimes:
        raise HTTPException(404, "Project is not owned by this Git host")
    service = host.runtime_for(project_id).sync
    if service.remote is None:
        raise HTTPException(409, "No Git peer configured for this project")
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.synchronize()
    except GitSyncBusy as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(
            503, "Sync status unavailable; inspect published Git before retry"
        ) from error


class GitMergeApply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review: str = Field(min_length=1, max_length=64)


@router.post("/merge", response_model=GitSyncStatus)
async def merge_git_project(
    project_id: str, request: GitMergeApply, host: GitHostDep, response: Response
) -> GitSyncStatus:
    """Apply a person's decisions from a review package; never reachable via the gateway."""
    if host is None or project_id not in host.runtimes:
        raise HTTPException(404, "Project is not owned by this Git host")
    service = host.runtime_for(project_id).sync
    if service.remote is None:
        raise HTTPException(409, "No Git peer configured for this project")
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.synchronize(review=request.review)
    except GitSyncBusy as error:
        raise HTTPException(409, str(error)) from error
    except ReviewIncomplete as error:
        raise HTTPException(422, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except OSError as error:
        raise HTTPException(
            503, "Sync status unavailable; inspect published Git before retry"
        ) from error
