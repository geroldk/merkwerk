"""Deny unconnected write paths before dependencies can accept any mutation."""

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute

from basic_memory.deps.config import AppConfigDep

GIT_NOTE_WRITES = frozenset(
    {
        "write_note",
        "create_entity",
        "update_entity_by_id",
        "edit_entity_by_id",
        "delete_entity_by_id",
        "move_entity",
        # A folder move inside one project, published as one commit (V1). The upstream
        # move_directory stays refused: it moves note by note and bytes outside Git.
        "move_git_directory",
    }
)
GIT_TRANSACTION_WRITES = frozenset(
    {
        "start_transaction",
        "write_transaction_note",
        "move_transaction_note",
        "delete_transaction_note",
        "keep_alive_transaction",
        "commit_transaction",
        "rollback_transaction",
    }
)


async def enforce_git_api_policy(request: Request, app_config: AppConfigDep) -> None:
    if not app_config.git_repositories:
        return
    route = request.scope["route"]
    if not isinstance(route, APIRoute):
        raise HTTPException(403, "Endpoint is not available on the Git host")
    module = route.endpoint.__module__
    name = route.endpoint.__name__
    if module.endswith(".schema_router") or module.endswith(".prompt_router"):
        # These older helpers can read raw editor files. Publish/import them
        # through a reviewed Git path before enabling them for Git projects.
        raise HTTPException(403, "This file-based helper is not connected to Git publication")
    if request.method in {"GET", "HEAD", "OPTIONS", "QUERY"}:
        return
    if module.endswith(".project_router") and name == "resolve_project_identifier":
        return
    if module.endswith(".git_transaction_router") and name in GIT_TRANSACTION_WRITES:
        return
    if module.endswith(".git_router") and name in {"sync_git_project", "merge_git_project"}:
        return
    if module.endswith((".knowledge_router", ".note_write_router")) and name in GIT_NOTE_WRITES:
        return
    raise HTTPException(403, "This mutation is not connected to Git publication")
