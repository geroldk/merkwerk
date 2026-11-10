"""FastAPI application for basic-memory knowledge graph API."""

import importlib
from contextlib import AsyncExitStack, asynccontextmanager

import logfire
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from loguru import logger

from basic_memory import __version__ as version
from basic_memory.api.container import ApiContainer, installed_container
from basic_memory.api.git_policy import enforce_git_api_policy
from basic_memory.index.git_host import run_git_host
from basic_memory.api.v2.routers.git_move_router import (
    host_router as git_host_move_router,
    router as git_move_router,
)
from basic_memory.api.v2.routers.git_router import router as git_router
from basic_memory.api.v2.routers.git_transaction_router import router as git_transaction_router
from basic_memory.repository.git_transaction_repository import GitTransactionError
from redis_fastapi.cache import add_redis_caching

from basic_memory.read_cache.policy import READ_CACHE_MAX_PAYLOAD_BYTES
from basic_memory.api.v2.routers import (
    accepted_content_router as v2_accepted_content,
    knowledge_router as v2_knowledge,
    project_router as v2_project,
    memory_router as v2_memory,
    scoped_search_router as v2_scoped_search,
    search_router as v2_search,
    resource_router as v2_resource,
    directory_router as v2_directory,
    prompt_router as v2_prompt,
    importer_router as v2_importer,
    schema_router as v2_schema,
    inspect_router as v2_inspect,
)
from basic_memory.index.note_content_materialization import drain_pending_materializations
from basic_memory.config import init_api_logging
from basic_memory.index.local_schedulers import drain_background_tasks
from basic_memory.services.exceptions import EntityAlreadyExistsError
from basic_memory.services.initialization import initialize_app
from basic_memory.runtime.git_routing import local_git_routing
from basic_memory import db
from basic_memory.workspace_context import (
    WORKSPACE_SLUG_HEADER,
    WORKSPACE_TYPE_HEADER,
    workspace_permalink_context_validation_error,
    workspace_permalink_context,
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    """Own Git recovery or legacy watching until requests have drained."""
    init_api_logging()
    container = ApiContainer.create()
    engine_was_none = db._engine is None
    absent = object()
    previous = {
        name: getattr(app.state, name, absent)
        for name in ("container", "engine", "session_maker", "watch_coordinator")
    }
    with installed_container(container), local_git_routing(bool(container.config.git_repositories)):
        app.state.container = container
        try:
            async with AsyncExitStack() as resources:
                with logfire.span(
                    "api.lifecycle.startup", entrypoint="api", mode=container.mode.name.lower()
                ):
                    logger.info(f"Starting Basic Memory API (mode={container.mode.name})")
                    await initialize_app(container.config)
                    engine, session_maker = await container.init_database()
                    app.state.engine = engine
                    app.state.session_maker = session_maker
                    logger.info("Database connections cached in app state")
                    if container.config.git_repositories:
                        await drain_pending_materializations()
                        await drain_background_tasks()
                    container.git_host = await resources.enter_async_context(
                        run_git_host(
                            container.config, session_maker, read_cache=container.read_cache
                        )
                    )
                    watcher = container.create_watch_coordinator()
                    await watcher.start()
                    app.state.watch_coordinator = watcher
                try:
                    yield
                finally:
                    with logfire.span(
                        "api.lifecycle.shutdown", entrypoint="api", mode=container.mode.name.lower()
                    ):
                        logger.info("Shutting down Basic Memory API")
                        await watcher.stop()
                        # Close admission and await owned Git tasks before the DB.
                        await resources.aclose()
                        container.git_host = None
                        await drain_pending_materializations()
                        await drain_background_tasks()
        finally:
            container.git_host = None
            if engine_was_none:
                await container.shutdown_database()
            for name, value in previous.items():
                if value is absent:
                    if hasattr(app.state, name):
                        delattr(app.state, name)
                else:
                    setattr(app.state, name, value)


# Initialize FastAPI app
app = FastAPI(
    title="Basic Memory API",
    description="Knowledge graph API for basic-memory",
    version=version,
    lifespan=lifespan,
    dependencies=[Depends(enforce_git_api_policy)],
)


# fastapi-redis-sdk's response capture. Registered before the workspace middleware
# so capture runs inside it; cached routes opt in with ``cached_read`` dependencies,
# and every other response passes straight through. The SDK's hit handler is
# registered too but never reached: ``ReadCacheRoute`` returns hits instead of
# raising them through the app.
add_redis_caching(app)
# fastapi-redis-sdk reads this module constant on every capture and defaults it
# to 10 MiB. The package re-exports a function named ``cache``, which shadows the
# submodule on attribute access, so the module is fetched by name.
setattr(
    importlib.import_module("redis_fastapi.cache"),
    "MAX_CACHEABLE_BODY_SIZE",
    READ_CACHE_MAX_PAYLOAD_BYTES,
)


@app.middleware("http")
async def workspace_permalink_context_middleware(request: Request, call_next):
    """Populate workspace permalink context from request headers."""
    workspace_slug = request.headers.get(WORKSPACE_SLUG_HEADER)
    workspace_type = request.headers.get(WORKSPACE_TYPE_HEADER)

    validation_error = workspace_permalink_context_validation_error(workspace_slug, workspace_type)
    if validation_error is not None:
        return JSONResponse(
            status_code=400,
            content={"detail": validation_error},
        )

    if not workspace_slug:
        return await call_next(request)

    # ContextVar state remains active across the awaited downstream handler while
    # this context manager is open, so entity creation can see request metadata.
    with workspace_permalink_context(
        workspace_slug=workspace_slug,
        workspace_type=workspace_type,
    ):
        return await call_next(request)


app.include_router(git_router, prefix="/v2/projects/{project_id}")
app.include_router(git_transaction_router, prefix="/v2/projects/{project_id}")
app.include_router(git_move_router, prefix="/v2/projects/{project_id}")
app.include_router(git_host_move_router, prefix="/v2")

# Include v2 routers FIRST (more specific paths must match before /{project} catch-all)
app.include_router(v2_accepted_content, prefix="/v2/projects/{project_id}")
app.include_router(v2_knowledge, prefix="/v2/projects/{project_id}")
app.include_router(v2_memory, prefix="/v2/projects/{project_id}")
app.include_router(v2_search, prefix="/v2/projects/{project_id}")
app.include_router(v2_resource, prefix="/v2/projects/{project_id}")
app.include_router(v2_directory, prefix="/v2/projects/{project_id}")
app.include_router(v2_prompt, prefix="/v2/projects/{project_id}")
app.include_router(v2_importer, prefix="/v2/projects/{project_id}")
app.include_router(v2_schema, prefix="/v2/projects/{project_id}")
app.include_router(v2_inspect, prefix="/v2/projects/{project_id}")
app.include_router(v2_project, prefix="/v2")
# Database-scoped search: one query over an explicit set of projects.
app.include_router(v2_scoped_search, prefix="/v2")

# Legacy web app proxy paths (compat with /proxy/projects/projects)
app.include_router(v2_project, prefix="/proxy/projects")

# V2 routers are the only public API surface


@app.exception_handler(GitTransactionError)
async def git_transaction_error_handler(request: Request, exc: GitTransactionError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc)},
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(EntityAlreadyExistsError)
async def entity_already_exists_error_handler(request: Request, exc: EntityAlreadyExistsError):
    """Handle entity creation conflicts (e.g., file already exists).

    This is expected behavior when users try to create notes that exist,
    so log at INFO level instead of ERROR.
    """
    logger.info(
        "Entity already exists",
        url=str(request.url),
        method=request.method,
        path=request.url.path,
        error=str(exc),
    )
    return await http_exception_handler(
        request,
        HTTPException(
            status_code=409,
            detail="Note already exists. Use bm_edit to modify it, or delete it first.",
        ),
    )


@app.exception_handler(Exception)
async def exception_handler(request, exc):  # pragma: no cover
    logger.exception(
        "API unhandled exception",
        url=str(request.url),
        method=request.method,
        client=request.client.host if request.client else None,
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    return await http_exception_handler(
        request,
        HTTPException(status_code=500, detail="Internal server error"),
    )
