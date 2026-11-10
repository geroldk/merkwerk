"""Host selection, routing ownership and refusal contracts outside request lifespans."""

from uuid import uuid4

import pytest
import typer
from fastapi import HTTPException
from starlette.requests import Request
from starlette.routing import Route
from typer.core import TyperCommand

from basic_memory.api.git_policy import GIT_NOTE_WRITES, enforce_git_api_policy
from basic_memory.cli.app import app_callback
from basic_memory.cli.container import CliContainer
from basic_memory.deps.git import get_git_host, get_git_runtime
from basic_memory.runtime.git_routing import git_local_routing_active, local_git_routing
from basic_memory.runtime.mode import RuntimeMode


def test_nested_local_routing_owners_restore_outer_state_on_failure():
    assert not git_local_routing_active()
    with local_git_routing(False):
        assert not git_local_routing_active()
        with local_git_routing(True):
            assert git_local_routing_active()
            with pytest.raises(RuntimeError), local_git_routing(True):
                raise RuntimeError("Failed second owner")
            assert git_local_routing_active()
        assert not git_local_routing_active()


@pytest.mark.asyncio
async def test_legacy_host_dependencies_do_not_create_git_runtime(app_config):
    assert await get_git_host(Request({"type": "http"}), app_config) is None
    assert await get_git_runtime(1, None) is None
    await enforce_git_api_policy(Request({"type": "http"}), app_config)


def test_git_note_writes_allow_the_one_commit_folder_move_but_not_the_upstream_one():
    """V1: folders move as one Git commit; the note-by-note upstream route stays refused."""
    assert "move_git_directory" in GIT_NOTE_WRITES
    assert "move_directory" not in GIT_NOTE_WRITES


@pytest.mark.asyncio
async def test_git_folder_move_route_passes_the_policy_and_upstream_does_not(app_config, tmp_path):
    from basic_memory.api.v2.routers.knowledge_router import router
    from fastapi.routing import APIRoute

    app_config.git_repositories = {uuid4(): tmp_path / "project.git"}
    routes = {route.name: route for route in router.routes if isinstance(route, APIRoute)}

    def request(name: str) -> Request:
        return Request({"type": "http", "method": "POST", "route": routes[name]})

    await enforce_git_api_policy(request("move_git_directory"), app_config)
    with pytest.raises(HTTPException) as error:
        await enforce_git_api_policy(request("move_directory"), app_config)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_non_api_route_is_refused_on_git_host(app_config, tmp_path):
    app_config.git_repositories = {uuid4(): tmp_path / "project.git"}

    async def endpoint(request):
        pass

    request = Request({"type": "http", "route": Route("/unknown", endpoint=endpoint)})
    with pytest.raises(HTTPException) as error:
        await enforce_git_api_policy(request, app_config)
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    "command", ["reindex", "prune", "project", "cloud", "update", "import", "reset"]
)
def test_cli_refuses_unconnected_commands_before_initialization(
    app_config, tmp_path, monkeypatch, command
):
    import basic_memory.cli.app as cli_app
    import basic_memory.cli.container as cli_container

    app_config.git_repositories = {uuid4(): tmp_path / "project.git"}
    container = CliContainer(config=app_config, mode=RuntimeMode.LOCAL)
    monkeypatch.setattr(CliContainer, "create", lambda: container)
    monkeypatch.setattr(cli_app, "init_cli_logging", lambda: None)
    monkeypatch.setattr(cli_container, "_container", None)
    ctx = typer.Context(TyperCommand("basic-memory"))
    try:
        ctx.invoked_subcommand = command
        with pytest.raises(typer.Exit) as error:
            app_callback(ctx, version=False)
        assert error.value.exit_code == 2
        assert not git_local_routing_active()
    finally:
        ctx.close()


def test_cli_read_context_keeps_routing_local_and_skips_upstream_auto_update(
    app_config, tmp_path, monkeypatch
):
    import basic_memory.cli.app as cli_app
    import basic_memory.cli.container as cli_container

    app_config.git_repositories = {uuid4(): tmp_path / "project.git"}
    container = CliContainer(config=app_config, mode=RuntimeMode.LOCAL)
    monkeypatch.setattr(CliContainer, "create", lambda: container)
    monkeypatch.setattr(cli_app, "init_cli_logging", lambda: None)
    monkeypatch.setattr(cli_container, "_container", None)
    monkeypatch.setattr(cli_app, "maybe_show_init_line", lambda command: None)

    def forbidden(command):
        pytest.fail("Git host must not auto-update to an upstream legacy writer")

    monkeypatch.setattr(cli_app, "maybe_run_periodic_auto_update", forbidden)
    monkeypatch.setattr(cli_app, "maybe_show_cloud_promo", forbidden)
    ctx = typer.Context(TyperCommand("basic-memory"))
    try:
        ctx.invoked_subcommand = "status"
        app_callback(ctx, version=False)
        assert git_local_routing_active()
    finally:
        ctx.close()
    assert not git_local_routing_active()
