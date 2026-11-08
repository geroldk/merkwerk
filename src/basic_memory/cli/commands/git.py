"""Explicit offline backups; no initialization of the caller's database."""

import os
from pathlib import Path
import subprocess
import sys

import typer

from basic_memory.cli.app import app

git_app = typer.Typer(
    help="Offline backup, restore, project creation and removal for adopted Git projects"
)
app.add_typer(git_app, name="git")


@git_app.command("backup")
def backup(
    destination: Path = typer.Argument(..., help="New private backup archive outside all sources"),
    config: Path = typer.Option(
        ..., "--config", help="Explicit source config.json; host must stop"
    ),
) -> None:
    from basic_memory.services.git_backup import create_git_backup

    manifest = create_git_backup(config, destination)
    typer.echo(f"Backup complete: {destination} ({len(manifest.projects)} projects)")


@git_app.command("restore")
def restore(
    backup_path: Path = typer.Argument(..., help="Private archive created by git backup"),
    destination: Path = typer.Argument(
        ..., help="New instance directory; existing targets refused"
    ),
    timeout: float = typer.Option(300, min=0.1, help="Index rebuild time limit in seconds"),
) -> None:
    from basic_memory.services.git_backup import restore_git_backup

    restore_git_backup(backup_path, destination)
    finish_restore(destination, timeout)


@git_app.command("finish-restore")
def finish_restore(
    destination: Path = typer.Argument(..., help="Incomplete independent restored instance"),
    timeout: float = typer.Option(300, min=0.1, help="Index rebuild time limit in seconds"),
) -> None:
    # No inherited Basic Memory routing or database override may redirect the
    # rebuild into the caller's installation. HOME and credentials stay unchanged.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("BASIC_MEMORY_")}
    result = subprocess.run(
        [sys.executable, "-m", "basic_memory.cli.git_recovery", str(destination), str(timeout)],
        env=environment,
        check=False,
    )
    if (
        result.returncode != 0
        or not (destination / "READY.json").is_file()
        or (destination / "RESTORE_INCOMPLETE").exists()
    ):
        typer.echo(
            "Restore incomplete; retained target can be retried with git finish-restore.", err=True
        )
        raise typer.Exit(1)
    typer.echo(
        f"Restore complete: {destination} (content, files, graph/full text, vectors rebuilt)"
    )


@git_app.command("create-project")
def create_project(
    projects: list[str] = typer.Argument(
        ..., help="NAME or NAME=/absolute/path (default path: <root>/NAME)"
    ),
    root: Path = typer.Option(
        Path.home() / "bm", "--root", help="Directory below which new project roots are created"
    ),
    repositories: Path | None = typer.Option(
        None,
        "--repositories",
        help="Directory of the bare repositories (default: <root>/repositories)",
    ),
) -> None:
    """Create new, empty Git projects. The service must be stopped; restart it afterwards."""
    import asyncio

    from basic_memory.services.git_project_create import (
        GitProjectCreateError,
        NewGitProject,
        create_git_projects,
    )

    requests = []
    for item in projects:
        name, separator, path = item.partition("=")
        requests.append(NewGitProject(name, Path(path) if separator else None))
    try:
        created = asyncio.run(
            create_git_projects(
                requests, base_dir=root, repositories_dir=repositories or root / "repositories"
            )
        )
    except (GitProjectCreateError, ValueError) as error:  # ValueError: GitProjectionConflict
        typer.echo(f"Refused: {error}", err=True)
        raise typer.Exit(1) from error
    for item in created:
        typer.echo(f"{item.name}: {item.project_id} root={item.path} repository={item.repository}")
    typer.echo(
        "Start the service to use them. Outside this tool: sync-projects.json, a peer "
        "(git_sync_remotes) and any gateway project lists are not changed."
    )


@git_app.command("remove-project")
def remove_project(
    projects: list[str] = typer.Argument(..., help="Names of empty projects to remove"),
) -> None:
    """Remove empty Git projects again (the way back for create-project). Service stopped."""
    import asyncio

    from basic_memory.services.git_project_create import GitProjectCreateError
    from basic_memory.services.git_project_remove import remove_empty_git_projects

    try:
        removed = asyncio.run(remove_empty_git_projects(projects))
    except (GitProjectCreateError, ValueError) as error:  # ValueError: GitProjectionConflict
        typer.echo(f"Refused: {error}", err=True)
        raise typer.Exit(1) from error
    for item in removed:
        typer.echo(f"{item.name}: removed ({item.project_id})")
        if item.had_peer:
            typer.echo("  Its peer repository was not touched; remove it by hand if wanted.")
        for leftover in item.leftovers:
            typer.echo(f"  Could not delete {leftover}; remove it by hand.", err=True)
    typer.echo("Outside this tool: sync-projects.json and gateway project lists are not changed.")


@git_app.command("peer-setup")
def peer_setup(
    projects: list[str] = typer.Argument(..., help="Project names"),
    peer: str | None = typer.Option(
        None,
        "--peer",
        help="Peer base: an absolute directory or ssh://host/dir; each project uses BASE/NAME.git",
    ),
    merge: str = typer.Option(
        "off", "--merge", help="auto or off (automatic merge of diverged history)"
    ),
    remove: bool = typer.Option(False, "--remove", help="Remove the peer of the named projects"),
) -> None:
    """Set or remove the sync peer of named projects. The service must be stopped."""
    import asyncio

    from basic_memory.services.git_project_create import GitProjectCreateError
    from basic_memory.services.git_peer_setup import setup_git_peers

    if merge not in {"auto", "off"} or remove == (peer is not None):
        typer.echo("Refused: give either --peer BASE (with --merge auto|off) or --remove", err=True)
        raise typer.Exit(1)
    try:
        changes = asyncio.run(
            setup_git_peers(projects, peer_base=None if remove else peer, merge=merge)
        )
    except (GitProjectCreateError, ValueError) as error:
        typer.echo(f"Refused: {error}", err=True)
        raise typer.Exit(1) from error
    for change in changes:
        target = f"{change.url} merge={change.merge}" if change.url else "no peer"
        typer.echo(f"{change.name}: {target}")
    typer.echo("Restart the service for the change to take effect.")
