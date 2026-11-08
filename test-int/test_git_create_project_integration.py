"""BM-009: new Git projects on a stopped host start with it, commit once per write, and back up."""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastmcp import Client
from filelock import FileLock
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import BasicMemoryConfig
from basic_memory.index.git_project_ownership import own_git_project
from basic_memory.models import Project
from basic_memory.services import git_project_create
from basic_memory.services.git_backup import create_git_backup, restore_git_backup
from basic_memory.services.git_project_create import (
    GitProjectCreateError,
    NewGitProject,
    create_git_projects,
)

from tests.git_host_support import host_setup as host_setup


async def caught_up(project_id: str) -> None:
    async with asyncio.timeout(60):
        host = resolve_container().git_host
        assert host is not None
        worker = host.runtime_for(project_id).worker
        while not (await worker.status()).caught_up:
            worker.notify()
            await asyncio.sleep(0.05)


def commit_count(repository: Path) -> int:
    out = subprocess.run(
        ["git", "--git-dir", str(repository), "rev-list", "--count", "main"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout)


async def create(host_setup, tmp_path: Path, *requests: NewGitProject, config_manager):
    await db.shutdown_db()  # the offline tool opens the database itself
    return await create_git_projects(
        requests,
        base_dir=tmp_path / "bm",
        repositories_dir=tmp_path / "bm" / "repositories",
        manager=config_manager,
    )


@pytest.mark.asyncio
async def test_created_projects_start_with_the_host_and_commit_once_per_write(
    host_setup, config_manager, tmp_path
):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    created = await create(
        host_setup,
        tmp_path,
        NewGitProject("haus"),
        NewGitProject("meldungen"),
        config_manager=config_manager,
    )
    assert [c.name for c in created] == ["haus", "meldungen"]
    config = BasicMemoryConfig.model_validate_json(config_manager.config_file.read_bytes())
    assert set(config.projects) == {"test-project", "haus", "meldungen"}
    assert not config.git_sync_remotes  # no peer unless asked for
    for item in created:
        assert (item.path / ".basic-memory" / "git-binding.json").is_file()
        assert (item.repository / "bm-local-binding.json").is_file()
        assert config.git_repositories[item.project_id] == item.repository
    haus = created[0]
    before = commit_count(haus.repository)

    async with Client(mcp) as client:
        await caught_up(str(haus.project_id))
        written = await client.call_tool(
            "bm_write",
            {
                "project": "haus",
                "title": "Dachboden",
                "directory": "a",
                "content": "- [fact] x\n",
                "output_format": "json",
            },
        )
        assert not written.is_error, written.content
        await caught_up(str(haus.project_id))
    assert commit_count(haus.repository) == before + 1
    assert commit_count(created[1].repository) == 1  # the other new project stays untouched


@pytest.mark.asyncio
async def test_a_backup_after_creation_includes_every_project(host_setup, config_manager, tmp_path):
    # The existing project got its binding markers at its first start; give it the same.
    _, factory = await db.get_or_create_db(host_setup.config.database_path)
    async with own_git_project(
        host_setup.repository,
        project_id=host_setup.project.id,
        project_external_id=host_setup.project.external_id,
        project_root=Path(host_setup.project.path),
        session_maker=factory,
    ):
        pass
    created = await create(
        host_setup, tmp_path, NewGitProject("ake"), config_manager=config_manager
    )
    backup = tmp_path / "backup.bm.zip"
    manifest = create_git_backup(config_manager.config_file, backup)
    assert {p.name for p in manifest.projects} == {"test-project", "ake"}
    restored = tmp_path / "restored"
    restore_git_backup(backup, restored)
    config = BasicMemoryConfig.model_validate_json((restored / "config/config.json").read_bytes())
    assert created[0].project_id in config.git_repositories


@pytest.mark.asyncio
async def test_a_failure_before_the_config_is_replaced_leaves_nothing_behind(
    host_setup, config_manager, tmp_path, monkeypatch
):
    before = config_manager.config_file.read_bytes()

    def broken(path: Path, data: dict[str, Any]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(git_project_create, "_write_config_atomically", broken)
    with pytest.raises(OSError, match="disk full"):
        await create(
            host_setup,
            tmp_path,
            NewGitProject("haus"),
            NewGitProject("ake"),
            config_manager=config_manager,
        )
    assert config_manager.config_file.read_bytes() == before
    assert not (tmp_path / "bm").exists() or not any((tmp_path / "bm").iterdir())
    _, factory = await db.get_or_create_db(host_setup.config.database_path)
    async with db.scoped_session(factory) as session:
        names = list(await session.scalars(select(Project.name)))
    assert names == ["test-project"]


@pytest.mark.asyncio
async def test_refusals_change_nothing(host_setup, config_manager, tmp_path):
    before = config_manager.config_file.read_bytes()
    with pytest.raises(GitProjectCreateError, match="already taken"):
        await create(
            host_setup, tmp_path, NewGitProject("test-project"), config_manager=config_manager
        )
    # A running service holds the project locks.
    root = Path(host_setup.project.path)
    (root / ".basic-memory").mkdir(exist_ok=True)
    with FileLock(root / ".basic-memory" / "git-projection.lock", thread_local=False):
        with pytest.raises(GitProjectCreateError, match="stop it first"):
            await create(host_setup, tmp_path, NewGitProject("haus"), config_manager=config_manager)
    assert config_manager.config_file.read_bytes() == before
    assert not (tmp_path / "bm").exists()


@pytest.mark.asyncio
async def test_a_profile_that_is_not_a_git_host_is_refused(host_setup, config_manager, tmp_path):
    raw: dict[str, Any] = json.loads(config_manager.config_file.read_text())
    raw["git_repositories"] = {}
    config_manager.config_file.write_text(json.dumps(raw))
    with pytest.raises(GitProjectCreateError, match="not a Git host"):
        await create(host_setup, tmp_path, NewGitProject("haus"), config_manager=config_manager)
    assert isinstance(UUID(host_setup.project.external_id), UUID)
