"""BM-009: the way back for create-project removes an empty project and nothing else."""

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from filelock import FileLock
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import BasicMemoryConfig
from basic_memory.models import Project
from basic_memory.services import git_project_remove
from basic_memory.services.git_peer_setup import setup_git_peers
from basic_memory.services.git_project_create import (
    GitProjectCreateError,
    NewGitProject,
    create_git_projects,
)
from basic_memory.services.git_project_remove import remove_empty_git_projects

from tests.git_host_support import host_setup as host_setup


async def caught_up(project_id: str) -> None:
    async with asyncio.timeout(60):
        host = resolve_container().git_host
        assert host is not None
        worker = host.runtime_for(project_id).worker
        while not (await worker.status()).caught_up:
            worker.notify()
            await asyncio.sleep(0.05)


async def create(tmp_path: Path, config_manager, *names: str):
    await db.shutdown_db()  # the offline tools open the database themselves
    return await create_git_projects(
        [NewGitProject(name) for name in names],
        base_dir=tmp_path / "bm",
        repositories_dir=tmp_path / "bm" / "repositories",
        manager=config_manager,
    )


async def project_names(host_setup) -> list[str]:
    _, factory = await db.get_or_create_db(host_setup.config.database_path)
    async with db.scoped_session(factory) as session:
        return list(await session.scalars(select(Project.name).order_by(Project.id)))


def profile(config_manager) -> dict[str, Any]:
    return json.loads(config_manager.config_file.read_text())


@pytest.mark.asyncio
async def test_create_then_remove_restores_the_installation(host_setup, config_manager, tmp_path):
    config_manager.config_file.chmod(0o600)
    before = profile(config_manager)
    created = await create(tmp_path, config_manager, "haus", "meldungen")
    assert profile(config_manager) != before
    await db.shutdown_db()

    removed = await remove_empty_git_projects(["haus", "meldungen"], manager=config_manager)

    assert [r.name for r in removed] == ["haus", "meldungen"]
    assert all(not r.leftovers and not r.had_peer for r in removed)
    assert profile(config_manager) == before
    assert stat.S_IMODE(config_manager.config_file.stat().st_mode) == 0o600
    assert await project_names(host_setup) == ["test-project"]
    for item in created:
        assert not item.path.exists()
        assert not item.repository.exists()
    # The name is free again, and the new project gets a new identity.
    again = await create(tmp_path, config_manager, "haus")
    assert again[0].project_id != created[0].project_id


@pytest.mark.asyncio
async def test_the_peer_entries_go_with_the_project_but_the_peer_stays(
    host_setup, config_manager, tmp_path
):
    created = await create(tmp_path, config_manager, "haus")
    await setup_git_peers(
        ["haus"], peer_base=str(tmp_path / "peer"), merge="auto", manager=config_manager
    )
    await db.shutdown_db()
    removed = await remove_empty_git_projects(["haus"], manager=config_manager)
    assert removed[0].had_peer
    config = BasicMemoryConfig.model_validate_json(config_manager.config_file.read_bytes())
    assert created[0].project_id not in config.git_sync_remotes
    assert created[0].project_id not in config.git_sync_merge
    assert (tmp_path / "peer" / "haus.git").is_dir()  # not ours to delete


@pytest.mark.asyncio
async def test_a_project_with_notes_stays_and_nothing_changes(host_setup, config_manager, tmp_path):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    created = await create(tmp_path, config_manager, "haus", "meldungen")
    async with Client(mcp) as client:
        await caught_up(str(created[0].project_id))
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
        await caught_up(str(created[0].project_id))
    await db.shutdown_db()
    before = config_manager.config_file.read_bytes()

    with pytest.raises(GitProjectCreateError, match="with notes are not removed: haus"):
        await remove_empty_git_projects(["haus", "meldungen"], manager=config_manager)

    assert config_manager.config_file.read_bytes() == before
    assert await project_names(host_setup) == ["test-project", "haus", "meldungen"]
    assert created[1].repository.is_dir()  # the empty one in the same request stays too


@pytest.mark.asyncio
async def test_a_used_history_without_notes_is_refused(host_setup, config_manager, tmp_path):
    from uuid import uuid4

    from basic_memory.repository.git_project_repository import (
        GitProjectRepository,
        settle_git_publication,
    )

    created = await create(tmp_path, config_manager, "haus")
    repository = GitProjectRepository(created[0].repository)
    prepared = repository.prepare(
        transaction_id=uuid4(),
        base_commit=repository.snapshot().commit_id,
        files={"a.md": "# a\n"},
        note_ids={"a.md": uuid4()},
        message="note",
    )
    await settle_git_publication(repository, prepared)
    await db.shutdown_db()
    with pytest.raises(GitProjectCreateError, match="has been used"):
        await remove_empty_git_projects(["haus"], manager=config_manager)
    assert created[0].repository.is_dir()


@pytest.mark.asyncio
async def test_refusals_change_nothing(host_setup, config_manager, tmp_path):
    created = await create(tmp_path, config_manager, "haus")
    await db.shutdown_db()
    before = config_manager.config_file.read_bytes()

    with pytest.raises(GitProjectCreateError, match="Unknown project"):
        await remove_empty_git_projects(["gibtsnicht"], manager=config_manager)
    with pytest.raises(GitProjectCreateError, match="No project named"):
        await remove_empty_git_projects([], manager=config_manager)
    with pytest.raises(GitProjectCreateError, match="At least one project must remain"):
        await remove_empty_git_projects(["test-project", "haus"], manager=config_manager)
    with pytest.raises(GitProjectCreateError, match="holds files"):
        (created[0].path / "fremd.md").write_text("x")
        await remove_empty_git_projects(["haus"], manager=config_manager)
    (created[0].path / "fremd.md").unlink()
    # A running service holds the project locks.
    root = Path(host_setup.project.path)
    (root / ".basic-memory").mkdir(exist_ok=True)
    with FileLock(root / ".basic-memory" / "git-projection.lock", thread_local=False):
        with pytest.raises(GitProjectCreateError, match="stop it first"):
            await remove_empty_git_projects(["haus"], manager=config_manager)

    assert config_manager.config_file.read_bytes() == before
    assert await project_names(host_setup) == ["test-project", "haus"]
    assert created[0].repository.is_dir()


@pytest.mark.asyncio
async def test_the_default_project_is_not_removed(host_setup, config_manager, tmp_path):
    await create(tmp_path, config_manager, "haus")
    raw = profile(config_manager)
    raw["default_project"] = "haus"
    config_manager.config_file.write_text(json.dumps(raw))
    await db.shutdown_db()
    with pytest.raises(GitProjectCreateError, match="default project"):
        await remove_empty_git_projects(["haus"], manager=config_manager)


@pytest.mark.asyncio
async def test_a_failing_configuration_write_keeps_rows_and_directories(
    host_setup, config_manager, tmp_path, monkeypatch
):
    created = await create(tmp_path, config_manager, "haus")
    await db.shutdown_db()
    before = config_manager.config_file.read_bytes()

    def broken(path: Path, data: dict[str, Any]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(git_project_remove, "_write_config_atomically", broken)
    with pytest.raises(OSError, match="disk full"):
        await remove_empty_git_projects(["haus"], manager=config_manager)

    assert config_manager.config_file.read_bytes() == before
    assert await project_names(host_setup) == ["test-project", "haus"]
    assert created[0].repository.is_dir() and created[0].path.is_dir()


@pytest.mark.asyncio
async def test_a_failure_after_the_configuration_write_restores_it(
    host_setup, config_manager, tmp_path, monkeypatch
):
    await create(tmp_path, config_manager, "haus")
    await db.shutdown_db()
    before = config_manager.config_file.read_bytes()
    real_write = git_project_remove._write_config_atomically

    def write_then_fail(path: Path, data: dict[str, Any]) -> None:
        real_write(path, data)
        raise OSError("commit failed")

    monkeypatch.setattr(git_project_remove, "_write_config_atomically", write_then_fail)
    with pytest.raises(OSError, match="commit failed"):
        await remove_empty_git_projects(["haus"], manager=config_manager)

    assert config_manager.config_file.read_bytes() == before
    assert await project_names(host_setup) == ["test-project", "haus"]
