"""BM-009: peers are set by project name, offline, and the host starts and syncs with them."""

import asyncio
import stat
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from filelock import FileLock
from pydantic import ValidationError

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import BasicMemoryConfig
from basic_memory.services import git_peer_setup
from basic_memory.services.git_peer_setup import setup_git_peers
from basic_memory.services.git_project_create import GitProjectCreateError

from tests.git_host_support import host_setup as host_setup


def load(config_manager) -> BasicMemoryConfig:
    return BasicMemoryConfig.model_validate_json(config_manager.config_file.read_bytes())


async def stopped_setup(config_manager, *args: Any, **kwargs: Any):
    await db.shutdown_db()  # the offline tool opens the database itself
    return await setup_git_peers(*args, manager=config_manager, **kwargs)


@pytest.mark.asyncio
async def test_a_local_peer_is_created_filled_by_the_first_sync_and_can_be_removed(
    host_setup, config_manager, tmp_path
):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    peer = tmp_path / "peer"
    changes = await stopped_setup(
        config_manager, ["test-project"], peer_base=str(peer), merge="auto"
    )
    assert [(c.name, c.url, c.merge) for c in changes] == [
        ("test-project", f"{peer}/test-project.git", "auto")
    ]
    assert (peer / "test-project.git" / "HEAD").is_file()
    config = load(config_manager)
    project_id = host_setup.repository.snapshot().manifest.project_external_id
    assert config.git_sync_remotes[project_id].url == f"{peer}/test-project.git"
    assert config.git_sync_merge == {project_id: "auto"}

    async with Client(mcp) as client:
        async with asyncio.timeout(60):
            host = resolve_container().git_host
            assert host is not None
            worker = host.runtime_for(host_setup.project.external_id).worker
            while not (await worker.status()).caught_up:
                worker.notify()
                await asyncio.sleep(0.05)
        synced = await client.call_tool(
            "bm_project",
            {"command": "sync", "project": "test-project", "output_format": "json"},
        )
        assert not synced.is_error, synced.content
        assert synced.data["last_attempt"]["outcome"] == "pushed"

    removed = await stopped_setup(config_manager, ["test-project"], peer_base=None)
    assert removed[0].url is None
    config = load(config_manager)
    assert config.git_sync_remotes == {} and config.git_sync_merge == {}


@pytest.mark.asyncio
async def test_an_ssh_base_creates_nothing_locally_and_merge_defaults_to_off(
    host_setup, config_manager
):
    changes = await stopped_setup(
        config_manager, ["test-project"], peer_base="ssh://nuc1/home/gerold/bm/peer"
    )
    assert changes[0].url == "ssh://nuc1/home/gerold/bm/peer/test-project.git"
    assert load(config_manager).git_sync_merge == {}


@pytest.mark.asyncio
async def test_refusals_change_nothing(host_setup, config_manager, tmp_path):
    before = config_manager.config_file.read_bytes()
    with pytest.raises(GitProjectCreateError, match="Unknown project"):
        await stopped_setup(config_manager, ["nope"], peer_base=str(tmp_path / "peer"))
    with pytest.raises(GitProjectCreateError, match="No project named"):
        await stopped_setup(config_manager, [], peer_base=str(tmp_path / "peer"))
    with pytest.raises(ValidationError):
        await stopped_setup(config_manager, ["test-project"], peer_base="relative/peer")
    root = Path(host_setup.project.path)
    (root / ".basic-memory").mkdir(exist_ok=True)
    with FileLock(root / ".basic-memory" / "git-projection.lock", thread_local=False):
        with pytest.raises(GitProjectCreateError, match="stop it first"):
            await stopped_setup(config_manager, ["test-project"], peer_base=str(tmp_path / "peer"))
    assert config_manager.config_file.read_bytes() == before
    assert not (tmp_path / "peer").exists()


@pytest.mark.asyncio
async def test_a_failed_write_removes_the_peer_it_created(
    host_setup, config_manager, tmp_path, monkeypatch
):
    before = config_manager.config_file.read_bytes()

    def broken(path: Path, data: dict[str, Any]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(git_peer_setup, "_write_config_atomically", broken)
    with pytest.raises(OSError, match="disk full"):
        await stopped_setup(config_manager, ["test-project"], peer_base=str(tmp_path / "peer"))
    assert config_manager.config_file.read_bytes() == before
    assert not (tmp_path / "peer" / "test-project.git").exists()


@pytest.mark.asyncio
async def test_a_profile_that_is_not_a_git_host_is_refused(host_setup, config_manager, tmp_path):
    raw = load(config_manager)
    raw.git_repositories = {}
    config_manager.config_file.write_bytes(raw.model_dump_json().encode())
    with pytest.raises(GitProjectCreateError, match="not a Git host"):
        await stopped_setup(config_manager, ["test-project"], peer_base=str(tmp_path / "peer"))


@pytest.mark.asyncio
async def test_a_peer_inside_a_project_place_is_refused_and_the_rights_of_the_file_stay(
    host_setup, config_manager, tmp_path
):
    config_manager.config_file.chmod(0o600)
    before = config_manager.config_file.read_bytes()
    root = Path(host_setup.project.path)
    for base in (root / "peers", host_setup.repository.path):
        with pytest.raises(GitProjectCreateError, match="overlaps"):
            await stopped_setup(config_manager, ["test-project"], peer_base=str(base))
    assert config_manager.config_file.read_bytes() == before
    assert not (root / "peers").exists()
    # A good peer works, and the configuration keeps its owner-only rights.
    await stopped_setup(config_manager, ["test-project"], peer_base=str(tmp_path / "peerside"))
    assert stat.S_IMODE(config_manager.config_file.stat().st_mode) == 0o600
