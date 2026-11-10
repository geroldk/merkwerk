"""Shared real Git-host fixtures, importable under pytest's importlib mode."""

import asyncio
import os
import time
from dataclasses import dataclass
from uuid import UUID

import pytest

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import BasicMemoryConfig, ProjectEntry
from basic_memory.models import Project
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.project_repository import ProjectRepository


@dataclass(frozen=True)
class HostSetup:
    config: BasicMemoryConfig
    project: Project
    repository: GitProjectRepository


@pytest.fixture
async def host_setup(app_config, config_manager, tmp_path, monkeypatch):
    await db.shutdown_db()
    root = tmp_path / "notes"
    root.mkdir()
    app_config.projects = {"test-project": ProjectEntry(path=str(root))}
    app_config.index_changes = True
    # Exercise real local watch/follow-up decisions, not the test-mode shortcuts.
    monkeypatch.setattr(BasicMemoryConfig, "is_test_env", property(lambda self: False))
    config_manager.save_config(app_config)
    _, factory = await db.get_or_create_db(app_config.database_path)
    async with db.scoped_session(factory) as session:
        project = await ProjectRepository().create(
            session,
            {
                "name": "test-project",
                "path": str(root),
                "is_active": True,
                "is_default": True,
            },
        )
    repository = GitProjectRepository.create(
        tmp_path / "project.git", project_external_id=UUID(project.external_id)
    )
    app_config.git_repositories = {UUID(project.external_id): repository.path}
    config_manager.save_config(app_config)
    try:
        yield HostSetup(app_config, project, repository)
    finally:
        await db.shutdown_db()


async def ready(client, project_id):
    async with asyncio.timeout(30):
        while True:
            response = await client.get(f"/v2/projects/{project_id}/git/status")
            assert response.status_code == 200, response.text
            if response.json()["caught_up"]:
                return response.json()
            host = resolve_container().git_host
            assert host is not None
            host.runtime_for(project_id).worker.notify()
            await asyncio.sleep(0.03)


@pytest.fixture
def local_time_with_offset():
    """Run in Europe/Berlin, a zone with a non-zero UTC offset.

    In UTC (as on CI runners) a UTC wall time stored as naive local time reads back as the
    right instant, so a test of stored times passes with or without the bug it guards.
    """
    # Trigger: Windows has no time.tzset, so the process zone cannot be switched.
    # Why: skipping would also drop the zone-independent checks of the same tests.
    # Outcome: the test runs in the runner's own zone; Linux and macOS rows pin it.
    if not hasattr(time, "tzset"):
        yield
        return
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Berlin"
    time.tzset()
    try:
        # Fail loudly if the zone database is missing: UTC would hide the bug again.
        assert time.localtime(0).tm_gmtoff != 0, "Europe/Berlin is not available here"
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()
