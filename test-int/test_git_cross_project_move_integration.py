"""bm_move across two Git projects: one commit per repository, no silent loss.

Upstream moves a note into another project as a write there followed by a delete here.
On a Git host each half must publish through its own project repository; the move is
deliberately not one atomic commit, because projects are the transaction boundary.
"""

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import pytest
from fastmcp import Client
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config import ProjectEntry
from basic_memory.file_utils import parse_frontmatter, remove_frontmatter
from basic_memory.models import Entity
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.project_repository import ProjectRepository

from tests.git_host_support import host_setup as host_setup


@pytest.fixture
async def second_repository(host_setup, config_manager):
    """A second SQL project with its own Git repository next to `test-project`."""
    root = host_setup.repository.path.parent / "second"
    root.mkdir()
    config = host_setup.config
    config.projects["second"] = ProjectEntry(path=str(root))
    _, factory = await db.get_or_create_db(config.database_path)
    async with db.scoped_session(factory) as session:
        second = await ProjectRepository().create(
            session, {"name": "second", "path": str(root), "is_active": True}
        )
    repository = GitProjectRepository.create(
        root.parent / "second.git", project_external_id=UUID(second.external_id)
    )
    config.git_repositories[UUID(second.external_id)] = repository.path
    config_manager.save_config(config)
    return second, repository


async def caught_up(*project_ids: str) -> None:
    """Wait until the owned workers have projected every published commit."""
    async with asyncio.timeout(30):
        for project_id in project_ids:
            host = resolve_container().git_host
            assert host is not None
            worker = host.runtime_for(project_id).worker
            while not (await worker.status()).caught_up:
                worker.notify()
                await asyncio.sleep(0.03)


@pytest.mark.asyncio
async def test_a_note_moves_between_git_projects_as_two_separate_commits(
    host_setup, second_repository
):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    second, second_repo = second_repository
    first_repo = host_setup.repository
    async with Client(mcp) as client:
        written = await client.call_tool(
            "bm_write",
            {
                "project": "test-project",
                "title": "Mover",
                "directory": "notes",
                "content": "- [fact] Moveprobe\n",
                "output_format": "json",
            },
        )
        assert not written.is_error, written.content
        await caught_up(host_setup.project.external_id)
        source_commit = first_repo.head()
        target_commit = second_repo.head()
        original = first_repo.snapshot().files["notes/Mover.md"]

        moved = await client.call_tool(
            "bm_move",
            {
                "identifier": "notes/Mover.md",
                "destination": "archive/Mover.md",
                "project": "test-project",
                "destination_project": "second",
                "output_format": "json",
            },
            raise_on_error=False,
        )
        assert not moved.is_error, moved.content
        await caught_up(host_setup.project.external_id, second.external_id)

    # Trigger: the move wrote a copy and then deleted the source.
    # Why: each project owns its repository, so both halves publish independently.
    # Outcome: the target gained the note, the source lost it, both heads advanced once.
    target = second_repo.snapshot()
    # The copy keeps the body; its title and frontmatter travel with it. (Its permalink is
    # formed again in the target project; see the permalink tests below.)
    assert (
        original.split("---", 2)[2].strip()
        == target.files["archive/Mover.md"].split("---", 2)[2].strip()
    )
    assert "title: Mover" in target.files["archive/Mover.md"]
    assert "notes/Mover.md" not in first_repo.snapshot().files
    assert first_repo.head() != source_commit
    assert second_repo.head() != target_commit


@pytest.mark.asyncio
async def test_a_stale_checksum_leaves_both_git_projects_untouched(host_setup, second_repository):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    second, second_repo = second_repository
    first_repo = host_setup.repository
    async with Client(mcp) as client:
        written = await client.call_tool(
            "bm_write",
            {
                "project": "test-project",
                "title": "Stale",
                "directory": "notes",
                "content": "- [fact] Staleprobe\n",
                "output_format": "json",
            },
        )
        assert not written.is_error, written.content
        await caught_up(host_setup.project.external_id)
        source_commit = first_repo.head()
        target_commit = second_repo.head()

        moved = await client.call_tool(
            "bm_move",
            {
                "identifier": "notes/Stale.md",
                "destination": "archive/Stale.md",
                "project": "test-project",
                "destination_project": "second",
                "expected_checksum": "0" * 64,
                "output_format": "json",
            },
            raise_on_error=False,
        )
        assert moved.is_error
        await caught_up(host_setup.project.external_id, second.external_id)

    assert first_repo.head() == source_commit
    assert second_repo.head() == target_commit
    assert "notes/Stale.md" in first_repo.snapshot().files
    assert second_repo.snapshot().files == {}


@pytest.mark.asyncio
async def test_a_taken_target_path_keeps_the_source_note(host_setup, second_repository):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    second, second_repo = second_repository
    first_repo = host_setup.repository
    async with Client(mcp) as client:
        for project, content in (("test-project", "Sourceprobe"), ("second", "Targetprobe")):
            written = await client.call_tool(
                "bm_write",
                {
                    "project": project,
                    "title": "Same",
                    "directory": "notes",
                    "content": f"- [fact] {content}\n",
                    "output_format": "json",
                },
            )
            assert not written.is_error, written.content
        await caught_up(host_setup.project.external_id, second.external_id)
        source_commit = first_repo.head()
        target_commit = second_repo.head()

        moved = await client.call_tool(
            "bm_move",
            {
                "identifier": "notes/Same.md",
                "destination": "notes/Same.md",
                "project": "test-project",
                "destination_project": "second",
                "output_format": "json",
            },
            raise_on_error=False,
        )
        assert moved.is_error
        await caught_up(host_setup.project.external_id, second.external_id)

    # Why: the move must never overwrite the target nor drop the source.
    assert first_repo.head() == source_commit
    assert second_repo.head() == target_commit
    assert "Sourceprobe" in first_repo.snapshot().files["notes/Same.md"]
    assert "Targetprobe" in second_repo.snapshot().files["notes/Same.md"]


CREATED = "2026-05-01T08:00:00+02:00"
MODIFIED = "2026-06-02T09:30:00+02:00"


def dated_note(permalink: str) -> str:
    return (
        f"---\ntitle: Jek\npermalink: {permalink}\ntags:\n- aufgabe\n"
        f"created: '{CREATED}'\nmodified: '{MODIFIED}'\n---\n\n# Jek\n\n- [fact] Umzug\n"
    )


async def written(client: Client[Any], content: str) -> None:
    result = await client.call_tool(
        "bm_write",
        {
            "project": "test-project",
            "title": "Jek",
            "directory": "a",
            "content": content,
            "output_format": "json",
        },
    )
    assert not result.is_error, result.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_permalink",
    ["personal/main/aufgaben/jek", "main/aufgaben/jek", "elsewhere/jek"],
    ids=["personal-main-prefix", "main-prefix", "other-prefix"],
)
async def test_a_move_into_another_git_project_forms_the_permalink_again_and_keeps_the_data(
    host_setup, second_repository, source_permalink
):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    second, second_repo = second_repository
    first_repo = host_setup.repository
    async with Client(mcp) as client:
        await written(client, dated_note(source_permalink))
        await caught_up(host_setup.project.external_id)
        original = first_repo.snapshot().files["a/Jek.md"]
        assert parse_frontmatter(original)["permalink"] == source_permalink  # the premise

        moved = await client.call_tool(
            "bm_move",
            {
                "identifier": "a/Jek.md",
                "destination": "aufgaben/Jek.md",
                "project": "test-project",
                "destination_project": "second",
                "output_format": "json",
            },
        )
        assert not moved.is_error, moved.content
        await caught_up(host_setup.project.external_id, second.external_id)

    # The permalink comes from the new path, as for a note created in the target project.
    new_permalink = "second/aufgaben/jek"
    assert moved.data["previous_permalink"] == source_permalink
    assert moved.data["permalink"] == new_permalink
    copy = second_repo.snapshot().files["aufgaben/Jek.md"]
    after, before = parse_frontmatter(copy), parse_frontmatter(original)
    assert after["permalink"] == new_permalink
    # The data stay the same: both times to the second, every other key, and the body.
    assert (after["created"], after["modified"]) == (CREATED, MODIFIED)
    assert {k: v for k, v in after.items() if k != "permalink"} == {
        k: v for k, v in before.items() if k != "permalink"
    }
    assert remove_frontmatter(copy, strip=False) == remove_frontmatter(original, strip=False)
    assert "a/Jek.md" not in first_repo.snapshot().files
    # SQL agrees with the file: same permalink, same instants.
    _, factory = await db.get_or_create_db(host_setup.config.database_path)
    async with db.scoped_session(factory) as session:
        entity = (
            await session.scalars(select(Entity).where(Entity.file_path == "aufgaben/Jek.md"))
        ).one()
    assert entity.permalink == new_permalink
    assert entity.created_at == datetime.fromisoformat(CREATED)
    assert entity.updated_at == datetime.fromisoformat(MODIFIED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("update_permalinks_on_move", "permalink_after"),
    [(False, "personal/main/aufgaben/jek"), (True, "test-project/b/jek")],
    ids=["setting-off-keeps-it", "setting-on-forms-it-again"],
)
async def test_a_move_inside_a_project_follows_the_setting_and_keeps_the_times(
    host_setup, config_manager, update_permalinks_on_move, permalink_after
):
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    host_setup.config.update_permalinks_on_move = update_permalinks_on_move
    config_manager.save_config(host_setup.config)
    first_repo = host_setup.repository
    async with Client(mcp) as client:
        await written(client, dated_note("personal/main/aufgaben/jek"))
        await caught_up(host_setup.project.external_id)
        moved = await client.call_tool(
            "bm_move",
            {
                "identifier": "a/Jek.md",
                "destination": "b/Jek.md",
                "project": "test-project",
                "output_format": "json",
            },
        )
        assert not moved.is_error, moved.content
        await caught_up(host_setup.project.external_id)

    metadata = parse_frontmatter(first_repo.snapshot().files["b/Jek.md"])
    assert metadata["permalink"] == permalink_after
    assert (metadata["created"], metadata["modified"]) == (CREATED, MODIFIED)
