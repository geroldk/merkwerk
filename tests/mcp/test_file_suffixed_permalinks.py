"""Notes whose permalink ends in a file suffix (`threads/fix-in-tax-costs.py`).

`Fix in tax_costs.py.md` stores `<project>/threads/fix-in-tax-costs.py`. A move leaves
that permalink while the file lives elsewhere, so only the permalink names the note,
and routing strips the project prefix before the resolver sees it. These run the real
ASGI stack through the shared `client` fixture.
"""

from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from basic_memory.index.local_project import LocalProjectIndexRunner
from basic_memory.mcp.tools import bm_delete, bm_edit, bm_move, bm_read, bm_search, bm_write
from basic_memory.workspace_context import workspace_permalink_context


async def write_moved_note(project: str, permalink: str, suffix: str = "py") -> dict[str, Any]:
    """A note filed under archive/ whose frontmatter keeps its suffixed permalink."""
    result = await bm_write(
        identifier=f"archive/threads/Fix in tax_costs.{suffix}.md",
        content=f"---\npermalink: {permalink}\n---\n# Fix\n\nThe fix in tax costs",
        project=project,
        output_format="json",
    )
    assert isinstance(result, dict)
    assert result["permalink"] == permalink
    return result


async def read_json(identifier: str, project: str) -> dict[str, Any]:
    result = await bm_read(identifier, project=project, output_format="json")
    assert isinstance(result, dict)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["py", "txt"])
async def test_read_by_permalink_with_and_without_the_project_prefix(client, test_project, suffix):
    project = test_project.name
    below = f"threads/fix-in-tax-costs.{suffix}"
    permalink = f"{test_project.permalink}/{below}"
    await write_moved_note(project, permalink, suffix)

    for identifier in (permalink, below, f"memory://{permalink}", f"memory://{below}"):
        payload = await read_json(identifier, project)
        assert payload["permalink"] == permalink, identifier
        assert payload["content"].rstrip().endswith("The fix in tax costs"), identifier


@pytest.mark.asyncio
async def test_search_one_note_by_its_permalink(client, test_project):
    project = test_project.name
    permalink = f"{test_project.permalink}/threads/fix-in-tax-costs.py"
    await write_moved_note(project, permalink)

    for identifier in (permalink, "threads/fix-in-tax-costs.py"):
        row = await bm_search(
            query="tax", identifier=identifier, project=project, output_format="json"
        )
        assert isinstance(row, dict)
        assert row["file_path"].endswith("archive/threads/Fix in tax_costs.py.md"), identifier
        assert row["match_count"] >= 1


@pytest.mark.asyncio
async def test_edit_move_and_delete_name_the_note_by_its_permalink(client, test_project):
    project = test_project.name
    permalink = f"{test_project.permalink}/threads/fix-in-tax-costs.py"
    await write_moved_note(project, permalink)

    edited = await bm_edit(
        identifier="threads/fix-in-tax-costs.py",
        command="append",
        content="Appended by permalink",
        project=project,
        output_format="json",
    )
    assert isinstance(edited, dict)
    assert edited["file_path"] == "archive/threads/Fix in tax_costs.py.md"

    moved = await bm_move(
        identifier=f"memory://{permalink}",
        destination="elsewhere/Other name.py.md",
        project=project,
        output_format="json",
    )
    assert isinstance(moved, dict)
    # A project-qualified identifier gets project-qualified paths back.
    assert moved["previous_file_path"].endswith("archive/threads/Fix in tax_costs.py.md")
    assert moved["file_path"].endswith("elsewhere/Other name.py.md")
    after_move = await read_json("elsewhere/Other name.py.md", project)
    assert "Appended by permalink" in after_move["content"]

    deleted = await bm_delete(
        identifier=after_move["permalink"], project=project, output_format="json"
    )
    assert isinstance(deleted, dict)
    assert deleted["file_path"].endswith("elsewhere/Other name.py.md")
    with pytest.raises(ToolError, match="^NOT_FOUND"):
        await bm_read("elsewhere/Other name.py.md", project=project)


@pytest.mark.asyncio
async def test_a_file_path_that_mirrors_the_permalink_still_reads(client, test_project):
    project = test_project.name
    written = await bm_write(
        identifier="mirror/korpus-zeilen.py.md",
        content="# Mirror\n\nmirrored",
        project=project,
        output_format="json",
    )
    assert isinstance(written, dict)
    assert written["permalink"] == f"{test_project.permalink}/mirror/korpus-zeilen.py"

    for identifier in (written["permalink"], "mirror/korpus-zeilen.py"):
        payload = await read_json(identifier, project)
        assert payload["file_path"].endswith("mirror/korpus-zeilen.py.md"), identifier


@pytest.mark.asyncio
async def test_a_resource_keeps_its_path_beside_a_note_spelled_the_same(
    client, test_project, test_files, project_repository, session_maker
):
    """`side/foo.txt` reads the text file; the note `side/foo.txt.md` reads by its path (#1629)."""
    folder = test_files["image"].parent / "side"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "foo.txt").write_text("RESOURCE BYTES\n")
    (folder / "foo.txt.md").write_text("# Note\n\nNOTE BODY\n")
    runner = LocalProjectIndexRunner(
        project_repository=project_repository, session_maker=session_maker
    )
    await runner.index_project(test_project.id, force_full=True)
    project = test_project.name

    for identifier in ("side/foo.txt", f"{test_project.permalink}/side/foo.txt"):
        assert "RESOURCE BYTES" in str(await bm_read(identifier, project=project)), identifier
    assert "NOTE BODY" in str(await bm_read("side/foo.txt.md", project=project))


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["edit", "delete"])
async def test_another_spelling_of_a_resource_path_changes_no_note(
    client, test_project, test_files, project_repository, session_maker, tool
):
    """`notes/Foo.txt` stays NOT_FOUND although a moved note's permalink is `<p>/notes/foo.txt`."""
    root = test_files["image"].parent
    (root / "notes").mkdir(parents=True, exist_ok=True)
    (root / "notes" / "foo.txt").write_text("RESOURCE BYTES\n")
    (root / "archive").mkdir(parents=True, exist_ok=True)
    note_file = root / "archive" / "Foo note.md"
    note_file.write_text(
        f"---\ntitle: Foo note\npermalink: {test_project.permalink}/notes/foo.txt\n---\n\n"
        "NOTE BODY\n"
    )
    await LocalProjectIndexRunner(
        project_repository=project_repository, session_maker=session_maker
    ).index_project(test_project.id, force_full=True)
    project = test_project.name

    with pytest.raises(ToolError, match="^NOT_FOUND"):
        if tool == "edit":
            await bm_edit(
                identifier="notes/Foo.txt", command="append", content="EDITED", project=project
            )
        else:
            await bm_delete(identifier="notes/Foo.txt", project=project)

    assert note_file.read_text().rstrip().endswith("NOTE BODY")
    assert (root / "notes" / "foo.txt").read_text() == "RESOURCE BYTES\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["plain", "qualified", "memory"])
@pytest.mark.parametrize("tool", ["edit", "delete", "move"])
async def test_the_permalink_owner_beats_a_forgiving_file_spelling(
    client, test_project, tool, spelling
):
    """`alias/fix-me.py` is the moved note's permalink; `alias/Fix_Me.py.md` only resembles it.

    Before suffixed permalinks were routed, all three tools changed the resembling note.
    """
    project = test_project.name
    permalink = f"{test_project.permalink}/alias/fix-me.py"
    await bm_write(
        identifier="archive/moved.md",
        content=f"---\npermalink: {permalink}\n---\n# Moved\n\nBODY-A",
        project=project,
    )
    await bm_write(
        identifier="alias/Fix_Me.py.md",
        content=f"---\npermalink: {test_project.permalink}/alias/fix-me-b.py\n---\n# B\n\nBODY-B",
        project=project,
    )
    identifier = {
        "plain": "alias/fix-me.py",
        "qualified": permalink,
        "memory": f"memory://{permalink}",
    }[spelling]

    if tool == "edit":
        result = await bm_edit(
            identifier=identifier,
            command="append",
            content="EDITED",
            project=project,
            output_format="json",
        )
        touched = result["file_path"] if isinstance(result, dict) else None
    elif tool == "delete":
        result = await bm_delete(identifier=identifier, project=project, output_format="json")
        touched = result["file_path"] if isinstance(result, dict) else None
    else:
        result = await bm_move(
            identifier=identifier,
            destination="dest/moved-here.md",
            project=project,
            output_format="json",
        )
        touched = result["previous_file_path"] if isinstance(result, dict) else None

    assert touched is not None and touched.endswith("archive/moved.md"), touched
    untouched = await read_json("alias/Fix_Me.py.md", project)
    assert untouched["content"].rstrip().endswith("BODY-B")


@pytest.mark.asyncio
async def test_workspace_permalink_reads_without_its_prefixes(client, test_project):
    project = test_project.name
    permalink = f"team-paul/{test_project.permalink}/threads/fix-in-tax-costs.py"

    with workspace_permalink_context(workspace_slug="team-paul", workspace_type="organization"):
        await write_moved_note(project, permalink)
        for identifier in (
            permalink,
            "threads/fix-in-tax-costs.py",
            f"memory://{permalink}",
        ):
            payload = await read_json(identifier, project)
            assert payload["permalink"] == permalink, identifier
