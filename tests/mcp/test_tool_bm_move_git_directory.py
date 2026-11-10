"""bm_move of a directory on a Git host: one commit, every note listed, refusals by reason.

The Git route itself is exercised in test-int/test_git_directory_move_integration.py. Here
the typed client is replaced, so the tool's choice of route and its wording are pinned
without a Git host.
"""

import json
from typing import Any

import httpx
import pytest
from fastmcp.exceptions import ToolError

from basic_memory.mcp.clients import KnowledgeClient
from basic_memory.mcp.tools import bm_move, bm_write
from basic_memory.runtime.git_routing import local_git_routing
from basic_memory.schemas.v2.entity import GitDirectoryMoveResult

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def answer(source: str = "drafts", destination: str = "archive/drafts") -> GitDirectoryMoveResult:
    return GitDirectoryMoveResult.model_validate(
        {
            "source": source,
            "destination": destination,
            "commit_id": COMMIT,
            "total_notes": 2,
            "moved_notes": [
                {
                    "previous_file_path": f"{source}/a.md",
                    "file_path": f"{destination}/a.md",
                    "external_id": "11111111-1111-1111-1111-111111111111",
                    "permalink": "test-project/drafts/a",
                },
                {
                    "previous_file_path": f"{source}/sub/b.md",
                    "file_path": f"{destination}/sub/b.md",
                    "external_id": "22222222-2222-2222-2222-222222222222",
                    "permalink": "test-project/drafts/sub/b",
                },
            ],
        }
    )


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, str]]:
    """Record the Git route's arguments; the note-by-note route must not run."""
    seen: list[tuple[str, str]] = []

    async def git_move(_self, source: str, destination: str) -> GitDirectoryMoveResult:
        seen.append((source, destination))
        return answer(destination=destination)

    async def upstream(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A Git host must not use the note-by-note folder move")

    monkeypatch.setattr(KnowledgeClient, "move_git_directory", git_move)
    monkeypatch.setattr(KnowledgeClient, "move_directory", upstream)
    return seen


def refused_with(monkeypatch, status: int, detail: Any) -> None:
    async def refused(*_args: Any, **_kwargs: Any) -> None:
        request = httpx.Request("POST", "http://test")
        response = httpx.Response(status, request=request, json={"detail": detail})
        error = ToolError("refused")
        error.__cause__ = httpx.HTTPStatusError("refused", request=request, response=response)
        raise error

    monkeypatch.setattr(KnowledgeClient, "move_git_directory", refused)


@pytest.mark.asyncio
async def test_a_git_host_moves_the_folder_through_one_commit(client, test_project, calls):
    with local_git_routing(True):
        result = await bm_move(
            path="drafts", destination="archive/", project=test_project.name, output_format="json"
        )

    assert calls == [("drafts", "archive/drafts")]
    assert result == {
        "action": "moved",
        "path": "drafts",
        "destination": "archive/drafts",
        "total_files": 2,
        "moved_files": ["archive/drafts/a.md", "archive/drafts/sub/b.md"],
        "failed": [],
        "commit_id": COMMIT,
        "moved_notes": [
            {
                "previous_file_path": "drafts/a.md",
                "file_path": "archive/drafts/a.md",
                "external_id": "11111111-1111-1111-1111-111111111111",
                "permalink": "test-project/drafts/a",
            },
            {
                "previous_file_path": "drafts/sub/b.md",
                "file_path": "archive/drafts/sub/b.md",
                "external_id": "22222222-2222-2222-2222-222222222222",
                "permalink": "test-project/drafts/sub/b",
            },
        ],
    }


@pytest.mark.asyncio
async def test_the_text_form_lists_every_note_on_its_own_line(client, test_project, calls):
    with local_git_routing(True):
        text = await bm_move(path="drafts", destination="archive/drafts", project=test_project.name)

    assert text == (
        f"Moved directory drafts to archive/drafts (2 notes, one commit {COMMIT[:12]})\n"
        "- drafts/a.md -> archive/drafts/a.md "
        "(id 11111111-1111-1111-1111-111111111111, permalink test-project/drafts/a)\n"
        "- drafts/sub/b.md -> archive/drafts/sub/b.md "
        "(id 22222222-2222-2222-2222-222222222222, permalink test-project/drafts/sub/b)"
    )


@pytest.mark.asyncio
async def test_spellings_the_server_resolved_are_reported_next_to_the_request(
    client, test_project, monkeypatch
):
    async def git_move(_self, source: str, destination: str) -> GitDirectoryMoveResult:
        return answer(source="Drafts", destination="Archive/drafts")

    monkeypatch.setattr(KnowledgeClient, "move_git_directory", git_move)
    with local_git_routing(True):
        result = await bm_move(
            path="drafts", destination="archive/", project=test_project.name, output_format="json"
        )
        text = await bm_move(path="drafts", destination="archive/", project=test_project.name)

    assert isinstance(result, dict)
    assert (result["path"], result["requested_path"]) == ("Drafts", "drafts")
    assert (result["destination"], result["requested_destination"]) == (
        "Archive/drafts",
        "archive/drafts",
    )
    assert isinstance(text, str)
    # Both resolutions are named in the text form too, not only in JSON.
    assert text.splitlines()[1:3] == [
        "requested source: drafts",
        "requested destination: archive/drafts",
    ]


@pytest.mark.asyncio
async def test_destination_project_naming_the_same_project_takes_the_git_route(
    client, test_project, calls
):
    with local_git_routing(True):
        await bm_move(
            path="drafts",
            destination="archive",
            destination_project=test_project.name,
            project=test_project.name,
        )

    assert calls == [("drafts", "archive")]


@pytest.mark.asyncio
async def test_a_host_without_git_keeps_the_note_by_note_move(client, test_project, monkeypatch):
    async def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Only a Git host moves a folder as one commit")

    monkeypatch.setattr(KnowledgeClient, "move_git_directory", forbidden)
    await bm_write(
        identifier="drafts/a.md",
        content="Body",
        project=test_project.name,
        allow_duplicate_title=True,
    )

    text = await bm_move(path="drafts", destination="old", project=test_project.name)

    assert text == "Moved directory drafts to old (1 files)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason,code,hint",
    [
        (409, "occupied", "ALREADY_EXISTS", "bm_list(path='archive'"),
        (422, "unpublished_file", "UNSUPPORTED", "move the files that are not notes out of"),
        (409, "unsupported_config", "UNSUPPORTED", "bm_move(identifier=...) for each note"),
        (409, "ambiguous_source", "AMBIGUOUS", "name one folder exactly"),
        (409, "base_advanced", "REVISION_CONFLICT", "repeat the same bm_move"),
        (422, "case_only", "INVALID_ARGUMENTS", "two moves: bm_move(path='drafts'"),
        (422, "invalid_draft", "INVALID_ARGUMENTS", "repair its frontmatter"),
        (409, "not_accepting", "UNAVAILABLE", "bm_project(command='status'"),
        (422, "inside_source", "INVALID_ARGUMENTS", None),
        (422, "same_path", "INVALID_ARGUMENTS", None),
        (422, "project_root", "INVALID_ARGUMENTS", None),
        (422, "invalid_path", "INVALID_ARGUMENTS", None),
    ],
)
async def test_each_refusal_reason_has_its_own_code_and_way_forward(
    client, test_project, monkeypatch, status, reason, code, hint
):
    refused_with(
        monkeypatch,
        status,
        {
            "message": "Server says no",
            "reason": reason,
            "conflicts": [
                {"path": "archive/a.md", "reason": "occupied", "other": "archive/a.md"},
                {"path": "drafts/img.png", "reason": "unpublished_file", "other": None},
            ],
        },
    )
    with local_git_routing(True), pytest.raises(ToolError) as error:
        await bm_move(
            path="drafts", destination="archive", project=test_project.name, output_format="json"
        )

    payload = json.loads(str(error.value))
    assert payload["error"] == code
    assert payload["message"] == "Server says no. Nothing was moved."
    assert payload.get("try_instead") is None if hint is None else hint in payload["try_instead"]
    assert payload["details"] == {
        "reason": reason,
        "conflicts": [
            {"path": "archive/a.md", "reason": "occupied", "other": "archive/a.md"},
            {"path": "drafts/img.png", "reason": "unpublished_file", "other": None},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kinds,wanted,unwanted",
    [
        (["edited_on_disk"], ["undo the edit", "bm_edit/bm_write"], ["not notes out of"]),
        (["unpublished_file"], ["not notes out of"], ["undo the edit"]),
        (["symlink", "edited_on_disk"], ["not notes out of", "undo the edit"], []),
    ],
)
async def test_an_unclean_source_names_the_step_that_clears_each_offender(
    client, test_project, monkeypatch, kinds, wanted, unwanted
):
    """Saving an editor does not clear a note edited on disk: Git mode never imports it."""
    refused_with(
        monkeypatch,
        422,
        {
            "message": "Not clean",
            "reason": "unpublished_file",
            "conflicts": [
                {"path": f"drafts/{index}.md", "reason": kind, "other": None}
                for index, kind in enumerate(kinds)
            ],
        },
    )
    with local_git_routing(True), pytest.raises(ToolError) as error:
        await bm_move(
            path="drafts", destination="archive", project=test_project.name, output_format="json"
        )

    hint = json.loads(str(error.value))["try_instead"]
    assert hint.endswith(", then repeat the same bm_move")
    assert "save or undo editor changes" not in hint
    assert all(part in hint for part in wanted), hint
    assert not any(part in hint for part in unwanted), hint


@pytest.mark.asyncio
async def test_a_missing_folder_may_already_have_moved(client, test_project, monkeypatch):
    """A retry after a lost answer finds the source empty; the hint looks at the destination."""
    refused_with(
        monkeypatch, 404, {"message": "No published notes", "reason": "not_found", "conflicts": []}
    )
    with local_git_routing(True), pytest.raises(ToolError) as error:
        await bm_move(path="notes/drafts", destination="archive", project=test_project.name)

    text = str(error.value)
    assert text.startswith(
        "NOT_FOUND: No published notes under 'notes/drafts' in project 'test-project'. If an "
        "earlier move of it got no answer, it may already have moved. Nothing was moved."
    )
    assert "bm_list(path='archive', project='test-project') to see whether it moved" in text
    assert "bm_list(path='notes', project='test-project') to find the folder" in text


@pytest.mark.asyncio
async def test_a_published_move_whose_index_lags_names_the_commit(
    client, test_project, monkeypatch
):
    refused_with(
        monkeypatch,
        503,
        {
            "message": "Git published; SQL pending. Check status before retry.",
            "published_commit": COMMIT,
        },
    )
    with local_git_routing(True), pytest.raises(ToolError) as error:
        await bm_move(path="drafts", destination="archive", project=test_project.name)

    text = str(error.value)
    assert text.startswith(
        f"OUTCOME_UNKNOWN: The move of 'drafts' was published as commit {COMMIT}"
    )
    assert "do not repeat the move" in text


@pytest.mark.asyncio
async def test_an_unanswered_git_move_is_an_unknown_outcome(client, test_project, monkeypatch):
    async def lost(*_args: Any, **_kwargs: Any) -> None:
        error = ToolError("timed out")
        error.__cause__ = httpx.ReadTimeout("timed out")
        raise error

    monkeypatch.setattr(KnowledgeClient, "move_git_directory", lost)
    with local_git_routing(True), pytest.raises(ToolError) as error:
        await bm_move(path="drafts", destination="archive", project=test_project.name)

    assert str(error.value).startswith("OUTCOME_UNKNOWN: The move of 'drafts' may or may not")
    assert "bm_list(path='archive'" in str(error.value)


@pytest.mark.asyncio
async def test_a_refusal_without_a_reason_is_read_as_a_plain_api_refusal(
    client, test_project, monkeypatch
):
    refused_with(monkeypatch, 404, "Folder moves in one commit need a local Git host")
    with local_git_routing(True), pytest.raises(ToolError) as error:
        await bm_move(path="drafts", destination="archive", project=test_project.name)

    assert str(error.value).startswith(
        "NOT_FOUND: Folder moves in one commit need a local Git host. Nothing was moved."
    )
