"""bm_move into another project on a Git host: the server's answers, worded (I1).

The server's route is replaced by canned answers here; the real route runs in
test-int/test_git_cross_project_move_w2_integration.py.
"""

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest

from basic_memory.mcp.cross_project_move import _move_keeping_id
from basic_memory.mcp.note_targets import ProjectTarget
from basic_memory.mcp.refusals import Refusal
from basic_memory.schemas.project_info import ProjectItem
from basic_memory.mcp.project_context import ProjectPathRoute

SOURCE, TARGET, NOTE = str(uuid4()), str(uuid4()), str(uuid4())


def target(client: httpx.AsyncClient, external_id: str, name: str) -> ProjectTarget:
    return ProjectTarget(
        client=client,
        project=ProjectItem(id=1, external_id=external_id, name=name, path=f"/{name}"),
        route=ProjectPathRoute(project=name, path="", stripped=False),
        project_ref=name,
    )


async def move_answered(
    status: int, body: dict[str, Any], *, destination: str = "neu/a.md"
) -> Refusal:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v2/projects/{SOURCE}/git/moves"
        return httpx.Response(status, json=body)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(answer), base_url="http://test"
    ) as client:
        with pytest.raises(Refusal) as refused:
            await _move_keeping_id(
                target(client, SOURCE, "quelle"),
                target(client, TARGET, "ziel"),
                NOTE,
                "notes/a.md",
                destination,
                None,
                "json",
                catch_up=True,
            )
    return refused.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "kind", "code"),
    [
        (404, "not_found", "NOT_FOUND"),
        (423, "locked", "LOCKED"),
        (422, "invalid_path", "INVALID_ARGUMENTS"),
        (409, "occupied", "ALREADY_EXISTS"),
        (409, "already_in_target", "ALREADY_EXISTS"),
        (409, "deferred", "UNAVAILABLE"),
        (409, "reserved", "UNAVAILABLE"),
        (503, "source_failed", "UNAVAILABLE"),
    ],
)
async def test_each_refusal_kind_gets_its_code(status, kind, code):
    refused = await move_answered(
        status, {"detail": {"kind": kind, "message": f"{kind}. Nothing was moved."}}
    )
    assert refused.code == code
    assert "Nothing was moved" in refused.message


@pytest.mark.asyncio
async def test_a_stale_revision_and_a_title_twin_are_worded_as_for_any_move():
    stale = await move_answered(
        409, {"detail": {"kind": "revision_conflict", "message": "x", "db_checksum": "abc"}}
    )
    assert stale.code == "REVISION_CONFLICT" and "abc" in stale.message
    twin = await move_answered(
        409,
        {
            "detail": {
                "kind": "duplicate_title",
                "message": "x",
                "title": "A",
                "file_path": "alt/a.md",
            }
        },
    )
    assert twin.code == "ALREADY_EXISTS" and "alt/a.md" in twin.message


@pytest.mark.asyncio
async def test_an_unknown_answer_falls_back_to_the_general_wording():
    plain = await move_answered(409, {"detail": "Note already exists."})
    assert plain.code == "ALREADY_EXISTS"
    unknown = await move_answered(500, {"detail": "boom"})
    assert unknown.code == "OUTCOME_UNKNOWN"


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "stuck"])
async def test_a_move_kept_open_is_a_partial_failure_naming_it(state):
    move_id = str(uuid4())
    refused = await move_answered(
        200,
        {
            "state": state,
            "move_id": move_id,
            "external_id": NOTE,
            "source_project_id": SOURCE,
            "target_project_id": TARGET,
            "previous_file_path": "notes/a.md",
            "file_path": "neu/a.md",
            "previous_permalink": "quelle/notes/a",
            "permalink": "ziel/neu/a",
            "message": "waiting",
        },
    )
    assert refused.code == "PARTIAL_FAILURE"
    assert refused.details is not None
    assert refused.details["move_id"] == move_id and refused.details["state"] == state
    assert refused.details["external_id"] == NOTE
    rendered = json.loads(str(refused))
    assert ("resolve" in rendered["message"]) is (state == "stuck")


@pytest.mark.asyncio
async def test_a_destination_that_is_no_note_path_is_refused_before_the_call():
    refused = await move_answered(200, {}, destination="neu/../a.md")
    assert refused.code == "INVALID_ARGUMENTS"
