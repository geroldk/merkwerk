"""A note keeps its UUID when it moves to another Git project (I1 part B, W2).

The server drafts the note in the target under its old UUID, deletes it in the source,
then commits the draft; an intent next to the target repository finishes the move after
a crash. These tests drive the normal API lifespan with real project repositories, and a
subprocess that exits at each boundary of the move.
"""

import asyncio
import json
import subprocess
import sys
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from basic_memory.api.container import resolve_container
from basic_memory.file_utils import parse_frontmatter
from basic_memory.repository.git_move_intent_repository import GitMoveIntentRepository
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_transaction_repository import GitTransactionRepository
from basic_memory.schemas.git_moves import GitMoveIntent

from tests.git_host_support import add_git_project, host_setup as host_setup, ready

CRASH_EXIT = 83


def canonical(title: str, path: str, extra: str = "") -> str:
    return f"---\ntitle: {title}\ntype: note\npermalink: {path[:-3]}\n{extra}---\n\nBody\n"


def publish_into(
    repository: GitProjectRepository,
    *,
    add: dict[str, tuple[str, UUID]] | None = None,
    moved: dict[UUID, UUID] | None = None,
) -> str:
    """Publish straight into a project repository, as another host or a crashed move did."""
    base = repository.snapshot()
    files, ids = dict(base.files), dict(base.manifest.note_ids)
    for path, (content, note_id) in (add or {}).items():
        files[path], ids[path] = content, note_id
    prepared = repository.prepare(
        transaction_id=uuid4(),
        base_commit=base.commit_id,
        files=files,
        note_ids=ids,
        moved_notes=moved,
    )
    return repository.publish(prepared).commit_id


async def write(
    client: AsyncClient,
    project: str,
    title: str,
    *,
    directory: str = "notes",
    content: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    note: dict[str, Any] = {
        "title": title,
        "directory": directory,
        "content": content if content is not None else f"- [fact] About {title}\n",
    }
    if file_name is not None:
        note["file_name"] = file_name
    response = await client.post(f"/v2/projects/{project}/knowledge/write", json={"note": note})
    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "created", response.text
    return response.json()["entity"]


async def move(client: AsyncClient, source: str, note: str, target: str, path: str, **extra: Any):
    return await client.post(
        f"/v2/projects/{source}/git/moves",
        json={
            "external_id": note,
            "target_project_id": target,
            "destination_path": path,
            **extra,
        },
    )


def commits(repository: GitProjectRepository) -> int:
    return int(repository._git("rev-list", "--count", "main"))


def last_message(repository: GitProjectRepository) -> str:
    return repository._git("log", "-1", "--format=%B").decode()


def intents(repository: GitProjectRepository, project: str) -> list[GitMoveIntent]:
    return GitMoveIntentRepository(repository, UUID(project)).load_all()


def reservations() -> dict[UUID, UUID]:
    host = resolve_container().git_host
    assert host is not None and host.identities is not None
    return dict(host.identities.reservations)


def open_drafts(repository: GitProjectRepository, project: str) -> list[str]:
    records = GitTransactionRepository(repository, UUID(project))
    return [
        str(path.stem)
        for path in records.path.glob("*.json")
        if records.load(UUID(path.stem)).state.kind in {"open", "frozen", "conflicted", "expired"}
    ]


@pytest.fixture
async def two_projects(host_setup, config_manager):
    second, second_repo = await add_git_project(host_setup, config_manager, "second")
    return (
        host_setup.project.external_id,
        host_setup.repository,
        second.external_id,
        second_repo,
    )


# --- One move ---


@pytest.mark.asyncio
async def test_a_moved_note_keeps_its_uuid_and_reads_at_once(host_setup, two_projects):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
            before = commits(a_repo), commits(b_repo)

            answer = await move(client, a, note, b, "archive/Mover.md")
            assert answer.status_code == 200, answer.text
            moved = answer.json()
            # Readable by its UUID in the target as soon as the answer is there.
            read = await client.get(f"/v2/projects/{b}/knowledge/entities/{note}")
            gone = await client.get(f"/v2/projects/{a}/knowledge/entities/{note}")
            listing = (await client.get("/v2/git/moves")).json()
            assert reservations() == {}
    assert moved["state"] == "completed" and not moved["index_pending"]
    assert moved["external_id"] == note
    assert moved["previous_file_path"] == "notes/Mover.md"
    assert moved["file_path"] == "archive/Mover.md"
    assert moved["permalink"] == "second/archive/mover"
    assert read.status_code == 200 and read.json()["file_path"] == "archive/Mover.md"
    assert gone.status_code == 404
    assert listing == {"moves": [], "unreadable": []}
    assert b_repo.snapshot().manifest.note_ids == {"archive/Mover.md": UUID(note)}
    assert UUID(note) not in a_repo.published_note_ids()
    # One commit in each repository; the target's carries BM's move trailer.
    assert (commits(a_repo), commits(b_repo)) == (before[0] + 1, before[1] + 1)
    assert f"Basic-Memory-Moved-Note: {note} from {a}" in last_message(b_repo)
    assert moved["source_commit"] == a_repo.head() and moved["target_commit"] == b_repo.head()
    assert intents(b_repo, b) == [] and open_drafts(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_folder_spelled_like_a_trailer_still_moves(host_setup, two_projects):
    from basic_memory.api.app import app

    a, _, b, b_repo = two_projects
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Spur", directory="Basic-Memory-Transaction"))[
                "external_id"
            ]
            await ready(client, a)
            answer = await move(client, a, note, b, "Basic-Memory-Moved-Note/Spur.md")
    assert answer.status_code == 200, answer.text
    assert answer.json()["state"] == "completed"
    # The commit message quotes the paths; the only trailer is BM's own.
    message = last_message(b_repo)
    assert message.count("Basic-Memory-Moved-Note:") == 1
    assert b_repo.moved_notes_of([b_repo.head()])[b_repo.head()] == {UUID(note): {UUID(a)}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["stale", "locked", "duplicate_title", "occupied", "same_project", "unknown", "not_markdown"],
)
async def test_a_refused_move_leaves_both_projects_as_they_were(host_setup, two_projects, case):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            content = "---\nlocked: true\n---\n\n- [fact] Fixed\n" if case == "locked" else None
            note = (await write(client, a, "Mover", content=content))["external_id"]
            if case == "duplicate_title":
                await write(client, b, "mover", directory="elsewhere")
            if case == "occupied":
                await write(client, b, "Other", directory="archive", file_name="Mover.md")
            await ready(client, a)
            await ready(client, b)
            before = a_repo.head(), b_repo.head()
            extra = {"expected_checksum": "0" * 64} if case == "stale" else {}
            target = a if case == "same_project" else b
            identity = str(uuid4()) if case == "unknown" else note
            path = "archive/Mover.txt" if case == "not_markdown" else "archive/Mover.md"
            answer = await move(client, a, identity, target, path, **extra)
            assert reservations() == {}
    expected = {
        "stale": (409, "revision_conflict"),
        "locked": (423, "locked"),
        "duplicate_title": (409, "duplicate_title"),
        "occupied": (409, "occupied"),
        "same_project": (422, "same_project"),
        "unknown": (404, "not_found"),
        "not_markdown": (422, None),
    }[case]
    assert answer.status_code == expected[0], answer.text
    if expected[1] is not None:
        assert answer.json()["detail"]["kind"] == expected[1]
        assert "Nothing was moved" in answer.json()["detail"]["message"]
    assert (a_repo.head(), b_repo.head()) == before
    assert UUID(note) in a_repo.published_note_ids()
    assert intents(b_repo, b) == [] and open_drafts(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_host_without_git_refuses_the_route_cleanly(host_setup, config_manager):
    from basic_memory.api.app import app

    host_setup.config.git_repositories = {}
    config_manager.save_config(host_setup.config)
    project = host_setup.project.external_id
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            answer = await move(client, project, str(uuid4()), str(uuid4()), "a.md")
            listing = await client.get("/v2/git/moves")
    assert answer.status_code == 404, answer.text
    assert answer.json()["detail"]["kind"] == "not_git_host"
    assert listing.status_code == 404


# --- The target is held from the draft to the commit ---


@pytest.mark.asyncio
async def test_a_write_to_the_target_waits_for_the_move_and_then_finds_it(
    host_setup, two_projects, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.services.note_content_writes import NoteContentMutationService

    a, _, b, b_repo = two_projects
    entered, release = asyncio.Event(), asyncio.Event()
    delete = NoteContentMutationService.delete_note

    async def paused_delete(self, **arguments):
        entered.set()
        await release.wait()
        return await delete(self, **arguments)

    monkeypatch.setattr(NoteContentMutationService, "delete_note", paused_delete)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
            moving = asyncio.create_task(move(client, a, note, b, "archive/Mover.md"))
            await asyncio.wait_for(entered.wait(), 10)
            # The draft exists; a write of the same path and title must not slip in.
            competing = asyncio.create_task(
                client.post(
                    f"/v2/projects/{b}/knowledge/write",
                    json={"note": {"title": "Mover", "directory": "archive", "content": "x\n"}},
                )
            )
            await asyncio.sleep(0.3)
            assert not competing.done()
            release.set()
            moved, written = await moving, await competing
    assert moved.status_code == 200 and moved.json()["state"] == "completed", moved.text
    # The write ran after the commit and found the moved note there.
    assert written.status_code == 200, written.text
    assert written.json()["kind"] in {"already_exists", "duplicate_title"}
    assert b_repo.snapshot().manifest.note_ids == {"archive/Mover.md": UUID(note)}


# --- Deferred projects ---


@pytest.mark.asyncio
async def test_a_move_into_a_waiting_target_completes_with_the_index_pending(
    host_setup, config_manager, two_projects
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    third, _ = await add_git_project(host_setup, config_manager, "third")
    c = third.external_id
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            held = (await write(client, c, "Held"))["external_id"]
            await ready(client, a)
            await ready(client, c)
            # The target publishes a note the third project still holds: it waits for it.
            publish_into(
                b_repo,
                add={"held.md": (canonical("Held", "held.md"), UUID(held))},
                moved={UUID(held): UUID(c)},
            )
            answer = await move(client, a, note, b, "archive/Mover.md")
            read = await client.get(f"/v2/projects/{b}/knowledge/entities/{note}")
    assert answer.status_code == 200, answer.text
    assert answer.json()["state"] == "completed"
    assert answer.json()["index_pending"] is True
    assert "waits" in answer.json()["message"]
    assert read.status_code == 200, read.text
    assert UUID(note) in b_repo.published_note_ids()
    assert UUID(note) not in a_repo.published_note_ids()


@pytest.mark.asyncio
async def test_a_move_out_of_a_waiting_source_is_refused_as_a_wait(
    host_setup, config_manager, two_projects
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    third, _ = await add_git_project(host_setup, config_manager, "third")
    c = third.external_id
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            held = (await write(client, c, "Held"))["external_id"]
            await ready(client, a)
            await ready(client, c)
            publish_into(
                a_repo,
                add={"held.md": (canonical("Held", "held.md"), UUID(held))},
                moved={UUID(held): UUID(c)},
            )
            before = a_repo.head(), b_repo.head()
            answer = await move(client, a, note, b, "archive/Mover.md")
            assert reservations() == {}
    # A wait, not "Git published; SQL pending" (503): 409 with kind deferred.
    assert answer.status_code == 409, answer.text
    assert answer.json()["detail"]["kind"] == "deferred"
    assert held in answer.json()["detail"]["waiting_note_ids"]
    assert (a_repo.head(), b_repo.head()) == before
    assert intents(b_repo, b) == [] and open_drafts(b_repo, b) == []


# --- Crashes: the next start finishes, undoes or keeps the move ---


async def crash(tmp_path, fault: str, a: str, a_repo, b: str, b_repo, note: str, path: str) -> None:
    request = tmp_path / f"crash-{fault}.json"
    request.write_text(
        json.dumps(
            {
                "fault": fault,
                "source_repository": str(a_repo.path),
                "target_repository": str(b_repo.path),
                "source_project_id": a,
                "body": {"external_id": note, "target_project_id": b, "destination_path": path},
            }
        )
    )
    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "test-int/fixtures/git_cross_move_crash.py", str(request)],
        capture_output=True,
        timeout=90,
    )
    assert child.returncode == CRASH_EXIT, child.stderr[-4000:]


async def seeded(app, a: str, b: str) -> str:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
    return note


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "arrives"),
    [
        ("after_draft_before_intent", False),
        ("after_intent_before_delete", False),
        ("after_source_git_before_sql", True),
        ("after_delete_before_commit", True),
        ("after_target_publish_before_record", True),
        ("after_record_before_intent_removed", True),
    ],
)
async def test_a_crashed_move_finishes_or_undoes_at_the_next_start(
    host_setup, two_projects, tmp_path, fault, arrives
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, fault, a, a_repo, b, b_repo, note, "moved/Mover.md")
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, a)
            await ready(client, b)
            listing = (await client.get("/v2/git/moves")).json()
            in_a = await client.get(f"/v2/projects/{a}/knowledge/entities/{note}")
            in_b = await client.get(f"/v2/projects/{b}/knowledge/entities/{note}")
            assert reservations() == {}
    assert listing["moves"] == []
    assert intents(b_repo, b) == []
    if arrives:
        assert in_b.status_code == 200 and in_a.status_code == 404
        assert b_repo.snapshot().manifest.note_ids == {"moved/Mover.md": UUID(note)}
        assert UUID(note) not in a_repo.published_note_ids()
        assert f"Basic-Memory-Moved-Note: {note} from {a}" in last_message(b_repo)
    else:
        assert in_a.status_code == 200 and in_b.status_code == 404
        assert a_repo.snapshot().manifest.note_ids == {"notes/Mover.md": UUID(note)}
        assert b_repo.published_note_ids() == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", ["expired", "cleaned"])
async def test_a_resumed_move_rebuilds_an_expired_or_cleaned_draft_from_its_intent(
    host_setup, config_manager, two_projects, tmp_path, draft
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    # The lease is fixed when the draft starts: the crashed host already ran with it.
    host_setup.config.git_transaction_idle_seconds = 1
    if draft == "cleaned":
        host_setup.config.git_transaction_retention_seconds = 0
    config_manager.save_config(host_setup.config)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    (intent,) = intents(b_repo, b)
    await asyncio.sleep(1.2)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, b)
            read = await client.get(f"/v2/projects/{b}/knowledge/entities/{note}")
    assert read.status_code == 200, read.text
    assert b_repo.snapshot().files["m/Mover.md"] == intent.target_content
    # A new transaction published it; the old draft never did.
    message = last_message(b_repo)
    assert f"Basic-Memory-Transaction: {intent.transaction_id}" not in message
    old = GitTransactionRepository(b_repo, UUID(b)).load(intent.transaction_id)
    assert old.state.kind in {"rolled_back", "expired_cleaned"}
    assert intents(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_permalink_taken_meanwhile_gets_the_suffix_and_the_new_draft_is_saved_first(
    host_setup, two_projects, tmp_path, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.services import git_cross_project_move

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    (planned,) = intents(b_repo, b)
    # Another host published a note with the planned permalink while this one was down.
    taken = f"---\ntitle: Andere\ntype: note\npermalink: {planned.target_permalink}\n---\n\nX\n"
    publish_into(b_repo, add={"other.md": (taken, uuid4())})
    drop = git_cross_project_move._drop_draft

    async def crash_before_old_draft_goes(target, transaction_id, key):
        if transaction_id == planned.transaction_id:
            # The rebuilt intent must already be on disk now.
            (saved,) = intents(b_repo, b)
            assert saved.transaction_id != planned.transaction_id
            raise RuntimeError("simulated crash between the new pointer and the old draft")
        await drop(target, transaction_id, key)

    monkeypatch.setattr(git_cross_project_move, "_drop_draft", crash_before_old_draft_goes)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, b)
            # The start-up resume stopped at the simulated crash; the intent stays open.
            (open_move,) = (await client.get("/v2/git/moves")).json()["moves"]
            assert open_move["transaction_id"] != str(planned.transaction_id)
            assert "target_content" not in open_move and "transaction_key" not in open_move
            monkeypatch.setattr(git_cross_project_move, "_drop_draft", drop)
            resumed = (await client.post("/v2/git/moves/resume")).json()
            await ready(client, b)
    assert [outcome["outcome"] for outcome in resumed["outcomes"]] == ["completed"]
    published = parse_frontmatter(b_repo.snapshot().files["m/Mover.md"])
    assert published["permalink"] == f"{planned.target_permalink}-1"
    assert b_repo.snapshot().manifest.note_ids["m/Mover.md"] == UUID(note)
    assert intents(b_repo, b) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["target", "source"])
async def test_a_taken_path_makes_the_move_stuck_visible_and_resolvable(
    host_setup, two_projects, tmp_path, choice
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    original = a_repo.snapshot().files["notes/Mover.md"]
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    publish_into(b_repo, add={"m/Mover.md": (canonical("Besetzt", "m/Mover.md"), uuid4())})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            # The host starts; the move is listed as stuck, its UUID stays reserved.
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            taken = await client.put(
                f"/v2/projects/{a}/knowledge/entities/{note}",
                json={"title": "Mover", "directory": "notes", "content": "Neu\n"},
            )
            assert reservations() == {UUID(note): UUID(stuck["move_id"])}
            body = (
                {"choice": "target", "destination_path": "m/Mover-2.md"}
                if choice == "target"
                else {"choice": "source"}
            )
            resolved = await client.post(f"/v2/git/moves/{stuck['move_id']}/resolve", json=body)
            await ready(client, a)
            await ready(client, b)
            after = (await client.get("/v2/git/moves")).json()
            assert reservations() == {}
    assert stuck["state"] == "stuck" and "m/Mover.md" in stuck["stuck_reason"]
    assert taken.status_code == 409 and taken.json()["detail"]["kind"] == "owned_elsewhere"
    assert resolved.status_code == 200, resolved.text
    assert after["moves"] == []
    if choice == "target":
        assert resolved.json()["state"] == "completed"
        assert b_repo.snapshot().manifest.note_ids["m/Mover-2.md"] == UUID(note)
        assert UUID(note) not in a_repo.published_note_ids()
    else:
        assert resolved.json()["state"] == "returned"
        assert a_repo.snapshot().manifest.note_ids == {"notes/Mover.md": UUID(note)}
        assert a_repo.snapshot().files["notes/Mover.md"] == original
        assert UUID(note) not in b_repo.published_note_ids()


@pytest.mark.asyncio
async def test_a_uuid_back_in_the_source_keeps_the_moved_bytes(host_setup, two_projects, tmp_path):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    (planned,) = intents(b_repo, b)
    # Between the delete and the commit the UUID came back into the source (here: straight
    # into Git, as a peer could). Resume must not drop the moved bytes as "never moved".
    publish_into(a_repo, add={"notes/Mover.md": (canonical("Mover", "notes/mover.md"), UUID(note))})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            resumed = (await client.post("/v2/git/moves/resume")).json()
    assert stuck["state"] == "stuck" and "in the source project again" in stuck["stuck_reason"]
    assert [outcome["outcome"] for outcome in resumed["outcomes"]] == ["stuck"]
    (kept,) = intents(b_repo, b)
    assert kept.target_content == planned.target_content
    assert UUID(note) not in b_repo.published_note_ids()
