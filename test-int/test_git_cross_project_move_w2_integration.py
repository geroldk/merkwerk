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
    if fault == "after_draft_before_intent":
        # The boundary was reached: a draft holding the note, and no intent for it.
        (orphan,) = open_drafts(b_repo, b)
        assert GitTransactionRepository(b_repo, UUID(b)).load(UUID(orphan)).revision == 1
        assert intents(b_repo, b) == []
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


@pytest.mark.asyncio
async def test_resolve_refuses_what_would_duplicate_and_returns_where_it_can(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    publish_into(b_repo, add={"m/Mover.md": (canonical("Besetzt", "m/Mover.md"), uuid4())})
    # The source's old path is taken by another note meanwhile, with the permalink an
    # ordinary create there gets (the moved note's old one).
    taken = canonical("Andere", "test-project/notes/mover.md")
    publish_into(a_repo, add={"notes/Mover.md": (taken, uuid4())})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            resolve = f"/v2/git/moves/{stuck['move_id']}/resolve"
            occupied = await client.post(resolve, json={"choice": "source"})
            elsewhere = await client.post(
                resolve, json={"choice": "source", "destination_path": "zurueck/Mover.md"}
            )
            unknown = await client.post(
                f"/v2/git/moves/{uuid4()}/resolve", json={"choice": "source"}
            )
            await ready(client, a)
            read = await client.get(f"/v2/projects/{a}/knowledge/entities/{note}")
    assert occupied.status_code == 409 and occupied.json()["detail"]["kind"] == "occupied"
    assert elsewhere.status_code == 200, elsewhere.text
    assert elsewhere.json()["state"] == "returned"
    assert elsewhere.json()["file_path"] == "zurueck/Mover.md"
    assert unknown.status_code == 404
    assert read.status_code == 200 and read.json()["file_path"] == "zurueck/Mover.md"
    # The returned note's permalink is formed again for its new path.
    returned = parse_frontmatter(a_repo.snapshot().files["zurueck/Mover.md"])
    assert returned["permalink"] == "test-project/zurueck/mover"
    assert intents(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_uuid_back_in_the_source_can_only_be_resolved_there(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    publish_into(a_repo, add={"notes/Mover.md": (canonical("Mover", "notes/mover.md"), UUID(note))})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            resolve = f"/v2/git/moves/{stuck['move_id']}/resolve"
            to_target = await client.post(resolve, json={"choice": "target"})
            to_source = await client.post(resolve, json={"choice": "source"})
            assert reservations() == {}
    assert to_target.status_code == 409 and to_target.json()["detail"]["kind"] == "in_source"
    assert to_source.status_code == 200 and to_source.json()["state"] == "returned"
    assert "stay in its history" in to_source.json()["message"]
    assert intents(b_repo, b) == []
    assert a_repo.snapshot().manifest.note_ids == {"notes/Mover.md": UUID(note)}


@pytest.mark.asyncio
async def test_an_intent_whose_source_left_the_host_stays_stuck_and_goes_to_the_target(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    (planned,) = intents(b_repo, b)
    store = GitMoveIntentRepository(b_repo, UUID(b))
    store.save(planned.model_copy(update={"source_project_id": uuid4()}))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            resolve = f"/v2/git/moves/{stuck['move_id']}/resolve"
            to_source = await client.post(resolve, json={"choice": "source"})
            to_target = await client.post(resolve, json={"choice": "target"})
    assert "not on this host" in stuck["stuck_reason"]
    assert to_source.status_code == 409 and to_source.json()["detail"]["kind"] == "no_source"
    assert to_target.status_code == 200 and to_target.json()["state"] == "completed"
    assert b_repo.snapshot().manifest.note_ids == {"m/Mover.md": UUID(note)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "fails",
        "published_sql_pending",
        "published_names_its_commit",
        "published_catch_up_fails",
        "says_nothing",
    ],
)
async def test_the_source_delete_is_judged_from_its_history_not_its_answer(
    host_setup, two_projects, monkeypatch, answer
):
    from basic_memory.api.app import app
    from basic_memory.services.note_content_writes import (
        NoteContentMutationService,
        NoteContentMutationServiceError,
    )

    a, a_repo, b, b_repo = two_projects
    delete = NoteContentMutationService.delete_note

    async def failing_catch_up(self):
        raise RuntimeError("SQL is busy")

    async def answered(self, **arguments):
        if answer == "fails":
            raise RuntimeError("connection reset")
        if answer == "says_nothing":
            return None
        await delete(self, **arguments)
        detail = {"message": "Git published; SQL pending. Check status before retry."}
        if answer == "published_names_its_commit":
            detail["published_commit"] = a_repo.head()
        raise NoteContentMutationServiceError(503, detail)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
            monkeypatch.setattr(NoteContentMutationService, "delete_note", answered)
            if answer == "published_catch_up_fails":
                from basic_memory.services.git_project_runtime import GitProjectRuntime

                # The delete's own SQL commit already let go; a failed catch-up is harmless.
                monkeypatch.setattr(GitProjectRuntime, "catch_up", failing_catch_up)
            moved = await move(client, a, note, b, "archive/Mover.md")
    if answer.startswith("published"):
        # The delete published: the move goes on and completes.
        assert moved.status_code == 200 and moved.json()["state"] == "completed", moved.text
        assert UUID(note) in b_repo.published_note_ids()
    else:
        # Nothing was deleted: nothing was moved, whatever the answer said.
        assert moved.status_code == (503 if answer == "fails" else 409), moved.text
        assert "Nothing was moved" in moved.json()["detail"]["message"]
        assert UUID(note) in a_repo.published_note_ids()
        assert open_drafts(b_repo, b) == []
    assert intents(b_repo, b) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("meanwhile", ["delete", "edit_then_delete"])
async def test_a_delete_by_someone_else_before_the_moves_own_is_not_undone(
    host_setup, two_projects, monkeypatch, meanwhile
):
    """The source gate is not held from the checks to the delete: another client may
    delete the note (or edit and delete it) meanwhile. That removal is not the move's;
    the move must not publish the note it read into the target."""
    from basic_memory.api.app import app
    from basic_memory.services import git_cross_project_move

    a, a_repo, b, b_repo = two_projects
    canonical_step = git_cross_project_move._canonical
    entity = f"/v2/projects/{a}/knowledge/entities"

    async def someone_else_first(runtime, **arguments):
        if meanwhile == "edit_then_delete":
            edited = await client.put(
                f"{entity}/{note}",
                json={"title": "Mover", "directory": "notes", "content": "Edited\n"},
            )
            assert edited.status_code in {200, 202}, edited.text
        deleted = await client.delete(f"{entity}/{note}")
        assert deleted.status_code in {200, 202}, deleted.text
        return await canonical_step(runtime, **arguments)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
            monkeypatch.setattr(git_cross_project_move, "_canonical", someone_else_first)
            answer = await move(client, a, note, b, "archive/Mover.md")
            monkeypatch.setattr(git_cross_project_move, "_canonical", canonical_step)
            await ready(client, a)
            await ready(client, b)
            assert reservations() == {}
    assert answer.status_code in {404, 409}, answer.text
    assert "Nothing was moved" in answer.json()["detail"]["message"]
    assert UUID(note) not in a_repo.published_note_ids()
    assert UUID(note) not in b_repo.published_note_ids()
    assert intents(b_repo, b) == [] and open_drafts(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_resumed_move_does_not_take_someone_elses_removal_for_its_own(
    host_setup, two_projects, tmp_path
):
    """A drafted intent after a crash; meanwhile the note was edited and deleted in the
    source. The move's guarded delete could not have removed those bytes: abandoned."""
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_intent_before_delete", a, a_repo, b, b_repo, note, "m/Mover.md")
    (path,) = a_repo.snapshot().files
    publish_into(a_repo, add={path: (canonical("Mover", path, "tags: [edited]\n"), UUID(note))})
    base = a_repo.snapshot()
    prepared = a_repo.prepare(
        transaction_id=uuid4(), base_commit=base.commit_id, files={}, note_ids={}
    )
    a_repo.publish(prepared)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, a)
            await ready(client, b)
            listing = (await client.get("/v2/git/moves")).json()
            assert reservations() == {}
    assert listing["moves"] == []
    assert UUID(note) not in b_repo.published_note_ids()
    assert UUID(note) not in a_repo.published_note_ids()
    assert intents(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_title_a_waiting_target_holds_in_its_head_is_refused(
    host_setup, config_manager, two_projects
):
    """The target waits for a note (not yet in its SQL); a move must still see its title."""
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
                b_repo,
                add={"held.md": (canonical("mover", "held.md"), UUID(held))},
                moved={UUID(held): UUID(c)},
            )
            before = a_repo.head(), b_repo.head()
            answer = await move(client, a, note, b, "archive/Mover.md")
            assert reservations() == {}
    assert answer.status_code == 409, answer.text
    assert answer.json()["detail"]["kind"] == "duplicate_title"
    assert answer.json()["detail"]["file_path"] == "held.md"
    assert (a_repo.head(), b_repo.head()) == before


# --- Edges: every refusal before the delete, every outcome after it ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "unknown_target",
        "target_not_accepting",
        "draft_refused",
        "on_disk",
        "unnormalized",
        "refused_plainly",
        "source_unsettled",
    ],
)
async def test_more_refusals_move_nothing(host_setup, two_projects, monkeypatch, case):
    from basic_memory.api.app import app
    from basic_memory.index.git_projection_worker import GitProjectionWorker
    from basic_memory.repository.git_projection_repository import GitProjectionConflict
    from basic_memory.repository.git_transaction_repository import GitTransactionError
    from basic_memory.services.git_project_runtime import GitProjectRuntime
    from basic_memory.services.git_transactions import GitTransactionService
    from basic_memory.services.note_content_writes import (
        NoteContentMutationService,
        NoteContentMutationServiceError,
    )

    a, a_repo, b, b_repo = two_projects
    if case == "source_unsettled":

        async def paused(self, *, vectors):
            await asyncio.Event().wait()

        monkeypatch.setattr(GitProjectionWorker, "_loop", paused)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            if case != "source_unsettled":
                await ready(client, a)
                await ready(client, b)
            target, path = b, "archive/Mover.md"
            if case == "unknown_target":
                target = str(uuid4())
            if case == "target_not_accepting":

                async def refusing(self, **_):
                    raise GitProjectionConflict("Target projection failed")

                monkeypatch.setattr(GitProjectRuntime, "catch_up_admitted", refusing)
            if case == "draft_refused":

                async def invalid(self, *arguments, **options):
                    raise GitTransactionError(422, "Invalid draft note archive/Mover.md")

                monkeypatch.setattr(GitTransactionService, "write_identified_admitted", invalid)
            if case == "on_disk":
                root = b_repo.path.parent / "second"
                (root / "archive").mkdir(parents=True)
                (root / "archive" / "Mover.md").write_text("A file nobody published\n")
            if case == "unnormalized":
                path = "frage?/Mover.md"  # a folder name keeps no "?": stored elsewhere
            if case == "refused_plainly":

                async def plain(self, **arguments):
                    raise NoteContentMutationServiceError(403, "You cannot change this project")

                monkeypatch.setattr(NoteContentMutationService, "delete_note", plain)
            if case == "source_unsettled":
                # Git is ahead of SQL (the worker is held back): the draft would not hold
                # the bytes the delete removes.
                snapshot = a_repo.snapshot()
                (path_a,) = snapshot.files
                publish_into(a_repo, add={path_a: (snapshot.files[path_a] + "\nNeu\n", UUID(note))})
            before = a_repo.head(), b_repo.head()
            answer = await move(client, a, note, target, path)
            assert reservations() == {}
    expected = {
        "unknown_target": (404, "not_found"),
        "target_not_accepting": (409, "not_accepting"),
        "draft_refused": (422, "invalid_draft"),
        "on_disk": (409, "occupied"),
        "unnormalized": (422, "invalid_path"),
        "refused_plainly": (403, "refused_by_source"),
        "source_unsettled": (409, "not_accepting"),
    }[case]
    assert (answer.status_code, answer.json()["detail"]["kind"]) == expected, answer.text
    assert "Nothing was moved" in answer.json()["detail"]["message"]
    assert (a_repo.head(), b_repo.head()) == before
    assert intents(b_repo, b) == [] and open_drafts(b_repo, b) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "after", ["answer_lost", "draft_refused", "keeps_changing", "catch_up_fails", "crashes"]
)
async def test_every_outcome_after_the_source_let_go(host_setup, two_projects, monkeypatch, after):
    from basic_memory.api.app import app
    from basic_memory.repository.git_transaction_repository import GitTransactionError
    from basic_memory.services import git_cross_project_move
    from basic_memory.services.git_project_runtime import GitProjectRuntime
    from basic_memory.services.git_transactions import GitTransactionService

    a, a_repo, b, b_repo = two_projects
    commit = GitTransactionService.commit_admitted
    catch_up = GitProjectRuntime.catch_up_admitted
    draft_status, commit_step = git_cross_project_move._draft_status, git_cross_project_move._commit
    calls = {"catch_up": 0}

    async def lost(self, *arguments, **options):
        await commit(self, *arguments, **options)
        raise GitTransactionError(503, "Transaction operation interrupted")

    async def refused(self, *arguments, **options):
        raise GitTransactionError(422, "Invalid draft note")

    async def conflicted(self, *arguments, **options):
        raise GitTransactionError(409, "Project advanced; draft preserved")

    async def failing_after_commit(self, **options):
        calls["catch_up"] += 1
        if options.get("known_ancestor") is not None:
            raise RuntimeError("SQL is busy")
        return await catch_up(self, **options)

    async def crashing(*arguments, **options):
        raise RuntimeError("Unexpected failure")

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
            match after:
                case "answer_lost":
                    monkeypatch.setattr(GitTransactionService, "commit_admitted", lost)
                case "draft_refused":
                    monkeypatch.setattr(GitTransactionService, "commit_admitted", refused)
                case "keeps_changing":
                    monkeypatch.setattr(GitTransactionService, "commit_admitted", conflicted)

                    async def always_conflicted(target, intent):
                        status = await target.transactions.status_admitted(
                            intent.transaction_id, intent.transaction_key
                        )
                        return status.model_copy(update={"state": "conflicted"})

                    monkeypatch.setattr(git_cross_project_move, "_draft_status", always_conflicted)
                case "catch_up_fails":
                    monkeypatch.setattr(
                        GitProjectRuntime, "catch_up_admitted", failing_after_commit
                    )
                case "crashes":
                    monkeypatch.setattr(git_cross_project_move, "_commit", crashing)
            answer = await move(client, a, note, b, "archive/Mover.md")
            # Only this test's own patches go; the fixtures keep theirs.
            monkeypatch.setattr(GitTransactionService, "commit_admitted", commit)
            monkeypatch.setattr(GitProjectRuntime, "catch_up_admitted", catch_up)
            monkeypatch.setattr(git_cross_project_move, "_draft_status", draft_status)
            monkeypatch.setattr(git_cross_project_move, "_commit", commit_step)
            listed = (await client.get("/v2/git/moves")).json()["moves"]
            resumed = (await client.post("/v2/git/moves/resume")).json()["outcomes"]
    assert answer.status_code == 200, answer.text
    state = answer.json()["state"]
    assert UUID(note) not in a_repo.published_note_ids()
    if after in {"answer_lost", "catch_up_fails"}:
        assert state == "completed" and listed == [] and resumed == []
        assert answer.json()["index_pending"] is (after == "catch_up_fails")
    else:
        # The source let go; the move stays open and resume finishes it.
        assert state == "pending", answer.text
        # Basic Memory's own words only: no raw exception text reaches the caller.
        assert "Unexpected failure" not in (answer.json()["message"] or "")
        assert [move["move_id"] for move in listed] == [answer.json()["move_id"]]
        assert [outcome["outcome"] for outcome in resumed] == ["completed"]
    assert b_repo.snapshot().manifest.note_ids == {"archive/Mover.md": UUID(note)}


@pytest.mark.asyncio
async def test_a_stuck_move_cannot_go_to_a_taken_path_and_its_uuid_cannot_be_moved_again(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    publish_into(b_repo, add={"m/Mover.md": (canonical("Besetzt", "m/Mover.md"), uuid4())})
    publish_into(a_repo, add={"notes/Mover.md": (canonical("Mover", "notes/mover.md"), UUID(note))})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, a)
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            again = await move(client, a, note, b, "anders/Mover.md")
    assert again.status_code == 409 and again.json()["detail"]["kind"] == "reserved"
    assert stuck["move_id"] in again.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_resolving_into_a_taken_path_is_refused_and_keeps_the_move(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    publish_into(b_repo, add={"m/Mover.md": (canonical("Besetzt", "m/Mover.md"), uuid4())})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            taken = await client.post(
                f"/v2/git/moves/{stuck['move_id']}/resolve", json={"choice": "target"}
            )
            (still,) = (await client.get("/v2/git/moves")).json()["moves"]
    assert taken.status_code == 409 and taken.json()["detail"]["kind"] == "occupied"
    assert still["move_id"] == stuck["move_id"] and still["state"] == "stuck"


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["target", "source"])
async def test_resolving_a_move_that_already_published_finishes_or_refuses(
    host_setup, two_projects, tmp_path, monkeypatch, choice
):
    from basic_memory.api.app import app
    from basic_memory.index import git_host

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(
        tmp_path, "after_target_publish_before_record", a, a_repo, b, b_repo, note, "m/Mover.md"
    )

    async def no_resume(host):
        return []

    # The start-up resume would finish it first; here a person comes first.
    monkeypatch.setattr(git_host, "resume_cross_project_moves", no_resume)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (open_move,) = (await client.get("/v2/git/moves")).json()["moves"]
            resolved = await client.post(
                f"/v2/git/moves/{open_move['move_id']}/resolve", json={"choice": choice}
            )
    if choice == "target":
        assert resolved.status_code == 200 and resolved.json()["state"] == "completed"
        assert intents(b_repo, b) == []
    else:
        assert resolved.status_code == 409 and resolved.json()["detail"]["kind"] == "completed"
    assert b_repo.snapshot().manifest.note_ids == {"m/Mover.md": UUID(note)}


@pytest.mark.asyncio
async def test_a_lost_draft_record_is_rebuilt_and_an_unreadable_intent_is_listed(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    (planned,) = intents(b_repo, b)
    # The draft's pointer is gone (a damaged copy); the intent still holds the bytes.
    (b_repo.path / "bm-transactions" / f"{planned.transaction_id}.json").unlink()
    # And next to that intent sit a stray file and a damaged intent holding a key.
    moves = b_repo.path / "bm-moves"
    (moves / "kaputt.json").write_text("{}")
    damaged = uuid4()
    (moves / f"{damaged}.json").write_text(
        f'{{"move_id": "{damaged}", "transaction_key": "KEY-PLAINTEXT-123"}}'
    )
    chosen = uuid4()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, b)
            listing = (await client.get("/v2/git/moves")).json()
            read = await client.get(f"/v2/projects/{b}/knowledge/entities/{note}")
            # While an intent cannot be read, its move's UUID is unknown: no caller may
            # choose a UUID (a random one is still fine).
            put = await client.put(
                f"/v2/projects/{a}/knowledge/entities/{chosen}",
                json={"title": "Gewaehlt", "directory": "notes", "content": "x\n"},
            )
            free = await client.post(
                f"/v2/projects/{a}/knowledge/write",
                json={"note": {"title": "Frei", "directory": "notes", "content": "x\n"}},
            )
            (moves / "kaputt.json").unlink()
            (moves / f"{damaged}.json").unlink()
            # Once the files are gone (and the moves read again), the UUID is free.
            assert (await client.get("/v2/git/moves")).json()["unreadable"] == []
            again = await client.put(
                f"/v2/projects/{a}/knowledge/entities/{chosen}",
                json={"title": "Gewaehlt", "directory": "notes", "content": "x\n"},
            )
    # The readable intent next to the bad files was resumed: the note arrived.
    assert read.status_code == 200, read.text
    assert listing["moves"] == []
    assert sorted(entry.rsplit("/", 1)[-1] for entry in listing["unreadable"]) == sorted(
        [f"{damaged}.json cannot be read", "kaputt.json is not a move intent"]
    )
    text = json.dumps(listing)
    assert "KEY-PLAINTEXT-123" not in text and str(b_repo.path) not in text
    assert put.status_code == 409 and "cannot be read" in put.json()["detail"]["message"]
    assert free.status_code == 200, free.text
    assert again.status_code in {200, 201, 202}, again.text


@pytest.mark.asyncio
async def test_a_base_outside_the_source_history_keeps_the_move_stuck(
    host_setup, two_projects, tmp_path
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    (planned,) = intents(b_repo, b)
    elsewhere = (
        a_repo._git("commit-tree", f"{a_repo.head()}^{{tree}}", input_bytes=b"Elsewhere\n")
        .strip()
        .decode()
    )
    GitMoveIntentRepository(b_repo, UUID(b)).save(
        planned.model_copy(update={"state": "drafted", "source_base_commit": elsewhere})
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
            resolve = f"/v2/git/moves/{stuck['move_id']}/resolve"
            to_source = await client.post(resolve, json={"choice": "source"})
            (still,) = (await client.get("/v2/git/moves")).json()["moves"]
            to_target = await client.post(resolve, json={"choice": "target"})
    assert stuck["state"] == "stuck" and "history" in stuck["stuck_reason"]
    # The bytes the note had in the source are gone from its history: only 'target'.
    assert to_source.status_code == 409, to_source.text
    assert to_source.json()["detail"]["kind"] == "no_source_bytes"
    assert still["move_id"] == stuck["move_id"]
    assert to_target.status_code == 200 and to_target.json()["state"] == "completed"
    assert b_repo.snapshot().manifest.note_ids == {"m/Mover.md": UUID(note)}


@pytest.mark.asyncio
async def test_a_catch_up_of_a_failing_project_is_refused_by_name(
    host_setup, two_projects, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.repository.git_projection_repository import GitProjectionConflict
    from basic_memory.services.git_project_runtime import GitProjectRuntime

    _, _, b, _ = two_projects

    async def failing(self):
        raise GitProjectionConflict("Projection requires canonical BM frontmatter")

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            fine = await client.post(f"/v2/projects/{b}/git/moves/catch-up")
            monkeypatch.setattr(GitProjectRuntime, "catch_up", failing)
            refused = await client.post(f"/v2/projects/{b}/git/moves/catch-up")
    assert fine.status_code == 200 and fine.json()["index_pending"] is False
    assert refused.status_code == 409 and refused.json()["detail"]["kind"] == "not_accepting"


@pytest.mark.asyncio
async def test_a_note_the_target_already_waits_for_is_refused_as_already_there(
    host_setup, two_projects
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            # A peer already published the move's target commit; the source has not let go.
            publish_into(
                b_repo,
                add={"moved/mover.md": (canonical("Mover", "moved/mover.md"), UUID(note))},
                moved={UUID(note): UUID(a)},
            )
            answer = await move(client, a, note, b, "anders/Mover.md")
    assert answer.status_code == 409 and answer.json()["detail"]["kind"] == "already_in_target"
    assert UUID(note) in a_repo.published_note_ids()


@pytest.mark.asyncio
async def test_an_intent_that_cannot_be_saved_leaves_the_source_untouched(
    host_setup, two_projects, monkeypatch
):
    from basic_memory.api.app import app

    a, a_repo, b, b_repo = two_projects

    def full_disk(self, intent):
        raise OSError(28, "No space left on device")

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            note = (await write(client, a, "Mover"))["external_id"]
            await ready(client, a)
            await ready(client, b)
            save = GitMoveIntentRepository.save
            monkeypatch.setattr(GitMoveIntentRepository, "save", full_disk)
            with pytest.raises(OSError, match="No space"):
                await move(client, a, note, b, "archive/Mover.md")
            monkeypatch.setattr(GitMoveIntentRepository, "save", save)
            assert reservations() == {}
    assert UUID(note) in a_repo.published_note_ids()
    assert open_drafts(b_repo, b) == [] and intents(b_repo, b) == []


@pytest.mark.asyncio
async def test_a_rebuilt_draft_the_target_refuses_makes_the_move_stuck(
    host_setup, two_projects, tmp_path, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.repository.git_transaction_repository import GitTransactionError
    from basic_memory.services.git_transactions import GitTransactionService

    a, a_repo, b, b_repo = two_projects
    note = await seeded(app, a, b)
    await crash(tmp_path, "after_delete_before_commit", a, a_repo, b, b_repo, note, "m/Mover.md")
    # The target advanced while the host was down, so the old draft must be rebuilt.
    publish_into(b_repo, add={"other.md": (canonical("Andere", "other.md"), uuid4())})

    async def refusing(self, *arguments, **options):
        raise GitTransactionError(422, "Invalid draft note m/Mover.md")

    monkeypatch.setattr(GitTransactionService, "write_identified_admitted", refusing)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            (stuck,) = (await client.get("/v2/git/moves")).json()["moves"]
    assert stuck["state"] == "stuck" and "Invalid draft" in stuck["stuck_reason"]
    assert UUID(note) not in b_repo.published_note_ids()
