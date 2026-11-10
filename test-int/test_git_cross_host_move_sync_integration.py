"""Moves that keep a note's UUID, travelling between two hosts by Git sync (I1 part A).

The other host is simulated by publishing straight into each project's peer repository
commits of the shape its W2 move leaves: the source loses the note, the target gains it
with BM's `Basic-Memory-Moved-Note` trailer (written by the same `prepare` the route uses,
and read back by the same reader the guard uses). The note bytes and the commit message
are simpler than the route's: the guard reads only manifests and trailers. This host syncs
the projects one by one, in any order; the sync guard and the projection must let a proven
move through and wait, and must still refuse a real duplicate by name.
"""

import asyncio
import json
import subprocess
from itertools import permutations
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.config_git import GitSyncRemote
from basic_memory.models import Entity
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.schemas.git_transactions import GitTransactionCommit, GitTransactionStart

from tests.git_host_support import add_git_project, host_setup as host_setup, ready


def git(*args):
    return subprocess.run(
        ["git", *map(str, args)], capture_output=True, check=True, timeout=30
    ).stdout


def canonical(title: str, path: str, body: str = "Body\n") -> str:
    return f"---\ntitle: {title}\ntype: note\npermalink: {path[:-3]}\n---\n\n{body}"


def peer_publish(
    peer: GitProjectRepository,
    *,
    add: dict[str, tuple[str, UUID]] | None = None,
    remove: tuple[str, ...] = (),
    moved: dict[UUID, UUID] | None = None,
) -> str:
    base = peer.snapshot()
    files = {path: text for path, text in base.files.items() if path not in remove}
    ids = {path: note for path, note in base.manifest.note_ids.items() if path not in remove}
    for path, (text, note) in (add or {}).items():
        files[path], ids[path] = text, note
    prepared = peer.prepare(
        transaction_id=uuid4(),
        base_commit=base.commit_id,
        files=files,
        note_ids=ids,
        moved_notes=moved,
    )
    return peer.publish(prepared).commit_id


class Hosted:
    """The projects of this host by name: external id, local repository and peer."""

    def __init__(self, host_setup):
        self.config = host_setup.config
        self.ids = {"a": host_setup.project.external_id}
        self.local = {"a": host_setup.repository}
        self.sql = {"a": host_setup.project.id}
        self.peers: dict[str, GitProjectRepository] = {}

    async def add(self, host_setup, config_manager, name: str, label: str) -> None:
        project, repository = await add_git_project(host_setup, config_manager, label)
        self.ids[name], self.local[name], self.sql[name] = (
            project.external_id,
            repository,
            project.id,
        )

    def clone_peers(self, config_manager, tmp_path, *, merge: tuple[str, ...] = ()) -> None:
        for name, repository in self.local.items():
            path = tmp_path / f"peer-{name}.git"
            git("clone", "--bare", "--no-local", repository.path, path)
            self.peers[name] = GitProjectRepository(path)
        self.config.git_sync_remotes = {
            UUID(self.ids[name]): GitSyncRemote(url=str(peer.path))
            for name, peer in self.peers.items()
        }
        self.config.git_sync_merge = {UUID(self.ids[name]): "auto" for name in merge}
        config_manager.save_config(self.config)

    def runtime(self, name: str):
        host = resolve_container().git_host
        assert host is not None
        return host.runtime_for(self.ids[name])

    async def sync(self, name: str, *, review: str | None = None):
        return await self.runtime(name).sync.synchronize(review=review)

    async def owner(self, note: UUID) -> list[tuple[str, str]]:
        names = {project_id: name for name, project_id in self.sql.items()}
        _, factory = await db.get_or_create_db(self.config.database_path)
        async with factory() as session:
            rows = await session.execute(
                select(Entity.project_id, Entity.file_path).where(Entity.external_id == str(note))
            )
            return [(names[project_id], path) for project_id, path in rows]


async def write(client, project: str, title: str) -> UUID:
    response = await client.post(
        f"/v2/projects/{project}/knowledge/write",
        json={"note": {"title": title, "directory": "", "content": f"About {title}\n"}},
    )
    assert response.status_code == 200, response.text
    return UUID(response.json()["entity"]["external_id"])


async def settled(hosted: Hosted, *names: str) -> None:
    """Every named project caught up, without any notify: wake-ups must suffice."""
    async with asyncio.timeout(20):
        for name in names:
            worker = hosted.runtime(name).worker
            while not (await worker.status()).caught_up:
                await asyncio.sleep(0.05)


def client_for(app):
    return AsyncClient(transport=ASGITransport(app), base_url="http://test")


async def seed(hosted: Hosted, notes: dict[str, str]) -> dict[str, UUID]:
    """Write `title -> project name` notes through the API and wait for every project."""
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with client_for(app) as client:
            ids = {
                title: await write(client, hosted.ids[name], title) for title, name in notes.items()
            }
            for name in hosted.ids:
                await ready(client, hosted.ids[name])
    return ids


async def move_here(hosted: Hosted, client, note: UUID, source: str, target: str, path: str):
    """This host's own W2 move, step by step: draft, source delete, commit with proof."""
    transactions = hosted.runtime(target).transactions
    started = await transactions.start(GitTransactionStart())
    title = path.rsplit("/", 1)[-1][:-3]
    await transactions.write_identified(
        started.transaction_id,
        started.transaction_key,
        path=path,
        content=canonical(title, path),
        note_id=note,
        expected_revision=0,
    )
    deleted = await client.delete(f"/v2/projects/{hosted.ids[source]}/knowledge/entities/{note}")
    assert deleted.status_code in {200, 202}, deleted.text
    committed = await transactions.commit(
        started.transaction_id,
        started.transaction_key,
        GitTransactionCommit(expected_revision=1, message=f"Move {path}"),
        moved_notes={note: UUID(hosted.ids[source])},
    )
    assert committed.state == "committed"


# --- P5: one move, both sync orders ---


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [("b", "a"), ("a", "b")])
async def test_a_move_on_the_other_host_arrives_in_either_sync_order(
    host_setup, config_manager, tmp_path, order
):
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path)
    peer_publish(hosted.peers["a"], remove=("Xnote.md",))
    peer_publish(
        hosted.peers["b"],
        add={"moved/xnote.md": (canonical("Xnote", "moved/xnote.md"), x)},
        moved={x: UUID(hosted.ids["a"])},
    )
    async with app.router.lifespan_context(app):
        # Before I1 the target-first order failed as invalid_incoming (P5).
        for name in order:
            result = await hosted.sync(name)
            assert result.last_attempt.outcome == "pulled", result
        await settled(hosted, "a", "b")
        assert await hosted.owner(x) == [("b", "moved/xnote.md")]


# --- P6: crossing moves converge, both sync orders ---


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
async def test_crossing_moves_converge_in_one_round(host_setup, config_manager, tmp_path, order):
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    ids = await seed(hosted, {"Znote": "a", "Ynote": "b"})
    z, y = ids["Znote"], ids["Ynote"]
    hosted.clone_peers(config_manager, tmp_path)
    # In one window the other host moved Z from a to b and Y from b to a.
    peer_publish(
        hosted.peers["a"],
        remove=("Znote.md",),
        add={"moved/ynote.md": (canonical("Ynote", "moved/ynote.md"), y)},
        moved={y: UUID(hosted.ids["b"])},
    )
    peer_publish(
        hosted.peers["b"],
        remove=("Ynote.md",),
        add={"moved/znote.md": (canonical("Znote", "moved/znote.md"), z)},
        moved={z: UUID(hosted.ids["a"])},
    )
    async with app.router.lifespan_context(app):
        # Before I1 both stayed invalid_incoming in every round (P6).
        for name in order:
            result = await hosted.sync(name)
            assert result.last_attempt.outcome == "pulled", result
        await settled(hosted, "a", "b")
        assert await hosted.owner(z) == [("b", "moved/znote.md")]
        assert await hosted.owner(y) == [("a", "moved/ynote.md")]


# --- Chains and double moves ---


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [("c", "a", "b"), ("b", "c", "a"), ("a", "c", "b")])
async def test_a_chain_of_moves_converges_in_any_sync_order(
    host_setup, config_manager, tmp_path, order
):
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    await hosted.add(host_setup, config_manager, "c", "third")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path)
    # The other host moved X from a to b and on to c before this host synced anything.
    peer_publish(hosted.peers["a"], remove=("Xnote.md",))
    peer_publish(
        hosted.peers["b"],
        add={"xnote.md": (canonical("Xnote", "xnote.md"), x)},
        moved={x: UUID(hosted.ids["a"])},
    )
    peer_publish(hosted.peers["b"], remove=("xnote.md",))
    peer_publish(
        hosted.peers["c"],
        add={"end/xnote.md": (canonical("Xnote", "end/xnote.md"), x)},
        moved={x: UUID(hosted.ids["b"])},
    )
    async with app.router.lifespan_context(app):
        for name in order:
            result = await hosted.sync(name)
            assert result.last_attempt.outcome == "pulled", (name, result)
        await settled(hosted, "a", "b", "c")
        assert await hosted.owner(x) == [("c", "end/xnote.md")]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(permutations("abc")))
async def test_crossing_chains_converge_in_one_round(host_setup, config_manager, tmp_path, order):
    """N1 goes a -> b -> c while N2 goes c -> b -> a, all before this host syncs."""
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    await hosted.add(host_setup, config_manager, "c", "third")
    ids = await seed(hosted, {"Nfirst": "a", "Nsecond": "c"})
    first, second = ids["Nfirst"], ids["Nsecond"]
    hosted.clone_peers(config_manager, tmp_path)
    peer_publish(
        hosted.peers["a"],
        remove=("Nfirst.md",),
        add={"in/nsecond.md": (canonical("Nsecond", "in/nsecond.md"), second)},
        moved={second: UUID(hosted.ids["b"])},
    )
    peer_publish(
        hosted.peers["b"],
        add={
            "nfirst.md": (canonical("Nfirst", "nfirst.md"), first),
            "nsecond.md": (canonical("Nsecond", "nsecond.md"), second),
        },
        moved={first: UUID(hosted.ids["a"]), second: UUID(hosted.ids["c"])},
    )
    peer_publish(hosted.peers["b"], remove=("nfirst.md", "nsecond.md"))
    peer_publish(
        hosted.peers["c"],
        remove=("Nsecond.md",),
        add={"in/nfirst.md": (canonical("Nfirst", "in/nfirst.md"), first)},
        moved={first: UUID(hosted.ids["b"])},
    )
    async with app.router.lifespan_context(app):
        for name in order:
            result = await hosted.sync(name)
            assert result.last_attempt.outcome == "pulled", (name, result)
        await settled(hosted, "a", "b", "c")
        assert await hosted.owner(first) == [("c", "in/nfirst.md")]
        assert await hosted.owner(second) == [("a", "in/nsecond.md")]


@pytest.mark.asyncio
async def test_one_note_moved_to_two_projects_on_two_hosts_is_refused_by_name(
    host_setup, config_manager, tmp_path
):
    from fastmcp import Client

    from basic_memory.api.app import app
    from basic_memory.mcp.server import mcp
    import basic_memory.mcp.tools  # noqa: F401

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    await hosted.add(host_setup, config_manager, "c", "third")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path)
    # The other host moved X from a to c ...
    peer_publish(hosted.peers["a"], remove=("Xnote.md",))
    peer_publish(
        hosted.peers["c"],
        add={"there/xnote.md": (canonical("Xnote", "there/xnote.md"), x)},
        moved={x: UUID(hosted.ids["a"])},
    )
    # The MCP lifespan owns the host here, so bm_project can render the status too.
    async with Client(mcp) as mcp_client:
        async with client_for(app) as client:
            # ... while this host moved it from a to b.
            await move_here(hosted, client, x, "a", "b", "here/xnote.md")
            await settled(hosted, "a", "b")
            local_c = hosted.local["c"].head()
            result = await hosted.sync("c")
            assert result.last_attempt.kind == "failed"
            assert result.last_attempt.reason == "invalid_incoming"
            assert result.failure_detail is not None
            assert str(x) in result.failure_detail
            assert "project 'second'" in result.failure_detail
            assert "incoming for project 'third'" in result.failure_detail
            # Both copies are intact; nothing was published here, and the host runs on.
            assert hosted.local["c"].head() == local_c
            assert x in hosted.peers["c"].snapshot().manifest.note_ids.values()
            assert await hosted.owner(x) == [("b", "here/xnote.md")]
            status = await client.get(f"/v2/projects/{hosted.ids['c']}/git/status")
            assert status.json()["synchronization"]["failure_detail"] == result.failure_detail
            text = await mcp_client.call_tool(
                "bm_project", {"command": "status", "project": "third"}
            )
            assert result.failure_detail in text.content[0].text
            # The refusal stays until a person resolves it (no recovery tool).
            again = await hosted.sync("c")
            assert again.last_attempt.reason == "invalid_incoming"


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [("b", "c", "a"), ("c", "b", "a")])
async def test_a_double_move_synced_before_its_source_is_refused_by_name(
    host_setup, config_manager, tmp_path, order
):
    """Two other hosts moved X a -> b and a -> c; this host syncs both targets first.

    While a still owns X, both claims name a move from a. The second target must be
    refused as invalid_incoming (not projected into SQL's unique index), and the host
    must start again afterwards.
    """
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    await hosted.add(host_setup, config_manager, "c", "third")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path)
    peer_publish(hosted.peers["a"], remove=("Xnote.md",))
    for name in ("b", "c"):
        peer_publish(
            hosted.peers[name],
            add={f"{name}/xnote.md": (canonical("Xnote", f"{name}/xnote.md"), x)},
            moved={x: UUID(hosted.ids["a"])},
        )
    first, second, source = order
    labels = {"b": "second", "c": "third"}
    async with app.router.lifespan_context(app):
        assert (await hosted.sync(first)).last_attempt.outcome == "pulled"
        refused = await hosted.sync(second)
        assert refused.last_attempt.reason == "invalid_incoming", refused
        assert refused.failure_detail is not None
        assert str(x) in refused.failure_detail
        assert f"project '{labels[first]}'" in refused.failure_detail
        assert f"incoming for project '{labels[second]}'" in refused.failure_detail
        assert x not in hosted.local[second].snapshot().manifest.note_ids.values()
        assert (await hosted.sync(source)).last_attempt.outcome == "pulled"
        await settled(hosted, "a", first, second)
        assert await hosted.owner(x) == [(first, f"{first}/xnote.md")]
        # The refusal stays named; it never turns into a raw database error.
        again = await hosted.sync(second)
        assert again.last_attempt.reason == "invalid_incoming", again
        for name in ("a", "b", "c"):
            stage = (await hosted.runtime(name).worker.status()).caught_up
            assert stage, name
    # Peer activity alone must not keep this host from starting.
    async with app.router.lifespan_context(app):
        assert await hosted.owner(x) == [(first, f"{first}/xnote.md")]


@pytest.mark.asyncio
async def test_a_local_move_of_a_note_a_peer_moved_elsewhere_moves_nothing(
    host_setup, config_manager, tmp_path
):
    """A peer moved X from a to c; this host synced c (it waits) and then moves X to b.

    The third project's head holds X: the target commit could never take it, nor could
    the source take it back. Refused before the source delete, with nothing moved.
    """
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    await hosted.add(host_setup, config_manager, "c", "third")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path)
    peer_publish(hosted.peers["a"], remove=("Xnote.md",))
    peer_publish(
        hosted.peers["c"],
        add={"there/xnote.md": (canonical("Xnote", "there/xnote.md"), x)},
        moved={x: UUID(hosted.ids["a"])},
    )
    async with app.router.lifespan_context(app):
        async with client_for(app) as client:
            assert (await hosted.sync("c")).last_attempt.outcome == "pulled"
            before = hosted.local["a"].head(), hosted.local["b"].head()
            answer = await client.post(
                f"/v2/projects/{hosted.ids['a']}/git/moves",
                json={
                    "external_id": str(x),
                    "target_project_id": hosted.ids["b"],
                    "destination_path": "here/xnote.md",
                },
            )
            listing = (await client.get("/v2/git/moves")).json()
            # The peer's move then completes here as usual.
            assert (await hosted.sync("a")).last_attempt.outcome == "pulled"
            await settled(hosted, "a", "b", "c")
            assert await hosted.owner(x) == [("c", "there/xnote.md")]
    assert answer.status_code == 409, answer.text
    assert answer.json()["detail"]["kind"] == "owned_elsewhere"
    assert "project 'third'" in answer.json()["detail"]["message"]
    assert "Nothing was moved" in answer.json()["detail"]["message"]
    assert before[1] == hosted.local["b"].head()
    assert listing == {"moves": [], "unreadable": []}


@pytest.mark.asyncio
async def test_a_git_error_during_validation_shows_no_raw_detail(
    host_setup, config_manager, tmp_path, monkeypatch
):
    """failure_detail carries BM's own words only; Git stderr can hold host paths."""
    from basic_memory.api.app import app
    from basic_memory.repository.git_project_repository import GitPublicationError

    hosted = Hosted(host_setup)
    hosted.clone_peers(config_manager, tmp_path)
    peer_publish(hosted.peers["a"], add={"n.md": (canonical("N", "n.md"), uuid4())})

    def failing(self, attempt_id, commit_id):
        raise GitPublicationError("fatal: /private/host/path/project.git is broken")

    monkeypatch.setattr(GitProjectRepository, "incoming_snapshots", failing)
    async with app.router.lifespan_context(app):
        result = await hosted.sync("a")
        assert result.last_attempt.reason == "invalid_incoming"
        assert result.failure_detail is None


# --- Merges ---


@pytest.mark.asyncio
async def test_an_automatic_merge_carries_a_move_from_the_other_host(
    host_setup, config_manager, tmp_path
):
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path, merge=("a",))
    peer_publish(hosted.peers["a"], remove=("Xnote.md",))
    peer_publish(
        hosted.peers["b"],
        add={"moved/xnote.md": (canonical("Xnote", "moved/xnote.md"), x)},
        moved={x: UUID(hosted.ids["a"])},
    )
    async with app.router.lifespan_context(app):
        async with client_for(app) as client:
            # This host wrote in a meanwhile, so a's histories diverge.
            await write(client, hosted.ids["a"], "Local")
            assert (await hosted.sync("b")).last_attempt.outcome == "pulled"
            merged = await hosted.sync("a")
            assert merged.last_attempt.outcome == "merged", merged
            await settled(hosted, "a", "b")
            assert await hosted.owner(x) == [("b", "moved/xnote.md")]


@pytest.mark.asyncio
async def test_keeping_a_note_this_host_moved_away_is_refused_and_delete_converges(
    host_setup, config_manager, tmp_path
):
    """The other host edited X in a while this host moved X from a to b.

    The merge of a reports delete_vs_change. "Keep" would put X's UUID back into a while
    b publishes it: refused by name, nothing published. "Delete" converges. (Re-keying X
    to a new UUID is not offered; documented limit.)
    """
    from basic_memory.api.app import app

    hosted = Hosted(host_setup)
    await hosted.add(host_setup, config_manager, "b", "second")
    x = (await seed(hosted, {"Xnote": "a"}))["Xnote"]
    hosted.clone_peers(config_manager, tmp_path, merge=("a",))
    edited = hosted.peers["a"].snapshot().files["Xnote.md"] + "\nEdited there\n"
    peer_publish(hosted.peers["a"], add={"Xnote.md": (edited, x)})
    async with app.router.lifespan_context(app):
        async with client_for(app) as client:
            await move_here(hosted, client, x, "a", "b", "moved/xnote.md")
            await settled(hosted, "a", "b")
            status = await hosted.sync("a")
            assert status.last_attempt.outcome == "needs_review", status
            assert status.open_review is not None
            package = hosted.local["a"].path / "bm-merge-review" / status.open_review.id
            document = json.loads((package / "konflikte.json").read_text())
            assert [conflict["kind"] for conflict in document["conflicts"]] == ["delete_vs_change"]
            local_a = hosted.local["a"].head()
            for decision, outcome in (("behalten", None), ("loeschen", "merged")):
                document["conflicts"][0]["decision"] = {"action": decision}
                (package / "konflikte.json").write_text(json.dumps(document))
                result = await hosted.sync("a", review=status.open_review.id)
                if outcome is None:
                    assert result.last_attempt.reason == "invalid_incoming", result
                    assert result.failure_detail is not None
                    assert str(x) in result.failure_detail
                    assert "project 'second'" in result.failure_detail
                    assert hosted.local["a"].head() == local_a
                else:
                    assert result.last_attempt.outcome == outcome, result
            await settled(hosted, "a", "b")
            assert await hosted.owner(x) == [("b", "moved/xnote.md")]
            assert x not in hosted.local["a"].snapshot().manifest.note_ids.values()
