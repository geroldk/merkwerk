"""Normal API/MCP explicit transactions with the real owned projection pipeline."""

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import pytest
from fastmcp import Client
from httpx import ASGITransport, AsyncClient
from mcp.types import TextContent
from sqlalchemy import select, func

from basic_memory import db
from basic_memory.api.container import resolve_container
from basic_memory.index.git_projection import GitProjectionPipeline
from basic_memory.models import Entity, NoteContent
from basic_memory.schemas.git_transactions import GitCommittedState
from basic_memory.schemas.git_transactions import GitTransactionWrite
from tests.git_host_support import host_setup as host_setup, ready


def note(title, text):
    return (
        f"---\ntitle: {title}\ntype: note\npermalink: test-project/{title.lower()}\n---\n\n{text}\n"
    )


async def start(client, prefix, **source):
    response = await client.post(prefix, json=source)
    assert response.status_code == 200, response.text
    return response.json()


def headers(transaction):
    return {"X-BM-Transaction-Key": transaction["transaction_key"]}


async def write(client, prefix, transaction, path, content, revision=0, operation="replace"):
    response = await client.post(
        f"{prefix}/{transaction['transaction_id']}/write",
        headers=headers(transaction),
        json={
            "path": path,
            "content": content,
            "expected_revision": revision,
            "operation": operation,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_real_api_private_bundle_publishes_once_before_heavy_indexes(host_setup, monkeypatch):
    from basic_memory.api.app import app

    apply = GitProjectionPipeline.apply
    release = asyncio.Event()

    async def blocked(self, stage, commit_id, *, graph_base_commit=None):
        if stage in {"graph", "vectors"}:
            await release.wait()
        await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)

    monkeypatch.setattr(GitProjectionPipeline, "apply", blocked)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            project = host_setup.project.external_id
            prefix = f"/v2/projects/{project}/git/transactions"
            transaction = await start(client, prefix)
            base = host_setup.repository.head()
            await write(
                client,
                prefix,
                transaction,
                "A.md",
                note("A", "- [fact] Bundlealpha\n- relates_to [[B]]"),
            )
            changed = await write(
                client, prefix, transaction, "B.md", note("B", "- [fact] Bundlebeta"), revision=1
            )
            assert changed["revision"] == 2 and host_setup.repository.head() == base
            listing = await client.get(
                f"{prefix}/{transaction['transaction_id']}/notes", headers=headers(transaction)
            )
            assert listing.headers["cache-control"] == "no-store"
            identities = {row["path"]: row["external_id"] for row in listing.json()["notes"]}
            assert set(identities) == {"A.md", "B.md"}
            for identifier in ("A.md", identities["A.md"], "test-project/a"):
                read = await client.get(
                    f"{prefix}/{transaction['transaction_id']}/note",
                    headers=headers(transaction),
                    params={"identifier": identifier},
                )
                assert "Bundlealpha" in read.json()["note"]["content"]
            basis = await client.get(
                f"{prefix}/{transaction['transaction_id']}/notes",
                headers=headers(transaction),
                params={"view": "base"},
            )
            assert basis.json()["notes"] == []
            _, factory = await db.get_or_create_db(host_setup.config.database_path)
            async with factory() as session:
                assert await session.scalar(select(func.count()).select_from(NoteContent)) == 0
            hidden = await client.get(
                f"/v2/projects/{project}/knowledge/entities/{identities['A.md']}"
            )
            assert hidden.status_code == 404
            hidden_search = await client.request(
                "QUERY",
                f"/v2/projects/{project}/search/",
                json={"text": "Bundlealpha", "retrieval_mode": "fts"},
            )
            assert (
                hidden_search.status_code == 200
                and '"results":[]' in hidden_search.text.replace(" ", "")
            )
            committed = await asyncio.wait_for(
                client.post(
                    f"{prefix}/{transaction['transaction_id']}/commit",
                    headers=headers(transaction),
                    json={"expected_revision": 2, "message": "Publish linked A and B"},
                ),
                5,
            )
            assert committed.status_code == 200, committed.text
            published = committed.json()["published_commit"]
            assert published != base and host_setup.repository.head() == published
            assert not release.is_set()
            assert host_setup.repository._git("rev-list", "--count", "main").strip() == b"2"
            assert (
                "Publish linked A and B"
                in host_setup.repository._git("log", "-1", "--format=%B").decode()
            )
            repeated = await client.post(
                f"{prefix}/{transaction['transaction_id']}/commit",
                headers=headers(transaction),
                json={"expected_revision": 2},
            )
            assert (
                repeated.json()["already_committed"]
                and repeated.json()["published_commit"] == published
            )
            release.set()
            await ready(client, project)
            found = await client.request(
                "QUERY",
                f"/v2/projects/{project}/search/",
                json={"text": "Bundlealpha", "retrieval_mode": "fts"},
            )
            assert found.status_code == 200 and "Bundlealpha" in found.text
            host = resolve_container().git_host
            assert host is not None
            search = host.runtime_for(project).pipeline.search_service.repository
            async with factory() as session:
                entities = list(await session.scalars(select(Entity)))
                assert len(entities) == 2
            for entity in entities:
                assert await search.get_entity_physical_chunk_keys(entity.id)
            status = await client.get(
                f"{prefix}/{transaction['transaction_id']}", headers=headers(transaction)
            )
            assert transaction["transaction_key"] not in status.text


@pytest.mark.asyncio
async def test_conflict_transfer_preserves_peer_changes_and_requires_a_new_review(host_setup):
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            prefix = f"/v2/projects/{host_setup.project.external_id}/git/transactions"
            source = await start(client, prefix)
            await write(
                client, prefix, source, "Intent.md", note("Intent", "old reviewed intention")
            )
            peer = await start(client, prefix)
            await write(client, prefix, peer, "Peer.md", note("Peer", "changed factual dependency"))
            committed = await client.post(
                f"{prefix}/{peer['transaction_id']}/commit",
                headers=headers(peer),
                json={"expected_revision": 1},
            )
            assert committed.status_code == 200
            conflict = await client.post(
                f"{prefix}/{source['transaction_id']}/commit",
                headers=headers(source),
                json={"expected_revision": 1},
            )
            assert conflict.status_code == 409 and "draft preserved" in conflict.text
            successor = await start(
                client,
                prefix,
                source_transaction_id=source["transaction_id"],
                source_key=source["transaction_key"],
                source_revision=1,
            )
            assert successor["base_commit"] == committed.json()["published_commit"]
            listing = await client.get(
                f"{prefix}/{successor['transaction_id']}/notes", headers=headers(successor)
            )
            assert [row["path"] for row in listing.json()["notes"]] == ["Peer.md"]
            await write(
                client, prefix, successor, "Intent.md", note("Intent", "new reviewed intention")
            )
            result = await client.post(
                f"{prefix}/{successor['transaction_id']}/commit",
                headers=headers(successor),
                json={"expected_revision": 1},
            )
            assert result.status_code == 200, result.text
            assert set(host_setup.repository.snapshot().files) == {"Peer.md", "Intent.md"}
            assert "new reviewed intention" in host_setup.repository.snapshot().files["Intent.md"]
            assert "changed factual dependency" in host_setup.repository.snapshot().files["Peer.md"]
            old = await client.get(
                f"{prefix}/{source['transaction_id']}/note",
                headers=headers(source),
                params={"identifier": "Intent.md"},
            )
            assert "old reviewed intention" in old.json()["note"]["content"]


@pytest.mark.asyncio
async def test_draft_cas_move_delete_and_rollback_never_publish(host_setup):
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            prefix = f"/v2/projects/{host_setup.project.external_id}/git/transactions"
            transaction = await start(client, prefix)
            path = f"{prefix}/{transaction['transaction_id']}"
            await write(client, prefix, transaction, "A.md", note("A", "first"))
            await write(
                client, prefix, transaction, "A.md", "second", revision=1, operation="append"
            )
            stale = await client.post(
                path + "/write",
                headers=headers(transaction),
                json={
                    "path": "A.md",
                    "content": "second",
                    "operation": "append",
                    "expected_revision": 1,
                },
            )
            assert stale.status_code == 409
            await write(
                client, prefix, transaction, "A.md", "before", revision=2, operation="prepend"
            )
            moved = await client.post(
                path + "/move",
                headers=headers(transaction),
                json={"path": "A.md", "destination": "moved/A.md", "expected_revision": 3},
            )
            assert moved.status_code == 200, moved.text
            read = await client.get(
                path + "/note", headers=headers(transaction), params={"identifier": "moved/A.md"}
            )
            assert read.json()["note"]["content"].count("second") == 1
            assert "test-project/a" in read.json()["note"]["content"]
            deleted = await client.post(
                path + "/delete",
                headers=headers(transaction),
                json={"path": "moved/A.md", "expected_revision": 4},
            )
            assert deleted.status_code == 200
            rolled = await client.post(
                path + "/rollback", headers=headers(transaction), json={"expected_revision": 5}
            )
            assert rolled.status_code == 200 and rolled.json()["state"] == "rolled_back"
            assert host_setup.repository.head() == transaction["base_commit"]


@pytest.mark.asyncio
async def test_api_lost_receipt_error_reports_uncertainty_and_recovers_without_republishing(
    host_setup, monkeypatch
):
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            project = host_setup.project.external_id
            prefix = f"/v2/projects/{project}/git/transactions"
            transaction = await start(client, prefix)
            await write(client, prefix, transaction, "A.md", note("A", "accepted once"))
            host = resolve_container().git_host
            assert host is not None
            store = host.runtime_for(project).transactions.repository
            save = store.save

            def failed(record):
                if isinstance(record.state, GitCommittedState):
                    raise OSError("Receipt temporarily unavailable")
                save(record)

            with monkeypatch.context() as patch:
                patch.setattr(store, "save", failed)
                lost = await client.post(
                    f"{prefix}/{transaction['transaction_id']}/commit",
                    headers=headers(transaction),
                    json={"expected_revision": 1},
                )
                assert lost.status_code == 503 and "inspect" in lost.text
            published = host_setup.repository.head()
            repeated = await client.post(
                f"{prefix}/{transaction['transaction_id']}/commit",
                headers=headers(transaction),
                json={"expected_revision": 1},
            )
            assert (
                repeated.json()["already_committed"]
                and repeated.json()["published_commit"] == published
            )
            await ready(client, project)


@pytest.mark.asyncio
async def test_normal_mcp_transaction_commands_are_routed_through_typed_api_client(host_setup):
    from basic_memory.mcp.server import mcp

    def payload(result):
        block = result.content[0]
        assert isinstance(block, TextContent)
        return json.loads(block.text)

    async with Client(mcp) as client:
        transaction = payload(
            await client.call_tool(
                "bm_transaction", {"command": "start", "project": "test-project"}
            )
        )
        common = {
            "project": "test-project",
            "transaction_id": transaction["transaction_id"],
            "transaction_key": transaction["transaction_key"],
        }
        created = payload(
            await client.call_tool(
                "bm_transaction",
                {
                    **common,
                    "command": "write",
                    "identifier": "MCP.md",
                    "content": note("MCP", "Private mcp bundle"),
                    "expected_revision": 0,
                },
            )
        )
        assert created["revision"] == 1
        assert "Private mcp bundle" in str(
            (
                await client.call_tool(
                    "bm_transaction", {**common, "command": "read", "identifier": "MCP.md"}
                )
            ).content
        )
        assert payload(await client.call_tool("bm_transaction", {**common, "command": "list"}))[
            "notes"
        ]
        assert (
            payload(await client.call_tool("bm_transaction", {**common, "command": "keep_alive"}))[
                "state"
            ]
            == "open"
        )
        text = await client.call_tool(
            "bm_transaction", {**common, "command": "status", "output_format": "text"}
        )
        assert "open" in str(text.content)
        await client.call_tool(
            "bm_transaction",
            {
                **common,
                "command": "move",
                "identifier": "MCP.md",
                "destination": "Moved.md",
                "expected_revision": 1,
            },
        )
        await client.call_tool(
            "bm_transaction",
            {**common, "command": "delete", "identifier": "Moved.md", "expected_revision": 2},
        )
        await client.call_tool(
            "bm_transaction",
            {
                **common,
                "command": "write",
                "identifier": "MCP.md",
                "content": note("MCP", "Published mcp bundle"),
                "expected_revision": 3,
            },
        )
        committed = payload(
            await client.call_tool(
                "bm_transaction", {**common, "command": "commit", "expected_revision": 4}
            )
        )
        assert committed["published_commit"] == host_setup.repository.head()
        second = payload(
            await client.call_tool(
                "bm_transaction", {"command": "start", "project": "test-project"}
            )
        )
        rolled = payload(
            await client.call_tool(
                "bm_transaction",
                {
                    "command": "rollback",
                    "project": "test-project",
                    "transaction_id": second["transaction_id"],
                    "transaction_key": second["transaction_key"],
                    "expected_revision": 0,
                },
            )
        )
        assert rolled["state"] == "rolled_back"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"command": "start"},
        {"command": "status", "project": "test-project"},
        {
            "command": "write",
            "project": "test-project",
            "transaction_id": str(uuid4()),
            "transaction_key": "x",
        },
        {
            "command": "read",
            "project": "test-project",
            "transaction_id": str(uuid4()),
            "transaction_key": "x",
        },
        {
            "command": "write",
            "project": "test-project",
            "transaction_id": str(uuid4()),
            "transaction_key": "x",
            "expected_revision": 0,
            "identifier": "A.md",
        },
        {
            "command": "move",
            "project": "test-project",
            "transaction_id": str(uuid4()),
            "transaction_key": "x",
            "expected_revision": 0,
            "identifier": "A.md",
        },
    ],
)
async def test_mcp_missing_transaction_arguments_are_refused_without_mutation(host_setup, payload):
    from basic_memory.mcp.server import mcp

    base = host_setup.repository.head()
    async with Client(mcp) as client:
        result = await client.call_tool("bm_transaction", payload, raise_on_error=False)
        assert result.is_error and "INVALID_ARGUMENTS" in str(result.content)
    assert host_setup.repository.head() == base


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", ["draft_before", "draft_after", "git_before", "git_after", "receipt_after"]
)
async def test_normal_api_process_death_preserves_draft_or_recovers_terminal_commit(
    host_setup, tmp_path, phase
):
    from basic_memory.api.app import app

    project = host_setup.project.external_id
    prefix = f"/v2/projects/{project}/git/transactions"
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            transaction = await start(client, prefix)
            await write(client, prefix, transaction, "A.md", note("A", "Crashtransactionprobe"))
    body = (
        {"expected_revision": 1, "path": "B.md", "content": note("B", "Private second")}
        if phase.startswith("draft_")
        else {"expected_revision": 1}
    )
    request = tmp_path / "crash-request.json"
    request.write_text(
        json.dumps(
            {
                "phase": phase,
                "project_id": project,
                "transaction_id": transaction["transaction_id"],
                "key": transaction["transaction_key"],
                "body": body,
            }
        )
    )
    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "test-int/fixtures/git_transaction_crash.py", str(request)],
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 83, child.stderr
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            status = await client.get(
                f"{prefix}/{transaction['transaction_id']}", headers=headers(transaction)
            )
            assert status.status_code == 200, status.text
            if phase in {"git_after", "receipt_after"}:
                assert status.json()["state"] == "committed"
                published = status.json()["published_commit"]
                repeated = await client.post(
                    f"{prefix}/{transaction['transaction_id']}/commit",
                    headers=headers(transaction),
                    json={"expected_revision": 1},
                )
                assert (
                    repeated.json()["already_committed"]
                    and repeated.json()["published_commit"] == published
                )
                await ready(client, project)
                assert host_setup.repository._git("rev-list", "--count", "main").strip() == b"2"
            else:
                assert status.json()["state"] == "open"
                assert status.json()["revision"] == (2 if phase == "draft_after" else 1)
                assert host_setup.repository.head() == transaction["base_commit"]
                listing = await client.get(
                    f"{prefix}/{transaction['transaction_id']}/notes", headers=headers(transaction)
                )
                assert {row["path"] for row in listing.json()["notes"]} == (
                    {"A.md", "B.md"} if phase == "draft_after" else {"A.md"}
                )


@pytest.mark.asyncio
async def test_cancelled_private_write_settles_before_releasing_runtime_ownership(
    host_setup, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.schemas.git_transactions import GitTransactionStart

    async with app.router.lifespan_context(app):
        host = resolve_container().git_host
        assert host is not None
        runtime = host.runtime_for(host_setup.project.external_id)
        service = runtime.transactions
        transaction = await service.start(GitTransactionStart())
        replace = service.repository.replace
        entered, release = Event(), Event()

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return replace(*args, **kwargs)

        monkeypatch.setattr(service.repository, "replace", blocked)
        task = asyncio.create_task(
            service.write(
                transaction.transaction_id,
                transaction.transaction_key,
                GitTransactionWrite(
                    path="A.md", content=note("A", "cancelled reply"), expected_revision=0
                ),
            )
        )
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and runtime.gate.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not runtime.gate.locked()
        assert (
            await service.status(transaction.transaction_id, transaction.transaction_key)
        ).revision == 1
        assert host_setup.repository.head() == transaction.base_commit


@pytest.mark.asyncio
async def test_api_rejects_invalid_private_changes_and_wrong_owner_without_publication(host_setup):
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            prefix = f"/v2/projects/{host_setup.project.external_id}/git/transactions"
            transaction = await start(client, prefix)
            path = f"{prefix}/{transaction['transaction_id']}"
            await write(client, prefix, transaction, "A.md", note("A", "unchanged"))
            cases = [
                (
                    "write",
                    {"path": "B.md", "content": "no frontmatter", "expected_revision": 1},
                    422,
                ),
                (
                    "write",
                    {
                        "path": "B.md",
                        "content": note("A", "duplicate permalink"),
                        "expected_revision": 1,
                    },
                    422,
                ),
                (
                    "write",
                    {
                        "path": "B.md",
                        "content": "---\ntitle: ''\ntype: note\npermalink: b\n---\n",
                        "expected_revision": 1,
                    },
                    422,
                ),
                (
                    "write",
                    {
                        "path": "B.md",
                        "content": "text",
                        "operation": "append",
                        "expected_revision": 1,
                    },
                    404,
                ),
                (
                    "write",
                    {
                        "path": "../private.md",
                        "content": note("B", "bad path"),
                        "expected_revision": 1,
                    },
                    422,
                ),
                (
                    "move",
                    {"path": "missing.md", "destination": "C.md", "expected_revision": 1},
                    404,
                ),
                ("move", {"path": "A.md", "destination": "A.md", "expected_revision": 1}, 409),
                ("delete", {"path": "missing.md", "expected_revision": 1}, 404),
                (
                    "commit",
                    {"expected_revision": 1, "message": "Basic-Memory-Transaction: forged"},
                    422,
                ),
            ]
            for action, body, expected in cases:
                response = await client.post(
                    path + "/" + action, headers=headers(transaction), json=body
                )
                assert response.status_code == expected, response.text
            missing = await client.get(
                path + "/note", headers=headers(transaction), params={"identifier": "missing.md"}
            )
            assert missing.status_code == 404
            wrong = await client.post(
                path + "/rollback",
                headers={"X-BM-Transaction-Key": "wrong"},
                json={"expected_revision": 1},
            )
            assert wrong.status_code == 404
            partial_source = await client.post(
                prefix, json={"source_transaction_id": transaction["transaction_id"]}
            )
            assert partial_source.status_code == 422
            assert host_setup.repository.head() == transaction["base_commit"]


@pytest.mark.asyncio
async def test_legacy_host_has_no_explicit_transaction_service():
    from basic_memory.api.v2.routers.git_transaction_router import get_git_transactions
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        await get_git_transactions(None)
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_recreating_a_published_path_cannot_silently_replace_its_uuid(host_setup):
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            project = host_setup.project.external_id
            original = await client.post(
                f"/v2/projects/{project}/knowledge/write",
                json={"note": {"title": "A", "content": "Original"}},
            )
            assert original.status_code == 200, original.text
            prefix = f"/v2/projects/{project}/git/transactions"
            transaction = await start(client, prefix)
            path = f"{prefix}/{transaction['transaction_id']}"
            deleted = await client.post(
                path + "/delete",
                headers=headers(transaction),
                json={"path": "A.md", "expected_revision": 0},
            )
            assert deleted.status_code == 200
            recreated = await client.post(
                path + "/write",
                headers=headers(transaction),
                json={"path": "A.md", "expected_revision": 1, "content": note("A", "New identity")},
            )
            assert recreated.status_code == 422 and "different identity" in recreated.text
            assert host_setup.repository.head() == transaction["base_commit"]


@pytest.mark.asyncio
async def test_cancelled_explicit_commit_keeps_published_evidence_and_wakes_recovery(
    host_setup, monkeypatch
):
    from basic_memory.api.app import app
    from basic_memory.schemas.git_transactions import GitTransactionStart, GitTransactionCommit

    async with app.router.lifespan_context(app):
        host = resolve_container().git_host
        assert host is not None
        runtime = host.runtime_for(host_setup.project.external_id)
        service = runtime.transactions
        transaction = await service.start(GitTransactionStart())
        await service.write(
            transaction.transaction_id,
            transaction.transaction_key,
            GitTransactionWrite(
                path="A.md", content=note("A", "Cancelledcommitprobe"), expected_revision=0
            ),
        )
        publish = host_setup.repository.publish
        entered, release = Event(), Event()

        def blocked(candidate):
            result = publish(candidate)
            entered.set()
            assert release.wait(10)
            return result

        monkeypatch.setattr(runtime.pipeline.repository, "publish", blocked)
        task = asyncio.create_task(
            service.commit(
                transaction.transaction_id,
                transaction.transaction_key,
                GitTransactionCommit(expected_revision=1),
            )
        )
        assert await asyncio.to_thread(entered.wait, 10)
        published = host_setup.repository.head()
        task.cancel()
        await asyncio.sleep(0)
        assert runtime.gate.locked() and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        status = await service.status(transaction.transaction_id, transaction.transaction_key)
        assert status.published_commit == published and status.state == "committed"
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            await ready(client, host_setup.project.external_id)


@pytest.mark.asyncio
async def test_commit_in_another_managed_project_does_not_conflict_with_open_transaction(
    host_setup, config_manager
):
    from basic_memory.api.app import app
    from basic_memory.config import ProjectEntry
    from basic_memory.repository.project_repository import ProjectRepository
    from basic_memory.repository.git_project_repository import GitProjectRepository
    from uuid import UUID

    root = host_setup.repository.path.parent / "second"
    root.mkdir()
    config = host_setup.config
    config.projects["second"] = ProjectEntry(path=str(root))
    _, factory = await db.get_or_create_db(config.database_path)
    async with db.scoped_session(factory) as session:
        second = await ProjectRepository().create(
            session, {"name": "second", "path": str(root), "is_active": True}
        )
    second_repository = GitProjectRepository.create(
        root.parent / "second.git", project_external_id=UUID(second.external_id)
    )
    config.git_repositories[UUID(second.external_id)] = second_repository.path
    config_manager.save_config(config)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            first_path = f"/v2/projects/{host_setup.project.external_id}/git/transactions"
            second_path = f"/v2/projects/{second.external_id}/git/transactions"
            first = await start(client, first_path)
            other = await start(client, second_path)
            await write(client, second_path, other, "B.md", note("B", "Independent second"))
            result = await client.post(
                f"{second_path}/{other['transaction_id']}/commit",
                headers=headers(other),
                json={"expected_revision": 1},
            )
            assert result.status_code == 200
            assert host_setup.repository.head() == first["base_commit"]
            await write(client, first_path, first, "A.md", note("A", "Independent first"))
            committed = await client.post(
                f"{first_path}/{first['transaction_id']}/commit",
                headers=headers(first),
                json={"expected_revision": 1},
            )
            assert committed.status_code == 200, committed.text
            assert set(second_repository.snapshot().files) == {"B.md"}
            assert set(host_setup.repository.snapshot().files) == {"A.md"}


@pytest.mark.asyncio
async def test_api_lease_configuration_and_passive_expiry_preserve_then_collect_private_draft(
    host_setup, config_manager
):
    from basic_memory.api.app import app

    host_setup.config.git_transaction_idle_seconds = 5
    host_setup.config.git_transaction_retention_seconds = 10
    config_manager.save_config(host_setup.config)
    async with app.router.lifespan_context(app):
        host = resolve_container().git_host
        assert host is not None
        clock = [datetime.now(UTC)]
        store = host.runtime_for(host_setup.project.external_id).transactions.repository
        store.clock = lambda: clock[0]
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            prefix = f"/v2/projects/{host_setup.project.external_id}/git/transactions"
            transaction = await start(client, prefix)
            path = f"{prefix}/{transaction['transaction_id']}"
            await write(client, prefix, transaction, "A.md", note("A", "expiry preview"))
            clock[0] += timedelta(seconds=4)
            status = await client.get(path, headers=headers(transaction))
            assert status.json()["expires_at"] == transaction["expires_at"]
            renewed = await client.post(path + "/keep-alive", headers=headers(transaction))
            assert datetime.fromisoformat(renewed.json()["expires_at"]) == clock[0] + timedelta(
                seconds=5
            )
            clock[0] += timedelta(seconds=5)
            expired = await client.get(path, headers=headers(transaction))
            assert expired.json()["state"] == "expired" and expired.json()["draft_available"]
            preview = await client.get(
                path + "/note", headers=headers(transaction), params={"identifier": "A.md"}
            )
            assert "expiry preview" in preview.json()["note"]["content"]
            refused = await client.post(
                path + "/commit", headers=headers(transaction), json={"expected_revision": 1}
            )
            assert refused.status_code == 409
            clock[0] += timedelta(seconds=10)
            cleaned = await client.get(path, headers=headers(transaction))
            assert (
                cleaned.json()["state"] == "expired_cleaned"
                and not cleaned.json()["draft_available"]
            )
            assert host_setup.repository.head() == transaction["base_commit"]


@pytest.mark.asyncio
async def test_a_draft_note_keeps_its_permalink(host_setup):
    from basic_memory.api.app import app

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            prefix = f"/v2/projects/{host_setup.project.external_id}/git/transactions"
            transaction = await start(client, prefix)
            await write(client, prefix, transaction, "A.md", note("A", "first"))

            other = note("A", "second").replace("test-project/a", "somewhere/else")
            refused = await client.post(
                f"{prefix}/{transaction['transaction_id']}/write",
                headers=headers(transaction),
                json={
                    "path": "A.md",
                    "content": other,
                    "expected_revision": 1,
                    "operation": "replace",
                },
            )
            assert refused.status_code == 400
            assert "cannot be changed by writing or editing" in refused.text

            # The refusal changed nothing: the same revision still accepts the same permalink.
            await write(client, prefix, transaction, "A.md", note("A", "third"), revision=1)
