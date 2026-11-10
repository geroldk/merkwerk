"""Real local Git peers: branch-only transport, CAS and private observation durability."""

import subprocess
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from basic_memory.repository.git_project_repository import (
    GitProjectRepository,
    GitPublicationError,
    GitPublicationConflict,
    PUBLISHED_REF,
)
from basic_memory.repository.git_sync_repository import GitSyncRepository
from basic_memory.schemas.git_sync import GitSyncRemote, SyncRunning


def git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], capture_output=True, check=True, timeout=30)
    return result.stdout


def publish(repository, text="one"):
    base = repository.snapshot()
    prepared = repository.prepare(
        transaction_id=uuid4(),
        base_commit=base.commit_id,
        files={"note.md": text},
        note_ids={"note.md": base.manifest.note_ids.get("note.md", uuid4())},
    )
    return repository.publish(prepared).commit_id


def clone(repository, path):
    git("clone", "--bare", "--no-local", str(repository.path), str(path))
    return GitProjectRepository(path)


def test_branch_only_transport_and_ancestry(tmp_path):
    local = GitProjectRepository.create(tmp_path / "local.git")
    initial = local.head()
    remote = tmp_path / "remote.git"
    git("init", "--bare", "--initial-branch=main", str(remote))
    assert local.fetch_published(str(remote), uuid4()) is None
    one = publish(local)
    draft = local.prepare(
        transaction_id=uuid4(),
        base_commit=one,
        files={"private.md": "private"},
        note_ids={"private.md": uuid4()},
    )
    local.push_published(str(remote), one)
    assert git("--git-dir", str(remote), "for-each-ref", "--format=%(refname)").splitlines() == [
        PUBLISHED_REF.encode()
    ]
    absent = subprocess.run(
        ["git", "--git-dir", str(remote), "cat-file", "-e", draft.commit_id], capture_output=True
    )
    assert absent.returncode != 0
    peer = clone(local, tmp_path / "peer.git")
    two = publish(peer, "two")
    peer.push_published(str(remote), two)
    attempt = uuid4()
    assert local.fetch_published(str(remote), attempt) == two
    assert local.head() == one
    assert {s.commit_id for s in local.incoming_snapshots(attempt, two)} == {two}
    assert local.relationship(one, None) == "unobserved"
    assert local.relationship(one, one) == "equal"
    assert local.relationship(one, two) == "behind"
    assert local.relationship(two, initial) == "ahead"
    local.publish_incoming(attempt, two, one)
    assert local.head() == two
    assert local.incoming_snapshots(attempt, two)[0].commit_id == two
    with pytest.raises(ValueError, match="SHA-1"):
        local.is_ancestor("main", two)
    with pytest.raises(GitPublicationError, match="ancestry"):
        local.is_ancestor("1" * 40, two)
    with pytest.raises(GitPublicationError, match="published ancestry"):
        local.push_published(str(remote), draft.commit_id)


def test_divergence_and_publication_guards(tmp_path):
    local = GitProjectRepository.create(tmp_path / "local.git")
    peer = clone(local, tmp_path / "peer.git")
    initial = local.head()
    other = publish(peer, "peer")
    attempt = uuid4()
    assert local.fetch_published(str(peer.path), attempt) == other
    local_tip = publish(local, "local")
    assert local.relationship(local_tip, other) == "diverged"
    with pytest.raises(GitPublicationError, match="fast-forward"):
        local.publish_incoming(attempt, other, local_tip)
    with pytest.raises(GitPublicationConflict):
        local.publish_incoming(attempt, other, initial)
    local._git("update-ref", f"refs/bm/incoming/{attempt}", initial)
    with pytest.raises(GitPublicationError, match="before validation"):
        local.incoming_snapshots(attempt, other)
    with pytest.raises(GitPublicationError, match="before publication"):
        peer._git("update-ref", f"refs/bm/incoming/{attempt}", initial)
        peer.publish_incoming(attempt, other, other)
    with pytest.raises(GitPublicationError, match="transfer failed"):
        local.push_published(str(peer.path), local_tip)
    with pytest.raises(GitPublicationError, match="transfer failed"):
        local.fetch_published(str(tmp_path / "offline"), uuid4())
    assert local.head() == local_tip and peer.head() == other


@pytest.mark.parametrize(
    "endpoint", ["/tmp/peer.git", "ssh://git@nuc1/project.git", "https://example.org/project.git"]
)
def test_remote_endpoints(endpoint):
    assert GitSyncRemote(url=endpoint).url == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "relative",
        "-upload-pack=evil",
        "ext::evil",
        "file:///tmp/p",
        "http://x/p",
        "https://u:p@x/p",
        "ssh://u:p@x/p",
        "ssh://x/p?q=secret",
        "https://x/p#secret",
        "https://u@x/p",
        "/tmp/with\nnewline",
        "ssh:///path",
    ],
)
def test_refuse_ambiguous_or_secret_endpoints(endpoint):
    with pytest.raises(ValueError):
        GitSyncRemote(url=endpoint)


def test_private_observations_are_atomic_and_peer_bound(tmp_path, monkeypatch):
    local = GitProjectRepository.create(tmp_path / "local.git")
    project = local.snapshot().manifest.project_external_id
    remote = GitSyncRemote(url="/tmp/peer.git")
    records = GitSyncRepository(local, project, remote)
    assert records.load().attempt is None
    running = records.load().model_copy(
        update={"attempt": SyncRunning(attempt_id=uuid4(), started_at=datetime.now(UTC))}
    )
    records.save(running)
    assert records.load() == running
    assert records.path.stat().st_mode & 0o777 == 0o600
    assert (
        GitSyncRepository(local, project, GitSyncRemote(url="/tmp/other.git")).load().attempt
        is None
    )
    with pytest.raises(ValueError, match="another project"):
        GitSyncRepository(local, uuid4(), remote).load()
    import basic_memory.repository.git_sync_repository as module

    replace = module.os.replace

    def fail(*args):
        raise OSError("injected pointer failure")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        records.save(running)
    monkeypatch.setattr(module.os, "replace", replace)
    assert records.load() == running
    assert not list(local.path.glob(".sync-*"))
    destination = tmp_path / "untouched"
    destination.write_text("untouched")
    records.path.unlink()
    records.path.symlink_to(destination)
    with pytest.raises(ValueError, match="symlink"):
        records.load()
    with pytest.raises(ValueError, match="symlink"):
        records.save(running)
    assert destination.read_text() == "untouched"


def test_server_receive_policy_still_rejects_a_push(tmp_path):
    local = GitProjectRepository.create(tmp_path / "local.git")
    peer = clone(local, tmp_path / "peer.git")
    hook = peer.path / "hooks/pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o700)
    initial = peer.head()
    head = publish(local)
    with pytest.raises(GitPublicationError):
        local.push_published(str(peer.path), head)
    assert peer.head() == initial


@pytest.mark.asyncio
async def test_note_ownership_refuses_only_real_duplicates(test_project, session_maker, tmp_path):
    """Without the host view any foreign UUID is refused; with it, a proven move passes (I1)."""
    from uuid import UUID

    from basic_memory.models import Entity, Project
    from basic_memory.repository.git_host_identities import GitHostIdentities, GitHostProject

    moved, twin = uuid4(), uuid4()
    async with session_maker() as session, session.begin():
        target = Project(name="target", path=str(tmp_path / "target"), is_active=True)
        session.add(target)
        await session.flush()
        target_id, target_uuid = target.id, UUID(target.external_id)
        for note in (moved, twin):
            session.add(
                Entity(
                    project_id=test_project.id,
                    external_id=str(note),
                    title=str(note),
                    note_type="note",
                    content_type="text/markdown",
                    file_path=f"{note}.md",
                )
            )
    source_uuid = UUID(test_project.external_id)
    source = GitProjectRepository.create(tmp_path / "source.git", project_external_id=source_uuid)
    source.publish(
        source.prepare(
            transaction_id=uuid4(),
            base_commit=source.head(),
            files={"moved.md": "m", "twin.md": "t"},
            note_ids={"moved.md": moved, "twin.md": twin},
        )
    )
    incoming = GitProjectRepository.create(tmp_path / "target.git", project_external_id=target_uuid)
    identities = GitHostIdentities(
        [
            GitHostProject(test_project.id, source_uuid, test_project.name, source),
            GitHostProject(target_id, target_uuid, "target", incoming),
        ]
    )
    records = GitSyncRepository(incoming, target_uuid, None)

    async def check(note_ids: dict[str, object], moved_notes=None, *, host=True):
        tip = incoming.prepare(
            transaction_id=uuid4(),
            base_commit=incoming.head(),
            files={path: path for path in note_ids},
            note_ids=note_ids,  # ty: ignore[invalid-argument-type]
            moved_notes=moved_notes,
        ).commit_id
        async with session_maker() as session:
            await records.require_note_ownership(
                session,
                target_id,
                incoming.read_snapshot(tip).manifest,
                identities=identities if host else None,
                history=(incoming, tip) if host else None,
            )

    # The target commit of a move the source has not caught up with: let through.
    await check({"moved.md": moved}, {moved: source_uuid})
    # Without the host view the old rule refuses it, and passes a fresh note.
    with pytest.raises(ValueError, match="already owned by another project"):
        await check({"moved.md": moved}, {moved: source_uuid}, host=False)
    await check({"fresh.md": uuid4()}, host=False)
    # A twin without a move proof is refused by name.
    with pytest.raises(ValueError) as refused:
        await check({"twin.md": twin})
    assert f"{twin} (owned by project '{test_project.name}'" in str(refused.value)
    assert "incoming for project 'target'" in str(refused.value)
