"""Real Git draft storage, terminal recovery, expiry and protected conflict sources."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import pytest

from basic_memory.repository.git_project_repository import GitProjectRepository, GitPublicationError
from dataclasses import replace as replace_value
from basic_memory.repository.git_transaction_repository import (
    GitTransactionError,
    GitTransactionRepository,
    draft_candidate,
)
from basic_memory.schemas.git_transactions import GitTransactionStart, GitCommittedState


@dataclass
class Clock:
    now: datetime = datetime(2030, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def store(tmp_path):
    identity = uuid4()
    repository = GitProjectRepository.create(tmp_path / "notes.git", project_external_id=identity)
    return GitTransactionRepository(repository, identity, clock=Clock())


def replace(store, started, files, *, revision=0):
    identities = {path: uuid4() for path in files}
    return store.replace(
        started.transaction_id,
        started.transaction_key,
        expected_revision=revision,
        files=files,
        note_ids=identities,
    )


def commit(store, started, *, revision=1):
    return store.commit(
        started.transaction_id,
        started.transaction_key,
        expected_revision=revision,
        message="Reviewed bundle",
    )


def test_private_fixed_tree_and_restart_require_the_owner_key(store):
    base = store.repository.head()
    started = store.start(GitTransactionStart())
    assert started.base_commit == base and started.revision == 0
    replace(store, started, {"A.md": "Draft A", "B.md": "Draft B"})
    record, snapshot = store.read(started.transaction_id, started.transaction_key)
    assert snapshot.files == {"A.md": "Draft A", "B.md": "Draft B"}
    assert store.read(started.transaction_id, started.transaction_key, view="base")[1].files == {}
    assert store.repository.snapshot().files == {} and store.repository.head() == base
    with pytest.raises(GitPublicationError):
        store.repository.snapshot(snapshot.commit_id)
    restarted = GitTransactionRepository(store.repository, store.project_id, clock=store.clock)
    assert (
        restarted.read(started.transaction_id, started.transaction_key)[1].files == snapshot.files
    )
    data = (store.path / f"{started.transaction_id}.json").read_text()
    assert started.transaction_key not in data and "Draft A" not in data
    assert not (store.path.stat().st_mode & 0o077)
    with pytest.raises(GitTransactionError) as error:
        store.status(started.transaction_id, "wrong-owner")
    assert error.value.status_code == 404
    with pytest.raises(GitTransactionError):
        store.status(uuid4(), started.transaction_key)
    assert record.base_commit == base


def test_bundle_commit_and_retry_after_later_history_and_cleanup(store):
    started = store.start(GitTransactionStart())
    replace(store, started, {"A.md": "A", "B.md": "B"})
    published = commit(store, started)
    assert published.state == "committed" and not published.already_committed
    assert store.repository.snapshot().files == {"A.md": "A", "B.md": "B"}
    other = store.start(GitTransactionStart())
    snapshot = store.repository.snapshot()
    store.replace(
        other.transaction_id,
        other.transaction_key,
        expected_revision=0,
        files={**snapshot.files, "C.md": "later publication"},
        note_ids={**snapshot.manifest.note_ids, "C.md": uuid4()},
    )
    later = commit(store, other)
    assert later.published_commit != published.published_commit
    store.maintain()
    assert not store.repository._git(
        "for-each-ref", f"refs/bm/candidates/{started.transaction_id}/"
    )
    repeated = commit(store, started, revision=0)
    assert repeated.already_committed and repeated.published_commit == published.published_commit
    with pytest.raises(GitTransactionError, match="cannot rewind"):
        store.rollback(started.transaction_id, started.transaction_key, expected_revision=1)
    with pytest.raises(GitTransactionError, match="no longer has"):
        draft_candidate(store.load(started.transaction_id))
    with pytest.raises(GitTransactionError, match="no longer available"):
        store.read(started.transaction_id, started.transaction_key)
    with pytest.raises(GitTransactionError, match="lease has ended"):
        store.status(started.transaction_id, started.transaction_key, keep_alive=True)


def test_noop_retains_a_terminal_receipt_without_an_empty_commit(store):
    started = store.start(GitTransactionStart())
    base = store.repository.head()
    result = commit(store, started, revision=0)
    assert result.published_commit == base and store.repository.head() == base
    assert commit(store, started, revision=0).already_committed


def test_passive_status_does_not_renew_and_expiry_never_publishes(store):
    started = store.start(GitTransactionStart())
    replace(store, started, {"A.md": "Preserved expired draft"})
    expires = store.status(started.transaction_id, started.transaction_key).expires_at
    clock = store.clock
    clock.advance(1700)
    assert store.status(started.transaction_id, started.transaction_key).expires_at == expires
    renewed = store.status(started.transaction_id, started.transaction_key, keep_alive=True)
    assert renewed.expires_at > expires
    clock.advance(1800)
    assert store.status(started.transaction_id, started.transaction_key).state == "expired"
    assert store.read(started.transaction_id, started.transaction_key)[1].files["A.md"]
    with pytest.raises(GitTransactionError, match="lease has ended"):
        store.status(started.transaction_id, started.transaction_key, keep_alive=True)
    with pytest.raises(GitTransactionError, match="expired"):
        commit(store, started)
    clock.advance(604800)
    cleaned = store.status(started.transaction_id, started.transaction_key)
    assert cleaned.state == "expired_cleaned" and not cleaned.draft_available
    assert store.repository.head() == started.base_commit
    with pytest.raises(GitTransactionError, match="unavailable"):
        store.start(
            GitTransactionStart(
                source_transaction_id=started.transaction_id,
                source_key=started.transaction_key,
                source_revision=1,
            )
        )


def test_conflict_freezes_source_and_successor_starts_on_current_without_copying(store):
    source = store.start(GitTransactionStart())
    replace(store, source, {"Intent.md": "old intent"})
    peer = store.start(GitTransactionStart())
    replace(store, peer, {"Peer.md": "new peer"})
    peer_commit = commit(store, peer).published_commit
    with pytest.raises(GitTransactionError, match="draft preserved"):
        commit(store, source)
    assert store.status(source.transaction_id, source.transaction_key).state == "conflicted"
    with pytest.raises(GitTransactionError, match="conflicted"):
        replace(store, source, {"Late.md": "not accepted"}, revision=1)
    successor = store.start(
        GitTransactionStart(
            source_transaction_id=source.transaction_id,
            source_key=source.transaction_key,
            source_revision=1,
        )
    )
    assert successor.base_commit == peer_commit and successor.source_revision == 1
    assert store.read(successor.transaction_id, successor.transaction_key)[1].files == {
        "Peer.md": "new peer"
    }
    assert store.read(source.transaction_id, source.transaction_key, view="base")[1].files == {}
    assert store.read(source.transaction_id, source.transaction_key)[1].files == {
        "Intent.md": "old intent"
    }
    with pytest.raises(GitTransactionError, match="successor"):
        store.rollback(source.transaction_id, source.transaction_key, expected_revision=1)


def test_transfer_pins_expired_source_past_retention_then_releases_it(store):
    source = store.start(GitTransactionStart())
    replace(store, source, {"Intent.md": "retained"})
    store.clock.advance(1800 + 604799)
    successor = store.start(
        GitTransactionStart(
            source_transaction_id=source.transaction_id,
            source_key=source.transaction_key,
            source_revision=1,
        )
    )
    store.clock.advance(2)
    store.maintain()
    assert store.read(source.transaction_id, source.transaction_key)[1].files == {
        "Intent.md": "retained"
    }
    assert commit(store, successor, revision=0).state == "committed"
    store.maintain()
    assert store.status(source.transaction_id, source.transaction_key).state == "expired_cleaned"


def test_transfer_freezes_a_live_source_and_rollback_releases_its_pin(store):
    source = store.start(GitTransactionStart())
    successor = store.start(
        GitTransactionStart(
            source_transaction_id=source.transaction_id,
            source_key=source.transaction_key,
            source_revision=0,
        )
    )
    assert store.status(source.transaction_id, source.transaction_key).state == "frozen"
    store.rollback(successor.transaction_id, successor.transaction_key, expected_revision=0)
    assert (
        store.rollback(source.transaction_id, source.transaction_key, expected_revision=0).state
        == "rolled_back"
    )


def test_draft_revision_conflict_does_not_overwrite_accepted_content(store):
    started = store.start(GitTransactionStart())
    replace(store, started, {"A.md": "first"})
    with pytest.raises(GitTransactionError, match="revision is 1"):
        replace(store, started, {"A.md": "second"})
    assert store.read(started.transaction_id, started.transaction_key)[1].files == {"A.md": "first"}


def test_lost_post_publication_metadata_write_is_recovered_before_expiry(store, monkeypatch):
    started = store.start(GitTransactionStart())
    replace(store, started, {"A.md": "published"})
    save = store.save

    def failed(record):
        if isinstance(record.state, GitCommittedState):
            raise OSError("lost receipt write")
        save(record)

    with monkeypatch.context() as patch:
        patch.setattr(store, "save", failed)
        with pytest.raises(OSError):
            commit(store, started)
    published = store.repository.head()
    store.clock.advance(1800 + 604800 + 1)
    assert (
        store.status(started.transaction_id, started.transaction_key).published_commit == published
    )
    assert commit(store, started).published_commit == published


def test_running_commit_is_not_expired_or_collected_concurrently(store, monkeypatch):
    started = store.start(GitTransactionStart())
    replace(store, started, {"A.md": "published after lease boundary"})
    entered, release = Event(), Event()
    publish = store.repository.publish

    def blocked(candidate):
        entered.set()
        assert release.wait(10)
        return publish(candidate)

    monkeypatch.setattr(store.repository, "publish", blocked)
    with ThreadPoolExecutor(max_workers=2) as executor:
        publishing = executor.submit(commit, store, started)
        assert entered.wait(10)
        store.clock.advance(1800 + 604800 + 1)
        cleanup = executor.submit(store.maintain)
        release.set()
        assert publishing.result(timeout=10).state == "committed"
        cleanup.result(timeout=10)
    assert store.status(started.transaction_id, started.transaction_key).state == "committed"


@pytest.mark.parametrize(
    "fault",
    ["record_symlink", "directory_symlink", "lock_symlink", "record_identity", "missing_source"],
)
def test_untrusted_operational_metadata_is_refused(store, tmp_path, fault):
    started = store.start(GitTransactionStart())
    path = store.path / f"{started.transaction_id}.json"
    if fault == "record_symlink":
        saved = tmp_path / "saved.json"
        path.replace(saved)
        path.symlink_to(saved)
    elif fault == "directory_symlink":
        saved = tmp_path / "saved"
        store.path.replace(saved)
        store.path.symlink_to(saved, target_is_directory=True)
    elif fault == "lock_symlink":
        store.lock_path.unlink()
        store.lock_path.symlink_to(tmp_path / "foreign.lock")
    else:
        data = json.loads(path.read_text())
        if fault == "record_identity":
            data["project_id"] = str(uuid4())
        else:
            data["source_transaction_id"] = str(uuid4())
        path.write_text(json.dumps(data))
    with pytest.raises(GitTransactionError):
        store.status(started.transaction_id, started.transaction_key)


@pytest.mark.parametrize("idle,retention", [(0, 1), (1, -1)])
def test_invalid_lease_policy_is_refused(store, idle, retention):
    with pytest.raises(ValueError):
        GitTransactionRepository(
            store.repository, store.project_id, idle_seconds=idle, retention_seconds=retention
        )


def test_private_candidate_reference_and_reserved_message_are_validated(store):
    started = store.start(GitTransactionStart())
    candidate = draft_candidate(store.load(started.transaction_id))
    with pytest.raises(GitPublicationError, match="reference"):
        store.repository.candidate_snapshot(
            replace_value(candidate, candidate_ref="refs/heads/main")
        )
    with pytest.raises(ValueError, match="reserved"):
        store.repository.prepare(
            transaction_id=uuid4(),
            base_commit=store.repository.head(),
            files={},
            note_ids={},
            message="Basic-Memory-Transaction: forged",
        )


def test_failed_pointer_replace_preserves_the_previous_accepted_revision(store, monkeypatch):
    import os

    started = store.start(GitTransactionStart())

    def failed(*args):
        raise OSError("Atomic pointer replacement interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failed)
        with pytest.raises(OSError):
            replace(store, started, {"A.md": "not acknowledged"})
    assert store.status(started.transaction_id, started.transaction_key).revision == 0
    assert not list(store.path.glob(".draft-*"))
    assert store.read(started.transaction_id, started.transaction_key)[1].files == {}


def test_another_change_during_transfer_conflicts_again_and_keeps_both_drafts(store):
    source = store.start(GitTransactionStart())
    replace(store, source, {"A.md": "old intended content"})
    successor = store.start(
        GitTransactionStart(
            source_transaction_id=source.transaction_id,
            source_key=source.transaction_key,
            source_revision=1,
        )
    )
    replace(store, successor, {"A.md": "rechecked content"})
    peer = store.start(GitTransactionStart())
    replace(store, peer, {"Peer.md": "new factual basis"})
    commit(store, peer)
    with pytest.raises(GitTransactionError, match="draft preserved"):
        commit(store, successor)
    assert store.read(source.transaction_id, source.transaction_key)[1].files == {
        "A.md": "old intended content"
    }
    assert store.read(successor.transaction_id, successor.transaction_key)[1].files == {
        "A.md": "rechecked content"
    }


def test_source_freeze_and_concurrent_write_do_not_lose_an_accepted_change(store):
    source = store.start(GitTransactionStart())

    def draft_write():
        try:
            return replace(store, source, {"A.md": "accepted if first"})
        except GitTransactionError as error:
            return error

    def transfer():
        try:
            return store.start(
                GitTransactionStart(
                    source_transaction_id=source.transaction_id,
                    source_key=source.transaction_key,
                    source_revision=0,
                )
            )
        except GitTransactionError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        writing, starting = executor.submit(draft_write), executor.submit(transfer)
        write_result, transfer_result = writing.result(timeout=10), starting.result(timeout=10)
    assert isinstance(write_result, GitTransactionError) != isinstance(
        transfer_result, GitTransactionError
    )
    record, snapshot = store.read(source.transaction_id, source.transaction_key)
    if not isinstance(write_result, GitTransactionError):
        assert record.revision == 1 and snapshot.files == {"A.md": "accepted if first"}
    else:
        assert record.state.kind == "frozen" and snapshot.files == {}


def test_maintenance_lists_candidate_refs_once_however_many_records_are_kept(store, monkeypatch):
    """Committed records stay for good; each call must not cost one Git call per record."""
    for number in range(5):
        started = store.start(GitTransactionStart())
        replace(store, started, {f"n{number}.md": "---\ntitle: N\ntype: note\npermalink: n\n---\n"})
        commit(store, started)
    # The last commit's candidate refs go on the next maintenance; after that, none remain.
    store.maintain()
    calls = []
    remove = store.repository.remove_candidates
    listed = store.repository.candidate_transaction_ids

    monkeypatch.setattr(
        store.repository, "remove_candidates", lambda i: calls.append(i) or remove(i)
    )
    monkeypatch.setattr(
        store.repository, "candidate_transaction_ids", lambda: calls.append("list") or listed()
    )
    store.maintain()
    assert calls == ["list"]
    # A rolled-back draft still holding refs is cleaned in the same pass.
    started = store.start(GitTransactionStart())
    calls.clear()
    store.rollback(started.transaction_id, started.transaction_key, expected_revision=0)
    assert store.repository.candidate_transaction_ids() == frozenset()
