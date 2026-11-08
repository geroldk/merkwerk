"""Canonical history, private transaction evidence and safe independent targets."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import warnings
from zipfile import ZipFile

from filelock import FileLock, Timeout
import pytest

from basic_memory.config import BasicMemoryConfig, ProjectEntry
from basic_memory.index.git_project_ownership import GitLocalBinding
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_transaction_repository import GitTransactionRepository
from basic_memory.repository.git_transaction_repository import draft_candidate
from basic_memory.schemas.git_transactions import GitTransactionStart
from basic_memory.schemas.git_sync import GitSyncRecord
from basic_memory.services import git_backup
from basic_memory.services.git_backup import create_git_backup, restore_git_backup, write_private


def note(title: str, body: str = "") -> str:
    return f"---\ntitle: {title}\ntype: note\npermalink: {title.lower()}\n---\n\n{body}"


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source-notes"
    (root / ".basic-memory").mkdir(parents=True)
    repository = GitProjectRepository.create(tmp_path / "source.git")
    identity = repository.snapshot().manifest.project_external_id
    binding = GitLocalBinding(
        project_external_id=identity,
        project_root=str(root),
        repository_path=str(repository.path),
    )
    write_private(repository.path / "bm-local-binding.json", binding.model_dump_json().encode())
    write_private(root / ".basic-memory/git-binding.json", binding.model_dump_json().encode())
    config = BasicMemoryConfig(
        projects={"Research": ProjectEntry(path=str(root))},
        git_repositories={identity: repository.path},
        default_project="Research",
        auto_update=False,
    )
    path = tmp_path / "source-config/config.json"
    write_private(path, config.model_dump_json().encode())
    return SimpleNamespace(root=root, repository=repository, identity=identity, config=path)


def repack(
    path: Path, *, replace: dict[str, bytes], omit: set[str] | frozenset[str] = frozenset()
) -> None:
    with ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist() if name not in omit}
    files.update(replace)
    with ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def add_inventory_member(backup: Path, name: str, content: bytes) -> None:
    with ZipFile(backup) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["files"][name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    repack(backup, replace={name: content, "manifest.json": json.dumps(manifest).encode()})


def test_history_drafts_source_pins_noop_receipts_and_working_edits_survive(source, tmp_path):
    records = GitTransactionRepository(source.repository, source.identity)
    committed = records.start(GitTransactionStart())
    text, identity = note("Published", "- [fact] Original\n"), uuid4()
    records.replace(
        committed.transaction_id,
        committed.transaction_key,
        expected_revision=0,
        files={"published.md": text},
        note_ids={"published.md": identity},
    )
    published = records.commit(
        committed.transaction_id,
        committed.transaction_key,
        expected_revision=1,
        message="Published",
    )
    noop = records.start(GitTransactionStart())
    records.commit(noop.transaction_id, noop.transaction_key, expected_revision=0, message="Noop")
    draft = records.start(GitTransactionStart())
    records.replace(
        draft.transaction_id,
        draft.transaction_key,
        expected_revision=0,
        files={"published.md": text, "private.md": note("Private", "Unpublished\n")},
        note_ids={"published.md": identity, "private.md": uuid4()},
    )
    successor = records.start(
        GitTransactionStart(
            source_transaction_id=draft.transaction_id,
            source_key=draft.transaction_key,
            source_revision=1,
        )
    )
    (source.root / "published.md").write_text(text + "Unpublished editor change\n")
    (source.repository.path / "bm-sync.json").write_text(
        GitSyncRecord(
            project_id=source.identity, peer_fingerprint="preserved-peer"
        ).model_dump_json()
    )
    (source.repository.path / "bm-adoption.json").write_text('{"audit": "adopted"}')
    backup, restored = tmp_path / "backup.bm.zip", tmp_path / "restored"
    # A profile with a peer and automatic merging must restore with both switched off.
    with_peer = BasicMemoryConfig.model_validate_json(source.config.read_bytes())
    with_peer.git_sync_remotes = {source.identity: {"url": str(tmp_path / "peer.git")}}
    with_peer.git_sync_merge = {source.identity: "auto"}
    source.config.write_bytes(with_peer.model_dump_json().encode())
    manifest = create_git_backup(source.config, backup)
    assert backup.stat().st_mode & 0o777 == 0o600
    assert manifest.projects[0].published_commit == published.published_commit
    assert not (source.config.parent / "memory.db").exists()
    restore_git_backup(backup, restored)
    assert (restored / "RESTORE_INCOMPLETE").exists()
    assert not (restored / "config/memory.db").exists()
    config = BasicMemoryConfig.model_validate_json((restored / "config/config.json").read_bytes())
    assert (
        config.skip_initialization_sync and not config.git_sync_remotes and not config.auto_update
    )
    assert config.git_sync_merge == {}
    repository = GitProjectRepository(config.git_repositories[source.identity])
    assert repository.head() == source.repository.head()
    assert repository.snapshot().manifest.note_ids == {"published.md": identity}
    assert not repository.snapshot().files.get("private.md")
    recovered = GitTransactionRepository(repository, source.identity)
    _, private = recovered.read(draft.transaction_id, draft.transaction_key)
    assert "Unpublished" in private.files["private.md"]
    assert recovered.load(successor.transaction_id).source_transaction_id == draft.transaction_id
    assert recovered.load(draft.transaction_id).state.kind == "frozen"
    with pytest.raises(Exception, match="key is invalid"):
        recovered.status(draft.transaction_id, "wrong-key")
    for transaction, revision in [(committed, 1), (noop, 0)]:
        retried = recovered.commit(
            transaction.transaction_id,
            transaction.transaction_key,
            expected_revision=revision,
            message="Lost answer retry",
        )
        assert retried.already_committed
        assert retried.published_commit == published.published_commit
    working = restored / "recovery/working" / str(source.identity) / "published.md"
    assert working.read_bytes() == (source.root / "published.md").read_bytes()
    assert not (repository.path / "bm-local-binding.json").exists()
    assert not list((restored / "notes").rglob("*.md"))
    with ZipFile(backup) as archive:
        metadata = b"".join(archive.read(n) for n in archive.namelist() if n.endswith(".json"))
        assert draft.transaction_key.encode() not in metadata
        assert noop.transaction_key.encode() not in metadata


@pytest.mark.parametrize("lock_name", ["bm-projection.lock", "bm-transactions.lock"])
def test_backup_refuses_running_ownership(source, tmp_path, lock_name):
    backup = tmp_path / "backup.zip"
    with FileLock(source.repository.path / lock_name):
        with pytest.raises(Timeout):
            create_git_backup(source.config, backup)
    assert not backup.exists()


@pytest.mark.parametrize("path", ["existing", "inside-notes", "symlink", "legacy", "binding"])
def test_unsafe_backup_destinations_and_unadopted_sources_are_refused(source, tmp_path, path):
    backup = tmp_path / "backup.zip"
    if path == "existing":
        backup.write_bytes(b"Keep")
    elif path == "inside-notes":
        backup = source.root / "backup.zip"
    elif path == "symlink":
        (source.root / "link.md").symlink_to(source.config)
    else:
        config = json.loads(source.config.read_text())
        if path == "legacy":
            config["git_repositories"] = {}
            source.config.write_text(json.dumps(config))
        else:
            marker = source.repository.path / "bm-local-binding.json"
            value = json.loads(marker.read_text())
            value["project_external_id"] = str(uuid4())
            marker.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileExistsError)):
        create_git_backup(source.config, backup)
    if path == "existing":
        assert backup.read_bytes() == b"Keep"
    else:
        assert not backup.exists()


@pytest.mark.parametrize("change", ["refs", "config"])
def test_changes_during_backup_never_publish_a_mixed_archive(source, tmp_path, monkeypatch, change):
    original = GitProjectRepository._git

    def changed(repository, *args, **kwargs):
        result = original(repository, *args, **kwargs)
        if args[:2] == ("bundle", "create"):
            if change == "refs":
                original(repository, "update-ref", "refs/bm/late", repository.head())
            else:
                source.config.write_text(source.config.read_text() + "\n")
        return result

    monkeypatch.setattr(GitProjectRepository, "_git", changed)
    with pytest.raises(ValueError, match="changed during"):
        create_git_backup(source.config, tmp_path / "backup.zip")
    assert not (tmp_path / "backup.zip").exists()


@pytest.mark.parametrize(
    "mutation", ["checksum", "size", "missing", "duplicate", "traversal", "directory", "symlink"]
)
def test_invalid_archive_never_creates_a_restore_target(source, tmp_path, mutation):
    backup, target = tmp_path / "backup.zip", tmp_path / "restore"
    create_git_backup(source.config, backup)
    if mutation in {"checksum", "size"}:
        with ZipFile(backup) as archive:
            value = json.loads(archive.read("manifest.json"))
        value["files"]["config.json"]["sha256" if mutation == "checksum" else "size"] = (
            "0" * 64 if mutation == "checksum" else 0
        )
        repack(backup, replace={"manifest.json": json.dumps(value).encode()})
    elif mutation == "missing":
        repack(backup, replace={}, omit={"config.json"})
    else:
        from zipfile import ZipInfo

        with warnings.catch_warnings(), ZipFile(backup, "a") as archive:
            warnings.simplefilter("ignore", UserWarning)
            if mutation == "symlink":
                info = ZipInfo("link")
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                archive.writestr(info, "destination")
            else:
                archive.writestr(
                    {"duplicate": "config.json", "traversal": "../escape", "directory": "dir/"}[
                        mutation
                    ],
                    b"untrusted",
                )
    with pytest.raises(ValueError):
        restore_git_backup(backup, target)
    assert not target.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "member", ["unknown", "metadata/unknown.json", "metadata/bm-transactions/bad.json"]
)
def test_unknown_content_stays_incomplete_and_never_escapes_target(source, tmp_path, member):
    backup, target = tmp_path / "backup.zip", tmp_path / "restore"
    create_git_backup(source.config, backup)
    add_inventory_member(backup, f"projects/{source.identity}/{member}", b"{}")
    with pytest.raises(ValueError):
        restore_git_backup(backup, target)
    assert (target / "RESTORE_INCOMPLETE").exists()
    assert not (target / "READY.json").exists()


def test_existing_restore_and_overlapping_source_are_never_overwritten(source, tmp_path):
    backup = tmp_path / "backup.zip"
    create_git_backup(source.config, backup)
    sentinel = source.root / "keep.txt"
    sentinel.write_text("Keep")
    with pytest.raises(FileExistsError):
        restore_git_backup(backup, source.root)
    with pytest.raises(ValueError, match="overlaps"):
        restore_git_backup(backup, source.root / "nested")
    assert sentinel.read_text() == "Keep"


@pytest.mark.parametrize(
    "name", ["", ".", "../escape", "/absolute", "a//b", "./a", "C:/x", "a\\b", "a\nb"]
)
def test_member_path_validation(name):
    with pytest.raises(ValueError, match="Unsafe"):
        git_backup.safe_member(name)


@pytest.mark.parametrize(
    "fault", ["relative", "lock_link", "metadata_link", "transaction_link", "foreign_sync", "cloud"]
)
def test_invalid_local_control_state_refuses_backup(source, tmp_path, fault):
    if fault in {"relative", "cloud"}:
        config = json.loads(source.config.read_bytes())
        if fault == "relative":
            config["git_repositories"][str(source.identity)] = "relative.git"
        else:
            config["projects"]["Research"]["mode"] = "cloud"
        source.config.write_text(json.dumps(config))
    elif fault == "foreign_sync":
        (source.repository.path / "bm-sync.json").write_text(
            GitSyncRecord(project_id=uuid4(), peer_fingerprint="peer").model_dump_json()
        )
    else:
        name = {
            "lock_link": "bm-projection.lock",
            "metadata_link": "bm-sync.json",
            "transaction_link": "bm-transactions",
        }[fault]
        (source.repository.path / name).symlink_to(
            source.config if fault != "transaction_link" else source.config.parent
        )
    with pytest.raises(ValueError):
        create_git_backup(source.config, tmp_path / "backup.zip")


@pytest.mark.parametrize("fault", ["receipt", "source_revision"])
def test_missing_terminal_evidence_or_source_pins_refuse_backup(source, tmp_path, fault):
    records = GitTransactionRepository(source.repository, source.identity)
    transaction = records.start(GitTransactionStart())
    if fault == "receipt":
        records.commit(
            transaction.transaction_id,
            transaction.transaction_key,
            expected_revision=0,
            message="Noop",
        )
        source.repository._git("update-ref", "-d", f"refs/bm/receipts/{transaction.transaction_id}")
    else:
        successor = records.start(
            GitTransactionStart(
                source_transaction_id=transaction.transaction_id,
                source_key=transaction.transaction_key,
                source_revision=0,
            )
        )
        with records.locked():
            record = records.load(successor.transaction_id)
            records.save(record.model_copy(update={"source_revision": 1}))
    with pytest.raises(ValueError):
        create_git_backup(source.config, tmp_path / "backup.zip")


@pytest.mark.parametrize("fault", ["project_inventory", "head", "filename_case"])
def test_structural_archive_mismatch_remains_incomplete(source, tmp_path, fault):
    backup, target = tmp_path / "backup.zip", tmp_path / "target"
    create_git_backup(source.config, backup)
    with ZipFile(backup) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    if fault == "filename_case":
        name = f"projects/{source.identity}/metadata/bm-transactions/{str(uuid4()).upper()}.json"
        add_inventory_member(backup, name, b"{}")
    else:
        if fault == "project_inventory":
            manifest["projects"] = []
        else:
            manifest["projects"][0]["published_commit"] = "0" * 40
        repack(backup, replace={"manifest.json": json.dumps(manifest).encode()})
    with pytest.raises(ValueError):
        restore_git_backup(backup, target)
    assert not (target / "READY.json").exists()


@pytest.mark.parametrize("phase", ["backup_before", "backup_after", "restore_files"])
def test_process_death_preserves_source_and_never_leaves_a_false_complete_restore(
    source, tmp_path, phase
):
    backup, target = tmp_path / "backup.zip", tmp_path / "target"
    before = source.repository.head()
    if phase == "restore_files":
        create_git_backup(source.config, backup)
    result = subprocess.run(
        [
            sys.executable,
            "test-int/fixtures/git_backup_crash.py",
            phase,
            str(backup if phase == "restore_files" else source.config),
            str(target if phase == "restore_files" else backup),
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 84, result.stderr
    assert source.repository.head() == before
    if phase == "backup_before":
        assert not backup.exists()
        create_git_backup(source.config, backup)
    elif phase == "backup_after":
        with pytest.raises(FileExistsError):
            create_git_backup(source.config, backup)
    else:
        assert (target / "RESTORE_INCOMPLETE").exists()
        assert not (target / "READY.json").exists()
        with pytest.raises(FileExistsError):
            restore_git_backup(backup, target)
        return
    restore_git_backup(backup, target)
    repository = GitProjectRepository(target / "repositories" / f"{source.identity}.git")
    assert repository.head() == before


def test_lost_publication_answer_reconciles_after_restore_without_an_extra_commit(source, tmp_path):
    records = GitTransactionRepository(source.repository, source.identity)
    transaction = records.start(GitTransactionStart())
    records.replace(
        transaction.transaction_id,
        transaction.transaction_key,
        expected_revision=0,
        files={"note.md": note("Published")},
        note_ids={"note.md": uuid4()},
    )
    # Git published, but the process died before updating the private JSON receipt.
    record = records.load(transaction.transaction_id)
    published = source.repository.publish(draft_candidate(record))
    assert records.load(transaction.transaction_id).state.kind == "open"
    backup, target = tmp_path / "backup.zip", tmp_path / "target"
    create_git_backup(source.config, backup)
    restore_git_backup(backup, target)
    repository = GitProjectRepository(target / "repositories" / f"{source.identity}.git")
    recovered = GitTransactionRepository(repository, source.identity)
    answer = recovered.commit(
        transaction.transaction_id,
        transaction.transaction_key,
        expected_revision=1,
        message="Retry",
    )
    assert answer.already_committed and answer.published_commit == published.commit_id
    assert repository.head() == published.commit_id


def test_root_binding_must_agree_with_bare_repository(source, tmp_path):
    marker = source.root / ".basic-memory/git-binding.json"
    binding = json.loads(marker.read_bytes())
    binding["project_external_id"] = str(uuid4())
    marker.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="Working root binding"):
        create_git_backup(source.config, tmp_path / "backup.zip")


def test_multiple_projects_restore_with_independent_uuid_and_root_mappings(source, tmp_path):
    second = GitProjectRepository.create(tmp_path / "second.git")
    identity = second.snapshot().manifest.project_external_id
    root = tmp_path / "second-root"
    (root / ".basic-memory").mkdir(parents=True)
    binding = GitLocalBinding(
        project_external_id=identity, project_root=str(root), repository_path=str(second.path)
    )
    for path in [second.path / "bm-local-binding.json", root / ".basic-memory/git-binding.json"]:
        write_private(path, binding.model_dump_json().encode())
    config = BasicMemoryConfig.model_validate_json(source.config.read_bytes())
    config.projects["Second"] = ProjectEntry(path=str(root))
    config.git_repositories[identity] = second.path
    source.config.write_text(config.model_dump_json())
    backup, target = tmp_path / "backup.zip", tmp_path / "target"
    manifest = create_git_backup(source.config, backup)
    restore_git_backup(backup, target)
    restored = BasicMemoryConfig.model_validate_json((target / "config/config.json").read_bytes())
    assert {p.name for p in manifest.projects} == {"Research", "Second"}
    assert set(restored.git_repositories) == {source.identity, identity}
    for project in manifest.projects:
        repository = GitProjectRepository(restored.git_repositories[project.project_id])
        assert repository.snapshot().manifest.project_external_id == project.project_id
        assert repository.head() == project.published_commit
        assert restored.projects[project.name].path == str(
            target / "notes" / str(project.project_id)
        )
