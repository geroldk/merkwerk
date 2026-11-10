"""Offline Git backup and independent filesystem restore, without opening SQLite.

Exclusive worker and transaction locks provide one cut across all projects. The
archive retains private drafts and receipts which ordinary Git synchronization
intentionally omits. A restore never trusts old local bindings or copies indexes.
"""

import hashlib
import os
import stat
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import UUID
from zipfile import ZipFile

from filelock import FileLock

from basic_memory import __version__
from basic_memory.config import BasicMemoryConfig, DatabaseBackend, ProjectEntry, ProjectMode
from basic_memory.index.git_project_ownership import GitLocalBinding
from basic_memory.repository.git_move_intent_repository import (
    MOVES_DIRECTORY,
    MOVES_LOCK,
    GitMoveIntentRepository,
)
from basic_memory.repository.git_project_repository import GitProjectRepository, PUBLISHED_REF
from basic_memory.repository.git_transaction_repository import (
    GitTransactionRepository,
    draft_candidate,
)
from basic_memory.schemas.git_backup import GitBackupFile, GitBackupManifest, GitBackupProject
from basic_memory.schemas.git_sync import GitSyncRecord
from basic_memory.schemas.git_transactions import GitCommittedState, GitDraftState

_METADATA_FILES = {"bm-local-binding.json", "bm-adoption.json", "bm-sync.json"}


def require_separate_path(path: Path, sources: list[Path]) -> None:
    if any(path.is_relative_to(p) or p.is_relative_to(path) for p in sources):
        raise ValueError("Backup/restore destination overlaps source data")


def safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"..", "."} or ":" in part or "\\" in part for part in path.parts)
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError("Unsafe backup member path")


def repository_refs(repository: GitProjectRepository) -> dict[str, str]:
    return dict(
        line.split(" ", 1)
        for line in repository._git("for-each-ref", "--format=%(refname) %(objectname)")
        .decode("ascii")
        .splitlines()
    )


def validate_transactions(repository: GitProjectRepository, project_id: UUID) -> None:
    records = GitTransactionRepository(repository, project_id)
    if records.path.is_symlink():
        raise ValueError("Transaction metadata must not use symlinks")
    for path in records.path.glob("*.json"):
        record = records.load(UUID(path.stem))
        repository.snapshot(record.base_commit)
        if isinstance(record.state, GitDraftState):
            repository.candidate_snapshot(draft_candidate(record))
        if isinstance(record.state, GitCommittedState):
            publication = repository.transaction_status(record.transaction_id)
            if publication is None or publication.commit_id != record.state.commit_id:
                raise ValueError("Committed transaction evidence is missing")
        if record.source_transaction_id is not None:
            source = records.load(record.source_transaction_id)
            if source.revision != record.source_revision:
                raise ValueError("Retained source draft revision differs")


def validate_moves(repository: GitProjectRepository, project_id: UUID) -> None:
    """Open cross-project moves into this project (I1) must name their own draft."""
    records = GitTransactionRepository(repository, project_id)
    for intent in GitMoveIntentRepository(repository, project_id).load_all():
        records.load(intent.transaction_id)


def write_private(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as out:
        path.chmod(0o600)
        out.write(content)
        out.flush()
        os.fsync(out.fileno())


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_git_backup(config_path: Path, destination: Path) -> GitBackupManifest:
    """Publish a new private archive, refusing a running host or existing output."""
    config_path, destination = config_path.resolve(), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    original_config = config_path.read_bytes()
    config = BasicMemoryConfig.model_validate_json(original_config)
    if (
        config.database_backend is not DatabaseBackend.SQLITE
        or not config.git_repositories
        or len(config.projects) != len(config.git_repositories)
    ):
        raise ValueError("Backup requires an adopted, entirely Git-managed local SQLite host")
    projects: list[GitBackupProject] = []
    roots = {Path(entry.path).resolve(): name for name, entry in config.projects.items()}
    lock_paths: set[Path] = set()
    for identity, repository_path in config.git_repositories.items():
        if not repository_path.is_absolute():
            raise ValueError("Git repository path must be absolute")
        repository = GitProjectRepository(repository_path)
        binding = GitLocalBinding.model_validate_json(
            (repository.path / "bm-local-binding.json").read_bytes()
        )
        root = Path(binding.project_root)
        control = root / ".basic-memory"
        if (
            control.is_symlink()
            or GitLocalBinding.model_validate_json((control / "git-binding.json").read_bytes())
            != binding
        ):
            raise ValueError("Working root binding differs from its Git repository")
        name = roots.get(root)
        if (
            name is None
            or config.projects[name].mode is not ProjectMode.LOCAL
            or binding.project_external_id != identity
            or Path(binding.repository_path) != repository.path
            or repository.snapshot().manifest.project_external_id != identity
        ):
            raise ValueError("Git project binding differs from configuration")
        projects.append(
            GitBackupProject(
                project_id=identity,
                name=name,
                source_root=str(root),
                source_repository=str(repository.path),
                published_commit=repository.head(),
                refs=repository_refs(repository),
            )
        )
        lock_paths.update(
            {
                root / ".basic-memory/git-projection.lock",
                repository.path / "bm-projection.lock",
                repository.path / "bm-transactions.lock",
                # An open cross-project move (I1) must not change during the cut either.
                repository.path / MOVES_LOCK,
            }
        )
    sources = [config_path.parent, *roots, *(p.resolve() for p in config.git_repositories.values())]
    require_separate_path(destination, sources)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with ExitStack() as locks:
        for path in sorted(lock_paths):
            if path.is_symlink():
                raise ValueError("Backup ownership files must not use symlinks")
            locks.enter_context(FileLock(path, timeout=0, thread_local=False))
        # Capture authoritative pointers only after all writers are excluded.
        projects = [
            p.model_copy(
                update={
                    "published_commit": GitProjectRepository(Path(p.source_repository)).head(),
                    "refs": repository_refs(GitProjectRepository(Path(p.source_repository))),
                }
            )
            for p in projects
        ]
        with (
            TemporaryDirectory(prefix=".bm-backup-", dir=destination.parent) as directory,
            NamedTemporaryFile(prefix=".bm-backup-", dir=destination.parent) as output,
        ):
            inventory: dict[str, GitBackupFile] = {}
            with ZipFile(output, "w") as archive:
                archive.writestr("config.json", original_config)
                inventory["config.json"] = GitBackupFile(
                    size=len(original_config), sha256=hashlib.sha256(original_config).hexdigest()
                )
                for project in projects:
                    repository = GitProjectRepository(Path(project.source_repository))
                    validate_transactions(repository, project.project_id)
                    validate_moves(repository, project.project_id)
                    bundle = Path(directory) / f"{project.project_id}.bundle"
                    repository._git("bundle", "create", str(bundle), "--all")
                    prefix = f"projects/{project.project_id}"
                    files = [(bundle, f"{prefix}/repository.bundle")]
                    for path in (repository.path / "bm-transactions").glob("*.json"):
                        files.append((path, f"{prefix}/metadata/bm-transactions/{path.name}"))
                    # Without its intent, an interrupted move's note would live only in
                    # the source's history after a restore.
                    for path in (repository.path / MOVES_DIRECTORY).glob("*.json"):
                        files.append((path, f"{prefix}/metadata/{MOVES_DIRECTORY}/{path.name}"))
                    for name in sorted(_METADATA_FILES):
                        path = repository.path / name
                        if path.exists() or path.is_symlink():
                            if name == "bm-sync.json" and not path.is_symlink():
                                record = GitSyncRecord.model_validate_json(path.read_bytes())
                                if record.project_id != project.project_id:
                                    raise ValueError(
                                        "Synchronization metadata belongs to another project"
                                    )
                            files.append((path, f"{prefix}/metadata/{name}"))
                    root = Path(project.source_root)
                    for path in sorted(root.rglob("*")):
                        if path.is_symlink() or not (path.is_dir() or path.is_file()):
                            raise ValueError("Working files must be regular files or directories")
                        if (
                            path.is_file()
                            and not path.relative_to(root).parts[0] == ".basic-memory"
                        ):
                            files.append(
                                (path, f"{prefix}/working/{path.relative_to(root).as_posix()}")
                            )
                    for path, member in files:
                        safe_member(member)
                        if path.is_symlink():
                            raise ValueError("Backup files must not use symlinks")
                        digest = hashlib.sha256()
                        size = 0
                        with path.open("rb") as source, archive.open(member, "w") as target:
                            while chunk := source.read(1024 * 1024):
                                target.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                        inventory[member] = GitBackupFile(size=size, sha256=digest.hexdigest())
                    if repository_refs(repository) != project.refs:
                        raise ValueError("Repository changed during offline backup")
                if config_path.read_bytes() != original_config:
                    raise ValueError("Configuration changed during offline backup")
                manifest = GitBackupManifest(
                    created_at=datetime.now(UTC),
                    producer_version=__version__,
                    source_config=str(config_path),
                    projects=tuple(projects),
                    files=inventory,
                )
                archive.writestr("manifest.json", manifest.model_dump_json(indent=2))
            output.flush()
            os.fsync(output.fileno())
            # link() publishes a complete file and atomically refuses any existing
            # destination, including one that appeared after the initial check.
            os.link(output.name, destination)
            sync_directory(destination.parent)
    return manifest


def restore_git_backup(backup: Path, destination: Path) -> GitBackupManifest:
    """Restore to a new private instance. Index rebuild must finish before READY."""
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    with ZipFile(backup) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate backup members")
        for info in archive.infolist():
            safe_member(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Backup members must be regular files")
        manifest = GitBackupManifest.model_validate_json(archive.read("manifest.json"))
        if set(names) != {"manifest.json", *manifest.files}:
            raise ValueError("Backup inventory differs from archive")
        for name, expected in manifest.files.items():
            digest = hashlib.sha256()
            with archive.open(name) as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if (
                archive.getinfo(name).file_size != expected.size
                or digest.hexdigest() != expected.sha256
            ):
                raise ValueError("Backup checksum or size differs from inventory")
        config = BasicMemoryConfig.model_validate_json(archive.read("config.json"))
        if (
            not manifest.projects
            or len({p.project_id for p in manifest.projects}) != len(manifest.projects)
            or len({p.name for p in manifest.projects}) != len(manifest.projects)
            or {p.project_id for p in manifest.projects} != set(config.git_repositories)
            or {p.name for p in manifest.projects} != set(config.projects)
        ):
            raise ValueError("Backup project inventory is inconsistent")
        require_separate_path(
            destination,
            [Path(manifest.source_config).parent]
            + [Path(p.source_root) for p in manifest.projects]
            + [Path(p.source_repository) for p in manifest.projects],
        )
        # Reserve the destination before creating anything in it. An interrupted
        # restore stays explicitly incomplete and cannot be silently retried over.
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        write_private(destination / "RESTORE_INCOMPLETE", b"Index rebuild has not completed.\n")
        write_private(destination / "recovery/original-config.json", archive.read("config.json"))
        write_private(destination / "recovery/manifest.json", archive.read("manifest.json"))
        claimed = {"config.json"}
        for project in manifest.projects:
            prefix = f"projects/{project.project_id}"
            bundle_member = f"{prefix}/repository.bundle"
            claimed.add(bundle_member)
            bundle = destination / "recovery" / f"{project.project_id}.bundle"
            with archive.open(bundle_member) as source:
                write_private(bundle, source.read())
            root = destination / "notes" / str(project.project_id)
            root.mkdir(mode=0o700, parents=True)
            repository = GitProjectRepository.create(
                destination / "repositories" / f"{project.project_id}.git",
                project_external_id=project.project_id,
            )
            repository._git("update-ref", "-d", PUBLISHED_REF)
            repository._git("bundle", "unbundle", str(bundle))
            commands = "".join(f"create {ref} {value}\n" for ref, value in project.refs.items())
            repository._git("update-ref", "--stdin", input_bytes=commands.encode("ascii"))
            repository._git("fsck", "--full", "--strict")
            snapshot = repository.snapshot()
            if (
                snapshot.commit_id != project.published_commit
                or snapshot.manifest.project_external_id != project.project_id
                or repository_refs(repository) != project.refs
            ):
                raise ValueError("Restored Git references or project identity differ")
            for name in names:
                if name.startswith(f"{prefix}/working/"):
                    claimed.add(name)
                    suffix = name.removeprefix(f"{prefix}/working/")
                    target = destination / "recovery/working" / str(project.project_id) / suffix
                elif name.startswith(f"{prefix}/metadata/"):
                    claimed.add(name)
                    suffix = name.removeprefix(f"{prefix}/metadata/")
                    parts = PurePosixPath(suffix).parts
                    if parts[0] in {"bm-transactions", MOVES_DIRECTORY} and len(parts) == 2:
                        if parts[1] != f"{UUID(Path(parts[1]).stem)}.json":
                            raise ValueError("Invalid transaction metadata filename")
                        target = repository.path / suffix
                    elif suffix in _METADATA_FILES:
                        if suffix == "bm-sync.json":
                            write_private(repository.path / suffix, archive.read(name))
                        target = (
                            destination / "recovery/metadata" / str(project.project_id) / suffix
                        )
                    else:
                        raise ValueError("Unknown local metadata member")
                else:
                    continue
                with archive.open(name) as source:
                    write_private(target, source.read())
            validate_transactions(repository, project.project_id)
            # The rebuild's host start resumes these moves (index/git_host.py).
            validate_moves(repository, project.project_id)
        if claimed != set(manifest.files):
            raise ValueError("Unknown backup members")
        config.projects = {
            p.name: ProjectEntry(path=str(destination / "notes" / str(p.project_id)))
            for p in manifest.projects
        }
        config.git_repositories = {
            p.project_id: destination / "repositories" / f"{p.project_id}.git"
            for p in manifest.projects
        }
        # The merge settings must go with the peers: a host refuses `git_sync_merge` for a
        # project that has no peer, so a restore of a merge-enabled profile would not start.
        config.git_sync_remotes = {}
        config.git_sync_merge = {}
        config.cloud_api_key = None
        config.skip_initialization_sync = True
        config.auto_update = False
        config.semantic_embedding_cache_dir = str(destination / "config/fastembed_cache")
        write_private(destination / "config/config.json", config.model_dump_json(indent=2).encode())
        sync_directory(destination)
    return manifest
