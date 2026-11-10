"""Durable cross-project move intents next to the target's bare repository (I1).

An intent is written before the source deletes the note and removed only when the move
has completed or a person resolved it; it never expires. It holds the target bytes and
the target draft's key, so the directory is private (0700), each file is private (0600),
symlinks are refused, and a write replaces the file atomically (temporary file, fsync,
rename, fsync of the directory), as the transaction pointers next to it are written.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from filelock import FileLock

from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.schemas.git_moves import GitMoveIntent

MOVES_DIRECTORY = "bm-moves"
MOVES_LOCK = "bm-moves.lock"


class GitMoveIntentError(RuntimeError):
    """An intent file is unusable: a symlink, another project's, or unreadable."""


@dataclass(frozen=True, slots=True)
class GitMoveIntentProblem:
    """One file in bm-moves/ that is no usable intent of this target."""

    file_name: str
    reason: str
    detail: str


class GitMoveIntentRepository:
    """The intents whose target is this repository's project."""

    def __init__(self, repository: GitProjectRepository, project_id: UUID) -> None:
        self.repository = repository
        self.project_id = project_id
        self.path = repository.path / MOVES_DIRECTORY
        self.lock_path = repository.path / MOVES_LOCK

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclude a concurrent offline backup; the host's move lock orders the writers."""
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise GitMoveIntentError("Move intents must not use symlinks")
        self.path.mkdir(mode=0o700, exist_ok=True)
        with FileLock(self.lock_path, timeout=30):
            yield

    def save(self, intent: GitMoveIntent) -> None:
        with self.locked():
            self._replace(intent)

    def _replace(self, intent: GitMoveIntent) -> None:
        target = self.path / f"{intent.move_id}.json"
        if target.is_symlink():
            raise GitMoveIntentError("A move intent must not be a symlink")
        # NamedTemporaryFile creates the file 0600, and os.replace keeps that mode.
        with NamedTemporaryFile(mode="wb", dir=self.path, prefix=".move-", delete=False) as out:
            temporary = Path(out.name)
            try:
                out.write(intent.model_dump_json(indent=2).encode("utf-8"))
                out.flush()
                os.fsync(out.fileno())
                os.replace(temporary, target)
                directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)

    def load(self, move_id: UUID) -> GitMoveIntent:
        path = self.path / f"{move_id}.json"
        if self.path.is_symlink() or path.is_symlink():
            raise GitMoveIntentError("A move intent must not be a symlink")
        if not path.is_file():
            raise KeyError(move_id)
        intent = GitMoveIntent.model_validate_json(path.read_bytes())
        if intent.move_id != move_id:
            raise GitMoveIntentError(f"Move intent {path.name} names another move")
        return intent

    def move_ids(self) -> list[UUID]:
        """Every intent file's move id; a stray file name is an error, not skipped."""
        if self.path.is_symlink():
            raise GitMoveIntentError("Move intents must not use symlinks")
        if not self.path.is_dir():
            return []
        identities = []
        for path in sorted(self.path.glob("*.json")):
            try:
                identity = UUID(path.stem)
            except ValueError as error:
                raise GitMoveIntentError(
                    f"Unknown file in {MOVES_DIRECTORY}: {path.name}"
                ) from error
            if path.name != f"{identity}.json":
                raise GitMoveIntentError(f"Unknown file in {MOVES_DIRECTORY}: {path.name}")
            identities.append(identity)
        return identities

    def load_each(self) -> tuple[list[GitMoveIntent], list[GitMoveIntentProblem]]:
        """Every readable intent of this target (oldest first), and each unusable file.

        One damaged or stray file must not hide its neighbours: each readable intent is
        still reserved and resumed. A problem carries Basic Memory's own wording for API
        callers and the raw error for the server log only.
        """
        if self.path.is_symlink():
            return [], [GitMoveIntentProblem(MOVES_DIRECTORY, "is a symlink", "symlink")]
        if not self.path.is_dir():
            return [], []
        intents: list[GitMoveIntent] = []
        problems: list[GitMoveIntentProblem] = []
        for path in sorted(self.path.glob("*.json")):
            try:
                identity = UUID(path.stem)
            except ValueError:
                identity = None
            if identity is None or path.name != f"{identity}.json":
                problems.append(GitMoveIntentProblem(path.name, "is not a move intent", ""))
                continue
            try:
                intent = self.load(identity)
            except GitMoveIntentError as error:
                problems.append(GitMoveIntentProblem(path.name, str(error), str(error)))
                continue
            except (KeyError, ValueError, OSError) as error:
                problems.append(GitMoveIntentProblem(path.name, "cannot be read", repr(error)))
                continue
            if intent.target_project_id != self.project_id:
                problems.append(GitMoveIntentProblem(path.name, "names another target project", ""))
                continue
            intents.append(intent)
        intents.sort(key=lambda intent: (intent.created_at, str(intent.move_id)))
        return intents, problems

    def load_all(self) -> list[GitMoveIntent]:
        """Every open intent of this target, oldest first; any unusable file raises.

        For callers that must refuse rather than skip (backup, project removal).
        """
        intents = [self.load(identity) for identity in self.move_ids()]
        if any(intent.target_project_id != self.project_id for intent in intents):
            raise GitMoveIntentError("A move intent names another target project")
        return sorted(intents, key=lambda intent: (intent.created_at, str(intent.move_id)))

    def remove(self, move_id: UUID) -> None:
        with self.locked():
            path = self.path / f"{move_id}.json"
            if path.is_symlink():
                raise GitMoveIntentError("A move intent must not be a symlink")
            path.unlink(missing_ok=True)
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
