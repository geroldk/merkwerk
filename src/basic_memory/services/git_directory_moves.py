"""Plan a folder move inside one Git project: every note, one commit, all or nothing (V1).

A Git project knows a folder only through the notes under it, so the plan is computed from
the published tree alone (path -> bytes, path -> UUID). Nothing here reads SQL, the disk or
Git; the runtime adds those checks and publishes the plan as one commit. Every refusal is
raised before anything changes and names all offenders at once, so a caller can fix them
together instead of discovering them one retry at a time.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID

from basic_memory.file_utils import ParseError, parse_frontmatter
from basic_memory.schemas.git_project import validate_note_path
from basic_memory.utils import (
    file_path_conflict_keys,
    nfc_path,
    normalize_file_path_for_comparison,
    portable_path_key,
    resolve_directory_casing,
)

type GitDirectoryMoveReason = Literal[
    "project_root",
    "invalid_path",
    "not_found",
    "ambiguous_source",
    "same_path",
    "case_only",
    "inside_source",
    "occupied",
    "unpublished_file",
    "unsupported_config",
    "invalid_draft",
    "base_advanced",
    "not_accepting",
]
type GitDirectoryMoveConflictReason = Literal[
    # A target path is a note outside the folder, exactly.
    "occupied",
    # A target names the same note as another one (case, Unicode form or permalink).
    "equivalent",
    # A target or one of its folders is a file, or a target is a folder of another note.
    "file_folder_clash",
    # A file under the source is not a published note (attachment, editor file).
    "unpublished_file",
    # A published note under the source was changed on disk outside Basic Memory.
    "edited_on_disk",
    # A symlink sits where the move would read or write.
    "symlink",
]


@dataclass(frozen=True, slots=True)
class GitDirectoryMoveConflict:
    """One offender: `path` is where the move would read or write, `other` what is there."""

    path: str
    reason: GitDirectoryMoveConflictReason
    other: str | None = None

    def describe(self) -> str:
        match self.reason:
            case "occupied":
                if self.other is None:
                    return f"{self.path} already exists on disk"
                return f"{self.path} is already a note"
            case "equivalent":
                return f"{self.path} names the same note as {self.other}"
            case "file_folder_clash":
                if self.other is None:
                    return f"{self.path} is a folder on disk"
                return f"{self.path} clashes with {self.other} (a file where a folder is needed)"
            case "unpublished_file":
                return f"{self.path} is not a published note"
            case "edited_on_disk":
                return f"{self.path} was changed on disk outside Basic Memory"
            case "symlink":
                return f"{self.path} is a symlink"


class GitDirectoryMoveRefused(Exception):
    """A folder move refused before anything changed; `conflicts` lists every offender."""

    def __init__(
        self,
        status_code: int,
        reason: GitDirectoryMoveReason,
        message: str,
        conflicts: Sequence[GitDirectoryMoveConflict] = (),
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason: GitDirectoryMoveReason = reason
        self.message = message
        self.conflicts = tuple(conflicts)


@dataclass(frozen=True, slots=True)
class GitPlannedNoteMove:
    previous_file_path: str
    file_path: str
    external_id: UUID
    permalink: str


@dataclass(frozen=True, slots=True)
class GitDirectoryMovePlan:
    """`source` and `destination` are the spellings the move uses, after resolution."""

    source: str
    destination: str
    moves: tuple[GitPlannedNoteMove, ...]

    @property
    def message(self) -> str:
        """The commit message: one line a person reading `git log` understands."""
        return f"Move {self.source}/ to {self.destination}/ ({len(self.moves)} notes)"


@dataclass(frozen=True, slots=True)
class GitDirectoryMoveOutcome:
    plan: GitDirectoryMovePlan
    commit_id: str


# --- Paths ---


def normalize_note_directory(value: str, *, role: Literal["source", "destination"]) -> str:
    """A project-relative folder as Git spells paths, or a refusal.

    Leading and trailing '/' name the same folder. Whitespace is never stripped: Git keeps
    ' notes' and 'notes' apart, so stripping could address a different folder.
    """
    directory = nfc_path(value.strip("/"))
    if not directory:
        raise GitDirectoryMoveRefused(
            422,
            "project_root",
            "The project root cannot be moved."
            if role == "source"
            else "A folder cannot move to the project root; name its new path.",
        )
    if directory != directory.strip():
        raise GitDirectoryMoveRefused(
            422,
            "invalid_path",
            f"'{value}' starts or ends with whitespace; name the folder exactly.",
        )
    try:
        # A folder is valid when a note inside it would be.
        validate_note_path(f"{directory}/x.md")
    except ValueError as error:
        raise GitDirectoryMoveRefused(
            422, "invalid_path", f"'{value}' is not a folder path a Git project can hold."
        ) from error
    return directory


def note_directories(paths: Iterable[str]) -> set[str]:
    """Every folder that holds a note, directly or below it."""
    directories: set[str] = set()
    for path in paths:
        parts = path.split("/")[:-1]
        directories.update("/".join(parts[: index + 1]) for index in range(len(parts)))
    return directories


def _under(path: str, directory: str) -> bool:
    return path.startswith(f"{directory}/")


def _resolve_source(requested: str, directories: set[str]) -> str:
    """The existing folder the caller means: exact spelling first, else one equivalent."""
    if requested in directories:
        return requested
    key = portable_path_key(requested)
    matches = sorted(directory for directory in directories if portable_path_key(directory) == key)
    if len(matches) > 1:
        # Upstream merges case-variant folders through SQL LIKE; one Git commit must not.
        raise GitDirectoryMoveRefused(
            409,
            "ambiguous_source",
            f"'{requested}' matches folders that differ only in case or Unicode form: "
            f"{', '.join(matches)}. Name one of them exactly.",
        )
    if not matches:
        raise GitDirectoryMoveRefused(404, "not_found", f"No published notes under '{requested}'.")
    return matches[0]


def _published_permalink(path: str, text: str) -> str:
    try:
        permalink = parse_frontmatter(text).get("permalink")
    except ParseError:
        permalink = None
    if not isinstance(permalink, str) or not permalink:
        raise GitDirectoryMoveRefused(
            422, "invalid_draft", f"Published note {path} has no readable permalink."
        )
    return permalink


# --- Plan ---


def plan_git_directory_move(
    files: Mapping[str, str],
    note_ids: Mapping[str, UUID],
    source: str,
    destination: str,
) -> GitDirectoryMovePlan:
    """Map every note under `source` to the same place under `destination`.

    The checks run in a fixed order; each raises GitDirectoryMoveRefused and changes
    nothing. Bytes are not touched here: the runtime stamps missing times and validates.
    """
    requested_source = normalize_note_directory(source, role="source")
    requested_destination = normalize_note_directory(destination, role="destination")

    # --- Source ---
    directories = note_directories(files)
    resolved_source = _resolve_source(requested_source, directories)
    moving = sorted(path for path in files if _under(path, resolved_source))
    if not moving:  # pragma: no cover - a resolved folder always holds a note
        raise GitDirectoryMoveRefused(
            404, "not_found", f"No published notes under '{resolved_source}'."
        )

    # --- Destination ---
    # Trigger: the destination differs from an existing folder only in case or Unicode form.
    # Why: two such folders merge on macOS and Windows checkouts of the repository.
    # Outcome: the existing spelling is used. Folders inside the source are left out, so a
    #   rename of the source itself is not resolved back onto the source.
    outside = sorted(
        directory
        for directory in directories
        if directory != resolved_source and not _under(directory, resolved_source)
    )
    resolved_destination = resolve_directory_casing(requested_destination, outside)
    if resolved_destination == resolved_source:
        raise GitDirectoryMoveRefused(
            422,
            "same_path",
            f"'{resolved_source}' is already there; name a different destination.",
        )
    source_key = portable_path_key(resolved_source)
    destination_key = portable_path_key(resolved_destination)
    if destination_key == source_key:
        raise GitDirectoryMoveRefused(
            422,
            "case_only",
            f"'{resolved_source}' and '{resolved_destination}' differ only in case or Unicode "
            "form; such a rename cannot be one move.",
        )
    if destination_key.startswith(f"{source_key}/"):
        raise GitDirectoryMoveRefused(
            422,
            "inside_source",
            f"'{resolved_destination}' is inside '{resolved_source}'; a folder cannot move "
            "into itself.",
        )
    # 'a/b' -> 'a' (the source inside the destination) is allowed: a moved note may land
    # on the path another moved note leaves, which one commit swaps cleanly.

    # --- Targets ---
    moves = tuple(
        GitPlannedNoteMove(
            previous_file_path=path,
            # Built from the resolved source spelling: its length is the prefix's length.
            file_path=f"{resolved_destination}{path[len(resolved_source) :]}",
            external_id=note_ids[path],
            permalink=_published_permalink(path, files[path]),
        )
        for path in moving
    )
    for move in moves:
        try:
            validate_note_path(move.file_path)
        except ValueError as error:  # pragma: no cover - a valid folder plus a valid remainder
            raise GitDirectoryMoveRefused(
                422, "invalid_path", f"'{move.file_path}' is not a path a Git project can hold."
            ) from error
    conflicts = _target_conflicts(
        [path for path in files if not _under(path, resolved_source)],
        [move.file_path for move in moves],
    )
    if conflicts:
        raise GitDirectoryMoveRefused(
            409,
            "occupied",
            f"{len(conflicts)} destination path(s) are taken: "
            + "; ".join(conflict.describe() for conflict in conflicts),
            conflicts,
        )
    return GitDirectoryMovePlan(resolved_source, resolved_destination, moves)


def _target_conflicts(
    remaining: Sequence[str], targets: Sequence[str]
) -> list[GitDirectoryMoveConflict]:
    """Every target that would collide with a note that stays, or with another target.

    Equivalence uses the same two keys as a single move (`file_path_conflict_keys`): the
    lowercased NFD path and the permalink the path would generate. A pair already inside
    the source ('a/X.md' beside 'a/x.md') therefore refuses the folder, as the second of
    two single moves would be refused.
    """
    taken = set(remaining)
    path_owner: dict[str, str] = {}
    permalink_owner: dict[str, str] = {}
    for path in remaining:
        keys = file_path_conflict_keys(path)
        path_owner.setdefault(keys.path_key, path)
        permalink_owner.setdefault(keys.permalink, path)
    # Folder and file keys of the tree that stays, for file/folder clashes.
    folder_owner: dict[str, str] = {}
    for path in remaining:
        for directory in note_directories([path]):
            folder_owner.setdefault(normalize_file_path_for_comparison(directory), path)

    conflicts: list[GitDirectoryMoveConflict] = []
    for target in sorted(targets):
        keys = file_path_conflict_keys(target)
        if target in taken:
            conflicts.append(GitDirectoryMoveConflict(target, "occupied", target))
        elif (
            other := path_owner.get(keys.path_key) or permalink_owner.get(keys.permalink)
        ) is not None:
            conflicts.append(GitDirectoryMoveConflict(target, "equivalent", other))
        elif (other := folder_owner.get(keys.path_key)) is not None:
            conflicts.append(GitDirectoryMoveConflict(target, "file_folder_clash", other))
        elif (
            other := next(
                (
                    path_owner[key]
                    for directory in note_directories([target])
                    if (key := normalize_file_path_for_comparison(directory)) in path_owner
                ),
                None,
            )
        ) is not None:
            conflicts.append(GitDirectoryMoveConflict(target, "file_folder_clash", other))
        # Later targets also collide with earlier ones (two notes in the source that
        # name the same note at the destination).
        path_owner.setdefault(keys.path_key, target)
        permalink_owner.setdefault(keys.permalink, target)
        taken.add(target)
        for directory in note_directories([target]):
            folder_owner.setdefault(normalize_file_path_for_comparison(directory), target)
    return conflicts


# --- Disk ---
# The published tree is the truth, but the files stage writes it to disk later and deletes
# old files only when their bytes are still the ones BM wrote. These checks refuse a move
# that would leave that stage blocked for good (an edited note, a file where a folder must
# go) and name every file in the source that is not a note, by Gerold's decision: nothing
# is ignored. Files BM is still due to delete (pending retirements) do not count.


@dataclass(frozen=True, slots=True)
class GitDirectoryMoveDisk:
    """What the files stage may find on disk: the bytes it wrote or is allowed to delete.

    `published` maps each moving note's current path to the checksums its file may have
    (published bytes, last written bytes, pending retirement); `retired` maps each path
    with a pending retirement to the checksums it may delete.
    """

    root: Path
    published: Mapping[str, frozenset[str]]
    retired: Mapping[str, frozenset[str]]


def _relative(root: Path, path: str) -> str:
    return nfc_path(PurePosixPath(os.path.relpath(path, root)).as_posix())


def _file_checksum(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def _folder_conflict(
    disk: GitDirectoryMoveDisk, path: str, folder: str
) -> GitDirectoryMoveConflict | None:
    """A conflict if `folder` exists on disk as anything but a real folder."""
    try:
        mode = os.lstat(disk.root / folder).st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode):
        return GitDirectoryMoveConflict(folder, "symlink")
    if not stat.S_ISDIR(mode):
        return GitDirectoryMoveConflict(path, "file_folder_clash", folder)
    return None


def _folders_of(directory: str) -> list[str]:
    parts = directory.split("/")
    return ["/".join(parts[: index + 1]) for index in range(len(parts))]


def unpublished_source_files(
    disk: GitDirectoryMoveDisk, directory: str
) -> list[GitDirectoryMoveConflict]:
    """Every file under `directory` that is neither a moving note nor due for deletion."""
    conflicts: list[GitDirectoryMoveConflict] = []
    for folder in _folders_of(directory):
        if (conflict := _folder_conflict(disk, directory, folder)) is not None:
            return [conflict]
    top = disk.root / directory
    if not top.is_dir():
        return conflicts
    # followlinks=False still lists a symlinked folder in `folders`; it is never entered.
    for current, folders, names in os.walk(top, followlinks=False):
        for name in sorted(folders):
            if os.path.islink(os.path.join(current, name)):
                conflicts.append(
                    GitDirectoryMoveConflict(
                        _relative(disk.root, os.path.join(current, name)), "symlink"
                    )
                )
        for name in sorted(names):
            full = os.path.join(current, name)
            path = _relative(disk.root, full)
            mode = os.lstat(full).st_mode
            if stat.S_ISLNK(mode):
                conflicts.append(GitDirectoryMoveConflict(path, "symlink"))
                continue
            checksum = _file_checksum(full) if stat.S_ISREG(mode) else None
            if path in disk.published:
                if checksum not in disk.published[path]:
                    conflicts.append(GitDirectoryMoveConflict(path, "edited_on_disk"))
            elif checksum is None or checksum not in disk.retired.get(path, frozenset()):
                conflicts.append(GitDirectoryMoveConflict(path, "unpublished_file"))
    return sorted(conflicts, key=lambda conflict: conflict.path)


def disk_conflicts(
    disk: GitDirectoryMoveDisk, plan: GitDirectoryMovePlan
) -> tuple[list[GitDirectoryMoveConflict], list[GitDirectoryMoveConflict]]:
    """Disk states that would leave the published move impossible to write out.

    Returns the offenders in the source (files that are not clean published notes) and
    those at the destination (something already where a note or folder must go).
    """
    leaving = {move.previous_file_path for move in plan.moves}
    seen: dict[str, GitDirectoryMoveConflict | None] = {}
    taken: list[GitDirectoryMoveConflict] = []
    for move in plan.moves:
        target = move.file_path
        folder_conflict = None
        for folder in _folders_of(target.rpartition("/")[0]):
            if folder not in seen:
                seen[folder] = _folder_conflict(disk, target, folder)
            if (folder_conflict := seen[folder]) is not None:
                break
        if folder_conflict is not None:
            if folder_conflict.reason == "file_folder_clash":
                taken.append(
                    GitDirectoryMoveConflict(target, "file_folder_clash", folder_conflict.other)
                )
            elif folder_conflict not in taken:
                taken.append(folder_conflict)
            continue
        # Trigger: the target is the current path of another moving note ('a/b' -> 'a').
        # Why: the files stage deletes that note's old bytes before it writes the new ones.
        # Outcome: only targets that no moving note leaves are checked for occupancy.
        if target in leaving:
            continue
        full = disk.root / target
        try:
            mode = os.lstat(full).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            taken.append(GitDirectoryMoveConflict(target, "symlink"))
        elif stat.S_ISDIR(mode):
            taken.append(GitDirectoryMoveConflict(target, "file_folder_clash"))
        elif not (
            stat.S_ISREG(mode)
            and _file_checksum(str(full)) in disk.retired.get(target, frozenset())
        ):
            # A file BM has not indexed: the single move refuses it the same way.
            taken.append(GitDirectoryMoveConflict(target, "occupied"))
    return unpublished_source_files(disk, plan.source), taken
