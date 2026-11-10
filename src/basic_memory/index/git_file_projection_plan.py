"""Order the Git files stage so that a note's new file exists before its old file goes (F2).

The files stage makes disk follow one published commit. It writes first and deletes
afterwards, so a note that moves is never missing from disk; for the length of one
note it can be there twice. Only a path that another note still occupies is cleared
right before its new note is written, one path at a time. A cycle (two notes swap
paths) cannot avoid a gap without a temporary path: one note is then absent from the
first delete until its own write, which spans two deletes and two materializations.

This module decides the order from what the stage read (SQL, disk, retirement rows).
It does no I/O, so every case is testable directly. The stage in git_projection.py
carries the steps out in order, skips a step whose prerequisites did not succeed, and
re-checks every delete against the bytes on disk.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from basic_memory.utils import portable_path_key

EDITOR_CHANGE = "Old published path contains an unpublished editor change"
FOREIGN_FILE = "File differs from published Git bytes; import requires review"
BLOCKED_PATH = "A file Basic Memory does not own blocks the published path"


# --- Inputs: what the stage read before acting ---


@dataclass(frozen=True, slots=True)
class LiveNote:
    """A note of the target commit and what SQL says about its file."""

    path: str
    # sha256 of the published bytes.
    checksum: str
    # SQL holds exactly these bytes for this note's UUID at this path.
    ready: bool = True
    # The row says the file at this path already holds these bytes (synced lineage).
    synced: bool = False
    # The materializer's own guard accepts a file holding one of these.
    file_checksum: str | None = None
    sync_checksum: str | None = None


@dataclass(frozen=True, slots=True)
class RetiredFile:
    """A path a note left, and the bytes that may be deleted there (a retirement row)."""

    path: str
    expected_checksums: frozenset[str]


@dataclass(frozen=True, slots=True)
class DiskView:
    """What the stage found on disk for the live and retired paths."""

    # A regular file at the path, by sha256 of its bytes.
    files: Mapping[str, str] = field(default_factory=dict)
    # Live or retired paths that are folders on disk.
    directories: frozenset[str] = frozenset()
    # Live path -> its ancestors that are regular files (a file where a folder must be).
    blocking_files: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Live path that is a folder -> the regular files below it.
    folder_files: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Live paths that are folders holding anything but regular files and folders.
    folder_foreign: frozenset[str] = frozenset()
    # Live path -> the retired path it shares storage with (case or Unicode alias).
    aliases: Mapping[str, str] = field(default_factory=dict)
    # Paths whose bytes on disk were published at that very path in Git history.
    published: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class BaseFile:
    """A file of the last commit the stage fully wrote, and where its note lives now."""

    checksum: str
    # The note's path in the target commit; None when the target dropped it.
    live_path: str | None


@dataclass(frozen=True, slots=True)
class Occupant:
    """The note whose older bytes lie at a path: its live path, or None if it left."""

    live_path: str | None


# --- Output: ordered steps ---


class StepKind(StrEnum):
    # Nothing to remove: only the retirement row is completed.
    complete = "complete"
    # The bytes are already right; publish the row as synced.
    settle = "settle"
    # Materialize into a free path, or over the note's own earlier file.
    write = "write"
    # Guarded delete of the bytes at the path, then materialize there at once.
    replace = "replace"
    # Guarded delete of a pure retirement (or, for an alias, rename the entry).
    delete = "delete"


@dataclass(frozen=True, slots=True)
class FileStep:
    kind: StepKind
    path: str
    # The bytes the guarded delete may remove (replace/delete).
    expected_checksum: str | None = None
    # Live paths whose note must be on disk first: the note the deleted bytes belong to.
    requires: frozenset[str] = frozenset()
    # Retired paths that must be gone first: a file where this path needs a folder.
    clears: frozenset[str] = frozenset()
    # Complete the retirement row of this path after the step succeeded.
    retire: bool = False
    # Live paths that may alias this retired path on a case-insensitive file system.
    aliases: tuple[str, ...] = ()
    # This delete runs before the note it belongs to is written (a cycle).
    breaks_cycle: bool = False
    # The note the deleted bytes belong to is unknown: run only when every other note
    # is on disk and nothing conflicted, so no note can lose its only copy.
    final: bool = False


@dataclass(frozen=True, slots=True)
class FileProjectionPlan:
    steps: tuple[FileStep, ...]
    # Live paths already verified on disk and synced; nothing to do.
    done: frozenset[str]
    # Live paths whose SQL has not reached this target: skipped, and decided at the end.
    stale: frozenset[str]
    # (path, reason) recorded without touching disk.
    conflicts: tuple[tuple[str, str], ...]


# --- Who owns the bytes at a path ---


def identify_occupants(
    notes: Iterable[LiveNote],
    retirements: Iterable[RetiredFile],
    disk: DiskView,
    base: Mapping[str, BaseFile] | None = None,
) -> dict[str, Occupant]:
    """Name the note whose older bytes lie at each path that may be cleared.

    Identified from the bytes, never from memory: the last fully written commit counts
    only where disk still holds its bytes, because a superseded pass may have written
    and overwritten files since. Paths without an answer are left out (unknown).
    """
    notes = tuple(notes)
    live = {note.path for note in notes}
    by_checksum: dict[str, list[str]] = defaultdict(list)
    by_file_checksum: dict[str, list[str]] = defaultdict(list)
    for note in notes:
        by_checksum[note.checksum].append(note.path)
        if note.file_checksum is not None:
            by_file_checksum[note.file_checksum].append(note.path)
    candidates = {retirement.path for retirement in retirements} | {
        path for path in (*disk.published, *disk.aliases) if path in live
    }
    occupants: dict[str, Occupant] = {}
    for path in sorted(candidates):
        checksum = disk.files.get(path)
        if checksum is None:
            continue
        known = base.get(path) if base is not None else None
        # Trigger: the last fully written commit had exactly these bytes here.
        # Why: that commit names the note by UUID, which survives moves with edits.
        # Outcome: the note's path now (or None when it was deleted).
        if known is not None and known.checksum == checksum:
            occupants[path] = Occupant(known.live_path)
            continue
        # A plain move keeps the bytes; a local move keeps the old file checksum.
        for index in (by_checksum, by_file_checksum):
            owners = [owner for owner in index.get(checksum, ()) if owner != path]
            if len(owners) == 1:
                occupants[path] = Occupant(owners[0])
                break
    return occupants


def publication_proof_candidates(
    notes: Iterable[LiveNote], retirements: Iterable[RetiredFile], disk: DiskView
) -> tuple[str, ...]:
    """Paths whose bytes nothing in SQL explains; the stage asks Git whether it published them.

    A write that died before its publish (or an orphaned write) leaves bytes that are
    neither the row's file checksum nor the target. If Git published them at that
    path, they are Basic Memory's, not an editor's, and may be replaced (C7).
    """
    retired = {retirement.path: retirement for retirement in retirements}
    candidates = []
    for note in notes:
        checksum = disk.files.get(note.path)
        if (
            checksum is None
            or not note.ready
            or checksum in {note.checksum, note.file_checksum, note.sync_checksum}
        ):
            continue
        accepted = _expected(retired, note.path) | _expected(retired, disk.aliases.get(note.path))
        if checksum not in accepted:
            candidates.append(note.path)
    live = {note.path for note in notes}
    for retirement in retired.values():
        checksum = disk.files.get(retirement.path)
        if (
            retirement.path not in live
            and checksum is not None
            and checksum not in retirement.expected_checksums
        ):
            candidates.append(retirement.path)
    return tuple(sorted(candidates))


def _expected(retired: Mapping[str, RetiredFile], path: str | None) -> frozenset[str]:
    retirement = retired.get(path) if path is not None else None
    return retirement.expected_checksums if retirement is not None else frozenset()


# --- Classification ---


def plan_file_projection(
    notes: Iterable[LiveNote],
    retirements: Iterable[RetiredFile],
    disk: DiskView,
    occupants: Mapping[str, Occupant],
) -> FileProjectionPlan:
    """Classify every path, then order the steps so that no note loses its only copy.

    Two independent questions per path: what happens to the bytes there (keep, delete,
    delete-then-write, or an editor's file that blocks), and what the note now living
    there needs (nothing, a status publish, or a write). A path a note keeps (same bytes
    back) is still settled, so its row never stays pending (C2).
    """
    notes = tuple(notes)
    live = {note.path: note for note in notes}
    retired = {retirement.path: retirement for retirement in retirements}
    conflicts: dict[str, str] = {}
    steps: dict[str, FileStep] = {}
    done: set[str] = set()
    stale: set[str] = set()

    def needs(path: str) -> tuple[frozenset[str], bool]:
        """What must happen before the bytes at `path` may be deleted (C4)."""
        occupant = occupants.get(path)
        if occupant is None:
            return frozenset(), True
        if occupant.live_path is None or occupant.live_path == path:
            return frozenset(), False
        return frozenset({occupant.live_path}), False

    # --- Paths notes left: delete what Basic Memory wrote, keep an editor's file ---
    keys: dict[str, list[str]] = defaultdict(list)
    for path in live:
        keys[portable_path_key(path)].append(path)
    for path, retirement in sorted(retired.items()):
        if path in live:
            continue
        checksum = disk.files.get(path)
        if checksum is None:
            # Gone already (or a folder now stands there): only the row is left.
            steps[path] = FileStep(StepKind.complete, path)
        elif checksum in retirement.expected_checksums or path in disk.published:
            requires, final = needs(path)
            steps[path] = FileStep(
                StepKind.delete,
                path,
                expected_checksum=checksum,
                requires=requires,
                aliases=tuple(sorted(other for other in keys[portable_path_key(path)])),
                final=final,
            )
        else:
            conflicts[path] = EDITOR_CHANGE

    # --- Notes of the target commit ---
    for note in notes:
        path = note.path
        if not note.ready:
            stale.add(path)
            continue
        retirement = retired.get(path)
        checksum = disk.files.get(path)
        clears = set(disk.blocking_files.get(path, ()))
        if path in disk.directories:
            # A folder stands where this file goes (directory to file, C8): its files
            # must be retired ones, which go first; anything else blocks the path.
            if path in disk.folder_foreign:
                conflicts[path] = BLOCKED_PATH
                continue
            clears |= set(disk.folder_files.get(path, ()))
            checksum = None
        if any(
            step is None or step.kind not in {StepKind.delete, StepKind.complete}
            for step in (steps.get(blocker) for blocker in clears)
        ):
            conflicts[path] = BLOCKED_PATH
            continue
        retire = retirement is not None
        if checksum == note.checksum:
            if note.synced:
                done.add(path)
                if retire:
                    # A crash after the write and before the row was completed.
                    steps[path] = FileStep(StepKind.complete, path)
            else:
                steps[path] = FileStep(StepKind.settle, path, retire=retire)
        elif checksum is None:
            # A free path, also one whose earlier bytes are already gone (C3).
            steps[path] = FileStep(StepKind.write, path, clears=frozenset(clears), retire=retire)
        elif checksum in {note.file_checksum, note.sync_checksum}:
            # The note's own earlier file: the materializer overwrites it atomically. A
            # lingering row on this path is completed afterwards, not mistaken for an
            # editor change (C3).
            steps[path] = FileStep(StepKind.write, path, retire=retire)
        elif (
            retirement is not None and checksum in retirement.expected_checksums
        ) or path in disk.published:
            # Another note's bytes (or this note's older published ones): delete them
            # right before the write, one path at a time.
            requires, final = needs(path)
            steps[path] = FileStep(
                StepKind.replace,
                path,
                expected_checksum=checksum,
                requires=requires,
                retire=retire,
                final=final,
            )
        elif (alias := disk.aliases.get(path)) is not None and checksum in _expected(
            retired, alias
        ):
            # Case-only rename on a case-insensitive file system: this path reaches the
            # note's old file. Its own retirement row is completed by the alias's step.
            steps[path] = FileStep(StepKind.replace, path, expected_checksum=checksum)
        else:
            # Never hand foreign bytes to the materializer: it would stamp the row as an
            # external change (C9).
            conflicts[path] = EDITOR_CHANGE if retirement is not None else FOREIGN_FILE

    # --- Drop steps whose prerequisites can never be met ---
    # Trigger: the note the bytes belong to is blocked, stale or conflicted.
    # Why: deleting them would remove that note's only copy (C4).
    # Outcome: those bytes stay; the stage reports the root cause at the end.
    changed = True
    while changed:
        available = done | set(steps)
        blocked = [
            path for path, step in steps.items() if not (step.requires | step.clears) <= available
        ]
        for path in blocked:
            del steps[path]
        changed = bool(blocked)

    # Unknown owners wait for everything else; so does whatever depends on them.
    final = {path for path, step in steps.items() if step.final}
    changed = True
    while changed:
        grown = {
            path
            for path, step in steps.items()
            if path not in final and (step.requires | step.clears) & final
        }
        final |= grown
        changed = bool(grown)
    for path in final:
        steps[path] = replace(steps[path], final=True)

    ordered = _order({p: s for p, s in steps.items() if p not in final}, set(done))
    ordered += _order(
        {p: s for p, s in steps.items() if p in final},
        set(done) | {step.path for step in ordered},
    )
    return FileProjectionPlan(
        tuple(ordered),
        frozenset(done),
        frozenset(stale),
        tuple(sorted(conflicts.items())),
    )


# --- Ordering ---


def _order(steps: Mapping[str, FileStep], satisfied: set[str]) -> list[FileStep]:
    """Topological order; a step that became ready runs next, so an old copy goes right
    after its new copy is written (C11). A cycle is broken at exactly one delete.
    """
    pending = dict(steps)
    finished = set(satisfied)
    dependents: dict[str, set[str]] = defaultdict(set)
    for path, step in pending.items():
        for dependency in step.requires | step.clears:
            dependents[dependency].add(path)

    def unmet(step: FileStep) -> set[str]:
        return {path for path in step.requires | step.clears if path not in finished}

    ready = deque(sorted(path for path, step in pending.items() if not unmet(step)))
    ordered: list[FileStep] = []
    while pending:
        if not ready:
            path = _cycle_break(pending, unmet)
            step = pending[path]
            pending[path] = replace(step, requires=step.requires - unmet(step), breaks_cycle=True)
            ready.append(path)
        # Each step enters `ready` once: when its last dependency finished, or when it
        # breaks a cycle (only then is `ready` empty).
        path = ready.popleft()
        ordered.append(pending.pop(path))
        finished.add(path)
        newly = sorted(
            other for other in dependents[path] if other in pending and not unmet(pending[other])
        )
        ready.extendleft(reversed(newly))
    return ordered


def _cycle_break(pending: Mapping[str, FileStep], unmet: Callable[[FileStep], set[str]]) -> str:
    """Pick the step that may run before the note its bytes belong to.

    Every pending step waits on another pending one, so some strongly connected group
    waits only on itself. In that group, take the first path whose wait is only for an
    owner's new copy (never a step that still needs a blocking file removed).
    """
    groups = _strongly_connected(
        {path: sorted(unmet(step) & set(pending)) for path, step in pending.items()}
    )
    member = {path: index for index, group in enumerate(groups) for path in group}
    for group in sorted(groups, key=min):
        inside = set(group)
        if len(group) == 1 or any(
            member[dependency] != member[path]
            for path in group
            for dependency in unmet(pending[path])
        ):
            continue
        # A group that waits only on itself always holds such a step: a step that waits
        # for a removal is waiting on a delete, and a delete waits only for an owner.
        return min(path for path in inside if unmet(pending[path]) <= pending[path].requires)
    raise AssertionError("A blocked dependency survived pruning")  # pragma: no cover


def _strongly_connected(edges: Mapping[str, list[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative so that long chains cannot exhaust the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    groups: list[list[str]] = []
    counter = 0
    for root in sorted(edges):
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, position = work.pop()
            if position == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            successors = edges[node]
            if position < len(successors):
                work.append((node, position + 1))
                successor = successors[position]
                if successor not in index:
                    work.append((successor, 0))
                elif successor in on_stack:
                    low[node] = min(low[node], index[successor])
                continue
            for successor in successors:
                if successor in on_stack:
                    low[node] = min(low[node], low[successor])
            if low[node] == index[node]:
                group = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    group.append(member)
                    if member == node:
                        break
                groups.append(group)
    return groups
