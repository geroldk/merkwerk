"""Opt-in local Git projection through BM's guarded file and real index pipeline.

No normal startup or write dependency enables this experimental composition yet.
The caller owns the project and must exclude legacy writers and other workers.
"""

import asyncio
import errno
import hashlib
import json
import os
import stat
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory.config import BasicMemoryConfig
from basic_memory.file_utils import compute_checksum
from basic_memory.index.git_file_projection_plan import (
    BLOCKED_PATH,
    BaseFile,
    DiskView,
    FileProjectionPlan,
    FileStep,
    LiveNote,
    RetiredFile,
    StepKind,
    identify_occupants,
    plan_file_projection,
    publication_proof_candidates,
)
from basic_memory.index.local_dependencies import build_local_markdown_file_indexer
from basic_memory.index.git_project_ownership import require_registered_git_root
from basic_memory.index.note_content_materialization import LocalNoteContentStorage
from basic_memory.indexing.batch_indexer import BatchIndexer
from basic_memory.indexing.index_file_runner import IndexFileExecutor
from basic_memory.indexing.models import IndexFrontmatterUpdate, IndexFrontmatterWriteResult
from basic_memory.indexing.note_materialization_runner import (
    ContentStoreNoteMaterializationFileWriter,
    RepositoryNoteMaterializationPreflight,
    RepositoryNoteMaterializationPublisher,
    RepositoryNoteMaterializationStatusPublisher,
    run_note_materialization,
)
from basic_memory.markdown import EntityParser, MarkdownProcessor
from basic_memory.models import Entity, NoteContent, Project
from basic_memory.read_cache import ReadCacheInvalidator, invalidate_cache
from basic_memory.repository import EntityRepository, ObservationRepository, RelationRepository
from basic_memory.repository.git_progress_repository import (
    GitProgressRepository,
    GitProjectionStage,
)
from basic_memory.repository.git_graph_inputs import git_graph_identities_unchanged
from basic_memory.repository.git_layout_repository import GitLayoutRepository, GitRetiredFile
from basic_memory.repository.git_project_repository import GitProjectRepository, GitSnapshot
from basic_memory.repository.git_projection_repository import (
    GitProjectionConflict,
    GitProjectionSuperseded,
)
from basic_memory.repository.search_repository import create_search_repository
from basic_memory.runtime.note_content import (
    RuntimeNoteMaterializationJobRequest,
    RuntimeNoteMaterializationResult,
    RuntimeNoteMaterializationStatus,
)
from basic_memory.runtime.cleanup import (
    RuntimeGuardedFileDeleteOutcome,
    RuntimeNoteFileDeleteJobRequest,
)
from basic_memory.services import EntityService, FileService
from basic_memory.services.git_project_projection import project_git_content
from basic_memory.services.link_resolver import LinkResolver
from basic_memory.services.search_service import SearchService
from basic_memory.utils import portable_path_key
from basic_memory.workspace_context import current_workspace_permalink_context


# --- Files stage helpers (F2) ---

CHANGED_DURING_CLEANUP = "Old file changed during guarded Git cleanup"
SQL_BEHIND = "SQL content has not reached this Git target"


@dataclass(frozen=True, slots=True)
class _WriteProblem:
    reason: str
    # The note changed in SQL before publish: a newer head explains it.
    stale: bool = False


@dataclass(frozen=True, slots=True)
class RetirementRowsOwnVacatedFiles:
    """Orphaned-write cleanup for the Git host: nothing to do here.

    The materializer deletes a file it wrote when the note moved away before publish.
    On the Git host every path a note leaves already has a retirement row, recorded by
    accept() or the layout projection, and the files stage deletes those bytes only
    after the note's new file is on disk. Deleting here could remove the note's only
    copy (a status publish for bytes already in place, then a concurrent move; C13).
    """

    async def enqueue_note_file_delete(self, request: RuntimeNoteFileDeleteJobRequest) -> None:
        _ = request


async def materialize_published_note(
    request: RuntimeNoteMaterializationJobRequest,
    *,
    session_maker: async_sessionmaker[AsyncSession],
    file_service: FileService,
) -> RuntimeNoteMaterializationResult:
    """Write one accepted note through BM's guarded materializer.

    The same preflight, writer, publisher and status publisher as the recovery path, so
    the db_version claim and the file guard apply unchanged; only vacated files are left
    to the retirement rows (see RetirementRowsOwnVacatedFiles).
    """
    storage = LocalNoteContentStorage(file_service)
    return await run_note_materialization(
        request,
        preflight=RepositoryNoteMaterializationPreflight(session_maker=session_maker),
        writer=ContentStoreNoteMaterializationFileWriter(storage),
        publisher=RepositoryNoteMaterializationPublisher(session_maker=session_maker),
        status_publisher=RepositoryNoteMaterializationStatusPublisher(session_maker=session_maker),
        cleanup_enqueuer=RetirementRowsOwnVacatedFiles(),
    )


def read_projection_disk(root: Path, live: Iterable[str], retired: Iterable[str]) -> DiskView:
    """Look once at every path the files stage may touch (runs in a worker thread)."""
    live = tuple(live)
    retired = tuple(retired)
    files: dict[str, str] = {}
    directories: set[str] = set()
    live_paths = set(live)
    for path in {*live, *retired}:
        full = root / path
        if path not in live_paths and full.is_symlink():
            # Basic Memory writes regular files only: a link at a path a note left is
            # someone else's, never deleted or renamed; the row is simply completed.
            continue
        if full.is_dir():
            directories.add(path)
        elif full.is_file():
            files[path] = hashlib.sha256(full.read_bytes()).hexdigest()

    # A file where a folder must be (file to directory, C8).
    ancestor_is_file: dict[str, bool] = {}
    blocking: dict[str, tuple[str, ...]] = {}
    for path in live:
        parents = [str(parent) for parent in PurePosixPath(path).parents][:-1]
        for parent in parents:
            if parent not in ancestor_is_file:
                ancestor_is_file[parent] = (root / parent).is_file()
        if blockers := tuple(sorted(parent for parent in parents if ancestor_is_file[parent])):
            blocking[path] = blockers

    # A folder where a file must be (directory to file, C8).
    folder_files: dict[str, tuple[str, ...]] = {}
    foreign: set[str] = set()
    for path in set(live) & directories:
        found: list[str] = []
        pending = [root / path]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        found.append(Path(entry.path).relative_to(root).as_posix())
                    else:
                        foreign.add(path)
        folder_files[path] = tuple(sorted(found))

    # A live path that is a retired file's own entry: a case- or Unicode-only rename on a
    # file system that folds them (C1). On Linux only a hard link behaves like this.
    retired_by_key: dict[str, list[str]] = defaultdict(list)
    for path in retired:
        retired_by_key[portable_path_key(path)].append(path)
    aliases: dict[str, str] = {}
    for path in live:
        if path not in files:
            continue
        for other in retired_by_key.get(portable_path_key(path), ()):
            if other != path and one_regular_file(root, path, other):
                aliases[path] = other
                break
    return DiskView(
        files=files,
        directories=frozenset(directories),
        blocking_files=blocking,
        folder_files=folder_files,
        folder_foreign=frozenset(foreign),
        aliases=aliases,
    )


def one_regular_file(root: Path, left: str, right: str) -> bool:
    """Whether two project paths are one directory entry of a regular file (C1).

    lstat, never stat: a symlink reads the same bytes but is an entry of its own, and
    renaming it over the note would replace the note with a link to itself.
    """
    try:
        first, second = os.lstat(root / left), os.lstat(root / right)
    except OSError:
        return False
    return (
        stat.S_ISREG(first.st_mode)
        and stat.S_ISREG(second.st_mode)
        and (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)
    )


def drop_second_name(root: Path, old: str, live: str) -> None:
    """After the alias rename, remove `old` if it is still a name of its own.

    rename(2) between two hard links of one file does nothing (POSIX), so on Linux the
    old spelling would stay as an unmanaged second name. On macOS or Windows the
    rename leaves one entry under the published spelling, and `old` is not listed.
    Only a name listed under its exact spelling, beside the live one, is unlinked;
    unlinking a second hard link never touches the note's bytes.
    """
    old_full, live_full = root / old, root / live
    if old_full.name == live_full.name and old_full.parent.samefile(live_full.parent):
        return  # one folder reached through two spellings: one entry
    if (
        old_full.name in os.listdir(old_full.parent)
        and live_full.name in os.listdir(live_full.parent)
        and one_regular_file(root, old, live)
    ):
        old_full.unlink()


def prune_empty_folders(root: Path, path: str) -> None:
    """Remove the folders that deleting `path` left empty, deepest first.

    Never the project root, never a hidden folder (so never .basic-memory), never a
    folder reached through a symlink, and only a folder that is truly empty: rmdir
    refuses anything else.
    """
    parents = [parent for parent in PurePosixPath(path).parents if parent != PurePosixPath(".")]
    # Trigger: a folder on the way is a symlink.
    # Why: rmdir through it would remove empty folders outside the project.
    # Outcome: nothing is pruned for this path.
    if any((root / parent).is_symlink() for parent in parents):
        return
    for parent in parents:
        if parent.name.startswith("."):
            return
        try:
            (root / parent).rmdir()
        except OSError as error:
            # Trigger: not empty (another file, or one an editor just created), already
            #   gone, not a folder, or refused (a folder open in another program).
            # Why: the delete that emptied it already happened; failing the pass here
            #   gains nothing and would only repeat the pass.
            # Outcome: this folder and everything above it stay.
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT, errno.ENOTDIR}:
                logger.warning(
                    "Git files stage left a folder it could not remove",
                    folder=parent.as_posix(),
                    error=str(error),
                )
            return


def clear_empty_folder(root: Path, path: str) -> bool:
    """Remove an empty folder tree where a note's file must go; True when the path is free.

    Only folders that hold nothing at all go, deepest first, with rmdir. A file, a
    symlink or a hidden folder anywhere inside keeps the tree, and the path stays
    blocked (C8).
    """
    top = root / path
    if top.is_symlink():
        return False
    for current, _folders, _files in os.walk(top, topdown=False):
        if any(part.startswith(".") for part in Path(current).relative_to(top).parts):
            continue
        try:
            os.rmdir(current)
        except OSError:
            pass  # anything left inside keeps the path blocked; reported by the caller
    return not os.path.lexists(top)


class PublishedGitIndexWriter:
    """Indexing may derive state, but never normalize already published Markdown."""

    async def write_frontmatter(
        self, update: IndexFrontmatterUpdate
    ) -> IndexFrontmatterWriteResult:
        raise GitProjectionConflict(f"Indexing would rewrite published Git bytes: {update.path}")


@dataclass(frozen=True, slots=True)
class GitProjectionPipeline:
    repository: GitProjectRepository
    project_id: int
    project_external_id: str
    session_maker: async_sessionmaker[AsyncSession]
    file_service: FileService
    file_indexer: IndexFileExecutor
    search_service: SearchService
    vector_key: str
    read_cache: ReadCacheInvalidator | None = None
    graph_key: str = "1"

    async def apply(
        self,
        stage: GitProjectionStage,
        commit_id: str,
        *,
        graph_base_commit: str | None = None,
    ) -> None:
        await self.require_registration()
        if stage == "content":
            await self.recover_content(commit_id)
            return

        snapshot = await asyncio.to_thread(self.repository.snapshot, commit_id)
        invalidation = (
            invalidate_cache(self.read_cache, self.project_external_id)
            if self.read_cache is not None
            else nullcontext()
        )
        if stage == "files":
            async with invalidation:
                await self.project_files(snapshot)
            return

        graph_paths: set[str] | None = None
        if stage == "graph" and graph_base_commit is not None:
            previous = await asyncio.to_thread(self.repository.snapshot, graph_base_commit)
            if git_graph_identities_unchanged(
                previous.files,
                previous.manifest.note_ids,
                snapshot.files,
                snapshot.manifest.note_ids,
            ):
                graph_paths = {
                    path
                    for path, content in snapshot.files.items()
                    if previous.files[path] != content
                }
        async with invalidation:
            if stage == "vectors" and not await self.search_service.semantic_effectively_enabled():
                # A missing backend must remain visible as unavailable, never "ready".
                raise GitProjectionConflict("Semantic backend unavailable; vectors remain pending")
            for note in snapshot.manifest.notes:
                if graph_paths is not None and note.path not in graph_paths:
                    continue
                await self.require_head(commit_id)
                async with self.session_maker() as session:
                    entity = await session.scalar(
                        select(Entity).where(
                            Entity.project_id == self.project_id,
                            Entity.external_id == str(note.external_id),
                            Entity.file_path == note.path,
                        )
                    )
                    content = (
                        await session.get(NoteContent, entity.id) if entity is not None else None
                    )
                    if content is None or content.markdown_content != snapshot.files[note.path]:
                        # A local write already moved SQL past this target: superseded.
                        await self.require_head(commit_id)
                        raise GitProjectionConflict(SQL_BEHIND)
                    entity_id = content.entity_id
                await self.require_file(snapshot, note.path)
                if stage == "graph":
                    indexed = await self.file_indexer.index_file(note.path, source="git-projection")
                    if (
                        indexed.content_superseded
                        or indexed.external_id != str(note.external_id)
                        or indexed.content_checksum
                        != await compute_checksum(snapshot.files[note.path])
                    ):
                        raise GitProjectionConflict(
                            "Index did not publish the target note generation"
                        )
                    await self.require_file(snapshot, note.path)
                else:
                    # The files stage has its own path; only vectors remain here.
                    vectors = await self.search_service.sync_entity_vectors(entity_id)
                    if (
                        vectors.entities_failed
                        or vectors.entities_deferred
                        # An unchanged entity is both synced and skipped by BM's
                        # repository; these counters are not disjoint outcomes.
                        or not (vectors.entities_synced == 1 or vectors.entities_skipped == 1)
                    ):
                        raise GitProjectionConflict(
                            "Vector sync incomplete: " + "; ".join(vectors.sample_errors)
                        )
            if stage == "graph":
                await self.search_service.repository.purge_stale_search_rows()
            else:
                await self.search_service.repository.delete_stale_vector_rows()
                await self.search_service.repository.reconcile_vector_index()
        await self.require_head(commit_id)

    async def recover_content(
        self,
        commit_id: str,
        *,
        known_ancestor: GitSnapshot | None = None,
    ) -> None:
        await self.require_registration()
        await project_git_content(
            self.repository,
            project_id=self.project_id,
            session_maker=self.session_maker,
            read_cache=self.read_cache,
            commit_id=commit_id,
            known_ancestor=known_ancestor,
        )

    # --- Files stage (F2): every new file first, every old file after ---

    async def project_files(self, snapshot: GitSnapshot) -> None:
        """Make disk hold `snapshot`; a note's old file goes only after its new one is there.

        Disk is a delayed projection of Git. One look at SQL, disk and the retirement
        rows classifies every path; the planner orders the steps; each delete re-checks
        the bytes right before it runs. HEAD is checked at the start and at the end only:
        a write is fenced by the materializer's db_version claim, a delete by its
        checksum, so a commit arriving mid-pass never strands half a swap (C5). Work
        done for an older target stays valid for a newer one wherever path, UUID and
        bytes agree, and the next pass proves that again with one hash per note.
        """
        commit_id = snapshot.commit_id
        await self.require_head(commit_id)
        notes, requests, retirements = await self._read_files_state(snapshot)
        retired = [
            RetiredFile(row.file_path, frozenset(row.expected_checksums))
            for row in retirements.values()
        ]
        root = self.file_service.base_path
        disk = await asyncio.to_thread(
            read_projection_disk, root, [note.path for note in notes], list(retirements)
        )
        if candidates := publication_proof_candidates(notes, retired, disk):
            disk = replace(disk, published=await self._published_bytes(commit_id, candidates, disk))
        occupants = identify_occupants(notes, retired, disk)
        if any(path in disk.files and path not in occupants for path in retirements):
            # Only then is the last fully written commit worth reading.
            if (base := await self._base_files(snapshot)) is not None:
                occupants = identify_occupants(notes, retired, disk, base)
        plan = plan_file_projection(notes, retired, disk, occupants)
        await self._carry_out(plan, snapshot, notes, requests, retirements)

    async def _read_files_state(
        self, snapshot: GitSnapshot
    ) -> tuple[
        list[LiveNote], dict[str, RuntimeNoteMaterializationJobRequest], dict[str, GitRetiredFile]
    ]:
        """One SQL read for every note and every retirement row."""
        layout = GitLayoutRepository(self.project_id)
        async with self.session_maker() as session:
            retirements = {row.file_path: row for row in await layout.retirements(session)}
            rows = {
                row.external_id: row
                for row in await session.execute(
                    select(
                        Entity.id,
                        Entity.external_id,
                        Entity.file_path,
                        Entity.sync_checksum,
                        NoteContent.markdown_content,
                        NoteContent.db_version,
                        NoteContent.db_checksum,
                        NoteContent.file_version,
                        NoteContent.file_checksum,
                        NoteContent.file_write_status,
                    )
                    .join(NoteContent, NoteContent.entity_id == Entity.id)
                    .where(Entity.project_id == self.project_id)
                )
            }
        notes: list[LiveNote] = []
        requests: dict[str, RuntimeNoteMaterializationJobRequest] = {}
        for note in snapshot.manifest.notes:
            content = snapshot.files[note.path]
            checksum = await compute_checksum(content)
            row = rows.get(str(note.external_id))
            # Trigger: no SQL row holds exactly these bytes for this UUID at this path.
            # Why: the materializer writes from SQL; a concurrent accept() may already
            #   have moved the note past this target (C6).
            # Outcome: the note is skipped, the others go on; decided at the end.
            if row is None or row.file_path != note.path or row.markdown_content != content:
                notes.append(LiveNote(note.path, checksum, ready=False))
                continue
            notes.append(
                LiveNote(
                    note.path,
                    checksum,
                    synced=row.file_write_status == "synced"
                    and row.file_checksum == row.db_checksum
                    and row.file_version == row.db_version,
                    file_checksum=row.file_checksum,
                    sync_checksum=row.sync_checksum,
                )
            )
            requests[note.path] = RuntimeNoteMaterializationJobRequest(
                project_id=self.project_id,
                entity_id=row.id,
                db_version=row.db_version,
                db_checksum=row.db_checksum,
                source="git-projection",
            )
        return notes, requests, retirements

    async def _published_bytes(
        self, commit_id: str, candidates: Iterable[str], disk: DiskView
    ) -> frozenset[str]:
        """Paths whose bytes Git once published there: Basic Memory's, not an editor's (C7).

        Bytes left by a write that died before its publish, or by an orphaned write, are
        recognized this way. Only for files nothing in SQL explains, so normally never.
        """
        proven = set()
        for path in candidates:
            data = await self.file_service.read_file_bytes(path)
            if hashlib.sha256(data).hexdigest() != disk.files[path]:
                continue  # changed since the first look; the next pass looks again
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue  # Git only ever published UTF-8 notes
            if await asyncio.to_thread(
                self.repository.contains_published_note_content, commit_id, path, text
            ):
                proven.add(path)
        return frozenset(proven)

    async def _base_files(self, snapshot: GitSnapshot) -> dict[str, BaseFile] | None:
        """The last commit this stage fully wrote, to name the owner of old bytes (C4)."""
        rows = {
            row.stage: row
            for row in await GitProgressRepository(self.project_id, self.session_maker).read()
        }
        files = rows.get("files")
        base_commit = files.completed_commit if files is not None else None
        if base_commit is None or not await asyncio.to_thread(
            self.repository.is_ancestor, base_commit, snapshot.commit_id
        ):
            return None
        base = await asyncio.to_thread(self.repository.snapshot, base_commit)
        live = snapshot.manifest.note_ids_by_uuid
        return {
            path: BaseFile(await compute_checksum(content), live.get(base.manifest.note_ids[path]))
            for path, content in base.files.items()
        }

    async def _carry_out(
        self,
        plan: FileProjectionPlan,
        snapshot: GitSnapshot,
        notes: list[LiveNote],
        requests: dict[str, RuntimeNoteMaterializationJobRequest],
        retirements: dict[str, GitRetiredFile],
    ) -> None:
        """Run the steps in order; skip a step whose prerequisite did not succeed."""
        storage = LocalNoteContentStorage(self.file_service)
        verified = set(plan.done)
        cleared: set[str] = set()
        conflicts = [reason for _, reason in plan.conflicts]
        stale = [SQL_BEHIND] if plan.stale else []
        final_paths = {step.path for step in plan.steps if step.final}
        others_on_disk: bool | None = None
        for step in plan.steps:
            if step.final:
                # Trigger: the owner of the bytes this step deletes is unknown.
                # Why: only when every other note is on disk, and no write of this pass
                #   failed (also one of these steps), can no note lose its copy.
                # Outcome: after any conflict these old bytes simply stay.
                if others_on_disk is None:
                    # Final steps run last, so the other notes' state no longer changes.
                    others_on_disk = all(
                        note.path in verified for note in notes if note.path not in final_paths
                    )
                if conflicts or stale or not others_on_disk:
                    break
            if not (step.requires <= verified and step.clears <= cleared):
                continue
            if step.kind is StepKind.complete:
                await self._complete(retirements[step.path])
                cleared.add(step.path)
                if step.path not in snapshot.files:
                    # A crash between an old file's delete and its prune left the folders.
                    await asyncio.to_thread(
                        prune_empty_folders, self.file_service.base_path, step.path
                    )
                continue
            if step.kind is StepKind.delete:
                if await self._retire_file(storage, step):
                    await self._complete(retirements[step.path])
                    cleared.add(step.path)
                else:
                    conflicts.append(CHANGED_DURING_CLEANUP)
                continue
            # settle, write or replace: the note at this path gets its file.
            if step.kind is StepKind.replace:
                assert step.expected_checksum is not None
                # No HEAD check and no prune between this delete and its write.
                deleted = await storage.delete_file_if_matches(
                    step.path, expected_checksum=step.expected_checksum
                )
                if deleted is RuntimeGuardedFileDeleteOutcome.changed:
                    conflicts.append(CHANGED_DURING_CLEANUP)
                    continue
            elif (self.file_service.base_path / step.path).is_dir():
                # Trigger: a folder stands where this note's file goes (directory to file,
                #   C8): left by the retired files that just went, by a crash before their
                #   prune, or by an editor.
                # Why: the materializer cannot write onto a folder; anything inside that
                #   Basic Memory does not own must stay.
                # Outcome: an empty tree is removed and the note written; else blocked.
                if not await asyncio.to_thread(
                    clear_empty_folder, self.file_service.base_path, step.path
                ):
                    conflicts.append(BLOCKED_PATH)
                    continue
            if step.breaks_cycle:
                logger.info(
                    "Git files stage writes one note of a path cycle first",
                    project_id=self.project_id,
                    path=step.path,
                )
            problem = await self._write(requests[step.path], snapshot, step.path)
            if problem is None:
                verified.add(step.path)
                if step.retire:
                    await self._complete(retirements[step.path])
            elif problem.stale:
                stale.append(problem.reason)
            else:
                conflicts.append(problem.reason)

        # --- The outcome: conflicts, then a newer head, then a full check of disk ---
        if conflicts:
            raise GitProjectionConflict("; ".join(dict.fromkeys(conflicts)))
        if stale:
            # Trigger: SQL moved past this target, or a write was superseded before publish.
            # Why: accept() commits SQL only after publishing, so a newer head explains it.
            # Outcome: superseded (next pass at once), else a real conflict.
            await self.require_head(snapshot.commit_id)
            raise GitProjectionConflict("; ".join(dict.fromkeys(stale)))
        # Every note, also those that needed nothing: a delete after classification must
        # never take a live file away unnoticed (C12).
        for path in snapshot.files:
            await self.require_file(snapshot, path)
        await self.require_head(snapshot.commit_id)

    async def _write(
        self,
        request: RuntimeNoteMaterializationJobRequest,
        snapshot: GitSnapshot,
        path: str,
    ) -> _WriteProblem | None:
        result = await materialize_published_note(
            request, session_maker=self.session_maker, file_service=self.file_service
        )
        if result.status is RuntimeNoteMaterializationStatus.written:
            if await self.file_service.read_file_bytes(path) == snapshot.files[path].encode():
                return None
            return _WriteProblem("File differs from published Git bytes; import requires review")
        return _WriteProblem(
            f"File materialization: {result.status.value}",
            stale=result.status
            in {RuntimeNoteMaterializationStatus.stale, RuntimeNoteMaterializationStatus.missing},
        )

    async def _retire_file(self, storage: LocalNoteContentStorage, step: FileStep) -> bool:
        """Remove a path a note left; False when an editor changed its bytes meanwhile."""
        root = self.file_service.base_path
        for live_path in step.aliases:
            # Trigger: the old path and a live path are one directory entry of a regular
            #   file (case- or Unicode-only rename on macOS/Windows; on Linux only a hard
            #   link). A symlink is not: it is an entry of its own.
            # Why: deleting the old spelling would delete the note itself (C1).
            # Outcome: no delete; the entry takes the published spelling, as a case-only
            #   rename does in the local runtime (InlineNoteFileDeleteEnqueuer). A second
            #   hard link, which the rename leaves in place, is unlinked.
            if await asyncio.to_thread(one_regular_file, root, step.path, live_path):
                await self.file_service.move_file(step.path, live_path)
                await asyncio.to_thread(drop_second_name, root, step.path, live_path)
                return True
        assert step.expected_checksum is not None
        deleted = await storage.delete_file_if_matches(
            step.path, expected_checksum=step.expected_checksum
        )
        if deleted is RuntimeGuardedFileDeleteOutcome.changed:
            return False
        await asyncio.to_thread(prune_empty_folders, root, step.path)
        return True

    async def _complete(self, retirement: GitRetiredFile) -> None:
        # Fenced by retirement_id: a row accept() refreshed meanwhile stays for next pass.
        async with self.session_maker() as session, session.begin():
            await GitLayoutRepository(self.project_id).complete_retirement(session, retirement)

    async def require_head(self, commit_id: str) -> None:
        if commit_id != await asyncio.to_thread(self.repository.head):
            raise GitProjectionSuperseded("Git head advanced; retry the latest published target")

    async def require_registration(self) -> None:
        async with self.session_maker() as session:
            await require_registered_git_root(
                session,
                project_id=self.project_id,
                project_external_id=self.project_external_id,
                root=self.file_service.base_path,
            )

    async def require_file(self, snapshot: GitSnapshot, path: str) -> None:
        actual = await self.file_service.read_file_bytes(path)
        if actual != snapshot.files[path].encode("utf-8"):
            raise GitProjectionConflict(
                "File differs from published Git bytes; import requires review"
            )


def build_git_projection_pipeline(
    repository: GitProjectRepository,
    *,
    project: Project,
    session_maker: async_sessionmaker[AsyncSession],
    app_config: BasicMemoryConfig,
    read_cache: ReadCacheInvalidator | None = None,
) -> GitProjectionPipeline:
    """Explicit local composition; no global config, new daemon or default enablement."""
    root = Path(project.path)
    parser = EntityParser(root)
    files = FileService(
        root, MarkdownProcessor(parser, app_config=app_config), app_config=app_config
    )
    entities = EntityRepository(project.id)
    observations = ObservationRepository(project.id)
    relations = RelationRepository(project.id)
    searches = create_search_repository(session_maker, project.id, app_config)
    search = SearchService(searches, entities, files, session_maker)
    entity_service = EntityService(
        parser,
        entities,
        observations,
        relations,
        files,
        LinkResolver(entities, search, session_maker, app_config),
        session_maker,
        search_service=search,
        app_config=app_config,
    )
    batch = BatchIndexer(
        project_id=project.id,
        app_config=app_config,
        entity_service=entity_service,
        entity_repository=entities,
        observation_repository=observations,
        relation_repository=relations,
        search_service=search,
        file_writer=PublishedGitIndexWriter(),
        session_maker=session_maker,
    )
    indexer = build_local_markdown_file_indexer(
        project_id=project.id,
        file_service=files,
        session_maker=session_maker,
        entity_repository=entities,
        batch_indexer=batch,
        search_service=search,
    )
    vector_key = json.dumps(
        {
            "version": 1,
            "model": searches.configured_embedding_model,
            "index": searches.configured_vector_index,
            "dimensions": app_config.semantic_embedding_dimensions,
            "max_tokens": app_config.semantic_embedding_max_tokens,
        },
        sort_keys=True,
    )
    workspace = current_workspace_permalink_context()
    graph_key = json.dumps(
        {
            "version": 2,
            "permalinks_include_project": app_config.permalinks_include_project,
            "ensure_frontmatter_on_sync": app_config.ensure_frontmatter_on_sync,
            "workspace": (
                workspace.workspace_slug
                if workspace is not None and workspace.should_prefix_permalinks
                else None
            ),
        },
        sort_keys=True,
    )
    return GitProjectionPipeline(
        repository,
        project.id,
        project.external_id,
        session_maker,
        files,
        indexer,
        search,
        vector_key,
        read_cache,
        graph_key,
    )
