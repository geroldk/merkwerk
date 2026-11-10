"""Lifecycle-owned, bounded Git catch-up; Git + SQL survive lost wakeups/restarts."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger

from basic_memory.index.git_projection import GitProjectionPipeline
from basic_memory.index.git_project_ownership import own_git_project
from basic_memory.repository.git_progress_repository import (
    GitProgressRepository,
    GitProjectionStage,
    GitStageProgress,
)
from basic_memory.repository.git_projection_repository import (
    GitProjectionDeferred,
    GitProjectionSuperseded,
)


class GitPassOutcome(StrEnum):
    done = "done"
    failed = "failed"
    # A newer commit arrived; the work done stays valid and the next pass starts at once.
    superseded = "superseded"
    # Notes wait for another project to release their UUIDs (I1); nothing failed.
    deferred = "deferred"


# A waiting project is woken when its owner's content stage completes; polling is only
# the fallback, so it runs at a longer, capped interval instead of backing off.
DEFERRED_POLL_FACTOR = 6
DEFERRED_POLL_CAP = 60.0


@dataclass(frozen=True, slots=True)
class GitProjectProgress:
    source_commit: str
    stages: tuple[GitStageProgress, ...]
    vector_key: str
    graph_key: str = "1"

    @property
    def caught_up(self) -> bool:
        return len(self.stages) == 4 and all(
            stage.target_commit == self.source_commit
            and stage.completed_commit == self.source_commit
            and stage.running_commit is None
            and stage.last_error is None
            and stage.projection_key
            == (
                self.vector_key
                if stage.stage == "vectors"
                else self.graph_key
                if stage.stage == "graph"
                else "1"
            )
            for stage in self.stages
        )


class GitProjectionWorker:
    """Two serial lanes: slow vectors cannot delay content/file/graph catch-up.

    The context owns both tasks. There is no in-memory job backlog: each pass
    samples the latest Git head and durable checkpoints. Partial failed stages
    are retried as a whole; work already on disk costs one hash per note then. A
    newer head supersedes a pass at once, without a failure or a backoff. A crash
    before acknowledgement safely repeats work.
    Local root/repository bindings and OS locks enforce unique ownership.
    """

    def __init__(
        self,
        pipeline: GitProjectionPipeline,
        *,
        poll_interval: float = 5,
        content_gate: asyncio.Lock | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.pipeline = pipeline
        self.progress = GitProgressRepository(pipeline.project_id, pipeline.session_maker)
        self.poll_interval = poll_interval
        self.content_gate = content_gate
        self._wake_index = asyncio.Event()
        self._wake_vectors = asyncio.Event()
        self._running = False
        self.startup_content_commit: str | None = None
        # In memory only: which notes the content stage waits for, at which target.
        self.deferral: GitProjectionDeferred | None = None
        self.deferral_commit: str | None = None

    @property
    def deferred_interval(self) -> float:
        return max(
            self.poll_interval, min(self.poll_interval * DEFERRED_POLL_FACTOR, DEFERRED_POLL_CAP)
        )

    def notify(self) -> None:
        """Optional latency hint after publication; polling never depends on delivery."""
        self._wake_index.set()
        self._wake_vectors.set()

    def wake_index(self) -> None:
        """Another project released UUIDs this one may wait for: retry content now."""
        self._wake_index.set()

    def _content_deferred(self, commit_id: str, deferral: GitProjectionDeferred) -> None:
        self.deferral, self.deferral_commit = deferral, commit_id
        if (identities := self.pipeline.identities) is not None:
            identities.wait_for(self.pipeline.project_id, deferral.owner_project_ids)

    def _content_settled(self) -> None:
        """The content stage ran to completion: wake the projects that wait for this one."""
        self.deferral = self.deferral_commit = None
        if (identities := self.pipeline.identities) is not None:
            identities.settled(self.pipeline.project_id)

    async def status(self) -> GitProjectProgress:
        await self.pipeline.require_registration()
        return GitProjectProgress(
            await asyncio.to_thread(self.pipeline.repository.head),
            await self.progress.read(),
            self.pipeline.vector_key,
            self.pipeline.graph_key,
        )

    @asynccontextmanager
    async def run(
        self, *, recover_content_before_yield: bool = False
    ) -> AsyncIterator["GitProjectionWorker"]:
        if self._running:
            raise RuntimeError("Git projection worker is already running")
        self._running = True
        self.startup_content_commit = None
        try:
            async with (
                own_git_project(
                    self.pipeline.repository,
                    project_id=self.pipeline.project_id,
                    project_external_id=self.pipeline.project_external_id,
                    project_root=self.pipeline.file_service.base_path,
                    session_maker=self.pipeline.session_maker,
                ),
                asyncio.TaskGroup() as group,
            ):
                if recover_content_before_yield:
                    # Only SQL content gates admission. Slow files/graph/vectors
                    # start in the background after this explicit startup barrier.
                    commit_id = await asyncio.to_thread(self.pipeline.repository.head)
                    try:
                        await self.pipeline.recover_content(commit_id)
                    except GitProjectionDeferred as deferral:
                        # Trigger: notes wait for another project to release their UUIDs.
                        # Why: one waiting project must not stop the host (P4); a real
                        #   duplicate still raises and refuses the start, loudly.
                        # Outcome: no SQL basis is claimed, so writes here are refused
                        #   until the loop completes the content stage.
                        self._content_deferred(commit_id, deferral)
                        logger.warning(
                            "Git projection deferred at startup",
                            project_id=self.pipeline.project_id,
                            commit_id=commit_id,
                            reason=str(deferral),
                        )
                    else:
                        self.startup_content_commit = commit_id
                        self._content_settled()
                index = group.create_task(self._loop(vectors=False), name="git-projection-index")
                vectors = group.create_task(self._loop(vectors=True), name="git-projection-vectors")
                try:
                    yield self
                finally:
                    index.cancel()
                    vectors.cancel()
        finally:
            self._running = False

    async def _loop(self, *, vectors: bool) -> None:
        wake = self._wake_vectors if vectors else self._wake_index
        failures = 0
        while True:
            # Clear before work: a publication during the pass schedules an immediate next pass.
            wake.clear()
            try:
                outcome = await self._pass(vectors=vectors)
            except Exception:
                # The lifecycle boundary retries storage/DB failures too. Cancellation
                # propagates; a failed checkpoint write can never acknowledge the stage.
                logger.exception("Git projection pass failed", project_id=self.pipeline.project_id)
                outcome = GitPassOutcome.failed
            if outcome is GitPassOutcome.deferred:
                # Trigger: notes wait for another project (I1).
                # Why: nothing failed; the owner's completed content stage wakes this
                #   lane, so polling is a capped fallback, not an exponential backoff.
                # Outcome: the failure count resets and the next poll waits longer.
                failures = 0
                delay = self.deferred_interval
            elif outcome is GitPassOutcome.superseded:
                # Trigger: a newer commit replaced the target during the pass.
                # Why: nothing failed, and the work done stays valid for the newer head;
                #   a backoff would only leave disk behind Git for longer.
                # Outcome: failures stay as they are. The index lane re-targets the newer
                #   head right away; vectors wait for the index lane, which wakes them.
                if not vectors:
                    await asyncio.sleep(0)
                    continue
                delay = self.poll_interval
            else:
                failures = 0 if outcome is GitPassOutcome.done else min(failures + 1, 6)
                delay = self.poll_interval * (2**failures)
            try:
                await asyncio.wait_for(wake.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _pass(self, *, vectors: bool) -> GitPassOutcome:
        await self.pipeline.require_registration()
        if not vectors:
            commit_id = await asyncio.to_thread(self.pipeline.repository.head)
            await self.progress.target(
                commit_id, vector_key=self.pipeline.vector_key, graph_key=self.pipeline.graph_key
            )
        stages: tuple[GitProjectionStage, ...] = (
            ("vectors",) if vectors else ("content", "files", "graph")
        )
        for stage in stages:
            # Content/layout recovery can invalidate graph reuse. Never claim an
            # earlier cached basis after that SQL work has changed identities.
            rows = {row.stage: row for row in await self.progress.read()}
            row = rows.get(stage)
            if row is None:
                return GitPassOutcome.done
            if stage == "vectors" and rows["graph"].completed_commit != row.target_commit:
                return GitPassOutcome.done
            if (
                row.completed_commit == row.target_commit
                and row.running_commit is None
                and row.last_error is None
            ):
                continue
            run_id = await self.progress.begin(
                stage, row.target_commit, reusable_commit=row.reusable_commit
            )
            if run_id is None:
                return GitPassOutcome.failed
            try:
                gate = self.content_gate if stage == "content" else None
                async with gate if gate is not None else nullcontext():
                    if stage == "graph":
                        await self.pipeline.apply(
                            stage, row.target_commit, graph_base_commit=row.reusable_commit
                        )
                    else:
                        await self.pipeline.apply(stage, row.target_commit)
                    await self.pipeline.require_head(row.target_commit)
            except GitProjectionSuperseded as error:
                await self.progress.finish(stage, row.target_commit, run_id, error="superseded")
                logger.info(
                    "Git projection stage superseded by a newer head",
                    project_id=self.pipeline.project_id,
                    stage=stage,
                    commit_id=row.target_commit,
                    reason=str(error),
                )
                return GitPassOutcome.superseded
            except GitProjectionDeferred as deferral:
                await self.progress.finish(stage, row.target_commit, run_id, error=str(deferral))
                self._content_deferred(row.target_commit, deferral)
                logger.info(
                    "Git projection content waits for another project",
                    project_id=self.pipeline.project_id,
                    commit_id=row.target_commit,
                    reason=str(deferral),
                )
                return GitPassOutcome.deferred
            except Exception as error:
                await self.progress.finish(stage, row.target_commit, run_id, error=str(error))
                if stage == "content":
                    self.deferral = self.deferral_commit = None
                logger.warning(
                    "Git projection stage pending after failure",
                    project_id=self.pipeline.project_id,
                    stage=stage,
                    commit_id=row.target_commit,
                    error=str(error),
                )
                return GitPassOutcome.failed
            if not await self.progress.finish(stage, row.target_commit, run_id):
                return GitPassOutcome.failed
            if stage == "content":
                self._content_settled()
            if stage == "graph":
                self._wake_vectors.set()
        return GitPassOutcome.done
