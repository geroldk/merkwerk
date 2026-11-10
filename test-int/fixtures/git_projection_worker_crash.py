"""Child-process fault boundary for the real Git projection worker tests."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

from basic_memory.config import BasicMemoryConfig
from basic_memory.db import DatabaseType, engine_session_factory
from basic_memory.index.git_projection import GitProjectionPipeline, build_git_projection_pipeline
from basic_memory.index.git_projection_worker import GitProjectionWorker
from basic_memory.index.note_content_materialization import LocalNoteContentStorage
from basic_memory.indexing.note_materialization_runner import (
    RepositoryNoteMaterializationPublisher,
)
from basic_memory.models import Project
from basic_memory.repository.git_progress_repository import GitProjectionStage
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.repository.git_layout_repository import GitLayoutRepository, GitRetiredFile
from sqlalchemy.ext.asyncio import AsyncSession
from basic_memory.repository.fastembed_provider import FastEmbedEmbeddingProvider
from basic_memory.repository import semantic_runtime
from tests.fake_embeddings import _embed_documents, _embed_query


async def main() -> None:
    database, repository, project_id, config_path, fault_stage = sys.argv[1:]
    config = BasicMemoryConfig.model_validate_json(Path(config_path).read_text())
    apply = GitProjectionPipeline.apply

    async def crash_after_work(
        self: GitProjectionPipeline,
        stage: GitProjectionStage,
        commit_id: str,
        *,
        graph_base_commit: str | None = None,
    ) -> None:
        await apply(self, stage, commit_id, graph_base_commit=graph_base_commit)
        if stage == fault_stage:
            os._exit(73)

    complete_retirement = GitLayoutRepository.complete_retirement

    async def crash_before_retirement_ack(
        self: GitLayoutRepository, session: AsyncSession, retirement: GitRetiredFile
    ) -> None:
        if fault_stage == "retirement":
            os._exit(76)
        await complete_retirement(self, session, retirement)

    # Files stage faults (F2). The head's paths tell a pure retirement (a path no note
    # keeps) from a reused one.
    live = set(GitProjectRepository(Path(repository)).snapshot().files)
    delete = LocalNoteContentStorage.delete_file_if_matches

    async def crash_around_delete(
        self: LocalNoteContentStorage, path: str, *, expected_checksum: str
    ):
        if fault_stage == "after_writes" and path not in live:
            # The first old file to go, right after its note's new file was written.
            os._exit(77)
        outcome = await delete(self, path, expected_checksum=expected_checksum)
        if fault_stage == "reused_delete" and path in live:
            # A reused path was cleared; its new note is not written yet.
            os._exit(79)
        return outcome

    publish = RepositoryNoteMaterializationPublisher.publish_written_file_state

    async def crash_before_publish(self, request, prepared_write, written_file):
        if fault_stage == "after_write_before_publish":
            # The new bytes are on disk; the row still says 'writing' with the old checksum.
            os._exit(78)
        return await publish(self, request, prepared_write, written_file)

    async with engine_session_factory(Path(database), DatabaseType.FILESYSTEM) as factory:
        async with factory[1]() as session:
            project = await session.get(Project, int(project_id))
            assert project is not None
        pipeline = build_git_projection_pipeline(
            GitProjectRepository(Path(repository)),
            project=project,
            session_maker=factory[1],
            app_config=config,
        )
        # A new interpreter does not inherit pytest's embedder fixture. Use the
        # same deterministic provider methods and model identity in both processes.
        with (
            patch.object(GitProjectionPipeline, "apply", crash_after_work),
            patch.object(GitLayoutRepository, "complete_retirement", crash_before_retirement_ack),
            patch.object(LocalNoteContentStorage, "delete_file_if_matches", crash_around_delete),
            patch.object(
                RepositoryNoteMaterializationPublisher,
                "publish_written_file_state",
                crash_before_publish,
            ),
            patch.object(FastEmbedEmbeddingProvider, "embed_documents", _embed_documents),
            patch.object(FastEmbedEmbeddingProvider, "embed_query", _embed_query),
            patch.object(semantic_runtime, "sqlite_vector_runtime_available", return_value=True),
        ):
            async with GitProjectionWorker(pipeline, poll_interval=0.02).run():
                await asyncio.sleep(15)
        raise RuntimeError("Worker did not reach the process fault boundary")


if __name__ == "__main__":
    asyncio.run(main())
