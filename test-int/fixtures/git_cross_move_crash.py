"""Exit the normal API host at each boundary of a cross-project move (I1, W2).

The parent test restarts the host afterwards; its startup resume must finish, undo or
keep the move, never lose the note. Faults (sys.argv[1] is a JSON request):

after_draft_before_intent           the target draft exists, no intent yet
after_intent_before_delete          the intent is saved (drafted), the source untouched
after_source_git_before_sql         the source published its delete; its SQL did not commit
after_delete_before_commit          the intent says source_deleted; no target commit
after_target_publish_before_record  the target published; its transaction record did not
after_record_before_intent_removed  the record says committed; the intent is still there
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from basic_memory.api.app import app
from basic_memory.index.git_projection_worker import GitProjectionWorker
from basic_memory.repository.git_move_intent_repository import GitMoveIntentRepository
from basic_memory.repository.git_project_repository import GitProjectRepository

EXIT = 83


async def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    fault = request["fault"]
    source_repository = Path(request["source_repository"]).resolve()
    target_repository = Path(request["target_repository"]).resolve()
    save, remove = GitMoveIntentRepository.save, GitMoveIntentRepository.remove
    publish = GitProjectRepository.publish

    def drafted(intent) -> bool:
        """The target draft record exists and its note is written (revision 1)."""
        record = target_repository / "bm-transactions" / f"{intent.transaction_id}.json"
        return record.is_file() and json.loads(record.read_text())["revision"] == 1

    def save_or_die(self, intent):
        # Only at its boundary: the first save of a drafted intent whose draft holds the
        # note already (otherwise the subprocess runs on and the parent test fails).
        if fault == "after_draft_before_intent" and intent.state == "drafted" and drafted(intent):
            os._exit(EXIT)
        save(self, intent)
        if (fault == "after_intent_before_delete" and intent.state == "drafted") or (
            fault == "after_delete_before_commit" and intent.state == "source_deleted"
        ):
            os._exit(EXIT)

    def publish_or_die(self, prepared):
        result = publish(self, prepared)
        if (fault == "after_source_git_before_sql" and self.path == source_repository) or (
            fault == "after_target_publish_before_record" and self.path == target_repository
        ):
            os._exit(EXIT)
        return result

    def remove_or_die(self, move_id):
        if fault == "after_record_before_intent_removed":
            os._exit(EXIT)
        remove(self, move_id)

    async def paused(self, *, vectors):
        # Only the foreground move runs here; the parent restarts the real worker.
        await asyncio.Event().wait()

    with (
        patch.object(GitMoveIntentRepository, "save", save_or_die),
        patch.object(GitMoveIntentRepository, "remove", remove_or_die),
        patch.object(GitProjectRepository, "publish", publish_or_die),
        patch.object(GitProjectionWorker, "_loop", paused),
    ):
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                response = await client.post(
                    f"/v2/projects/{request['source_project_id']}/git/moves",
                    json=request["body"],
                )
                raise RuntimeError(
                    f"Crash boundary not reached: {response.status_code} {response.text}"
                )


if __name__ == "__main__":
    asyncio.run(main())
