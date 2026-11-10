"""Recompute entity file path conflict keys that went stale under Git relocations.

Until commit 72e2922a, ``GitLayoutRepository.apply`` relocated a note (after a peer sync,
a restore, or recovery after a failed publication) with a Core UPDATE that set
``file_path`` but not ``file_path_key`` or ``file_path_permalink``. A Core statement skips
the ``Entity`` validators that keep those columns in step, so such a row kept the keys of
its old path: a later note written at the vacated path was refused as "equivalent", and a
real case or Unicode clash at the new path went unnoticed. The fix covers future
relocations only; this revision repairs the rows written before it.

The keys derive from ``file_path`` alone, so recomputing them is exact. Only rows whose
stored keys differ are updated, so correct rows (every row on a host without Git
relocations) stay untouched, and running it twice changes nothing.

Not repaired here: ``created_at``/``updated_at`` of the same rows. The old UPDATE stored
an aware UTC value that SQLite kept as naive wall time, so on a host east of UTC those
rows read back shifted by the offset. Nothing in the row says which rows were relocated
that way, so no migration can tell a shifted time from a correct one.

Revision ID: 0049
Revises: 0048
Create Date: 2026-11-10 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# The keys must be byte-identical to what the running application computes.
from basic_memory.utils import file_path_conflict_keys

revision: str = "0049"
down_revision: Union[str, None] = "0048"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

REPAIR_BATCH_SIZE = 1000


def upgrade() -> None:
    """Set both keys from ``file_path`` wherever they disagree, one id-ordered batch at a time."""
    connection = op.get_bind()
    last_id = 0
    while True:
        rows = connection.execute(
            sa.text(
                "SELECT id, file_path, file_path_key, file_path_permalink FROM entity "
                "WHERE id > :last_id ORDER BY id LIMIT :limit"
            ),
            {"last_id": last_id, "limit": REPAIR_BATCH_SIZE},
        ).all()
        if not rows:
            return
        updates = []
        for entity_id, file_path, path_key, permalink in rows:
            keys = file_path_conflict_keys(file_path)
            if (path_key, permalink) != (keys.path_key, keys.permalink):
                updates.append(
                    {"id": entity_id, "path_key": keys.path_key, "permalink": keys.permalink}
                )
        if updates:
            connection.execute(
                sa.text(
                    "UPDATE entity SET file_path_key = :path_key, "
                    "file_path_permalink = :permalink WHERE id = :id"
                ),
                updates,
            )
        last_id = rows[-1][0]


def downgrade() -> None:
    """Nothing to undo: the repaired keys are the ones every writer stores."""
