"""One-off backfill of `created`/`modified` into the published notes of Git projects (BM-010).

Run with the service stopped (the script takes the same project ownership as the service,
so a running service makes it refuse). Without ``--apply`` it only reports.

    git_backfill_times.py [--tsv FILE ...] [--project NAME ...] [--apply]

Source of the times, per note UUID: first the given TSV files (columns ``external_id``,
``created_at_db``, ``updated_at_db``; naive values are local time), else the indexed entity row
in the active profile's database. A note that already has both keys is left alone.

For every changed note the script proves that nothing but the two keys differs from the
published bytes. With ``--apply`` it publishes one commit per project; the next service start
projects it (the SQL times stay the same instants, now taken from the frontmatter).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import difflib
from datetime import datetime
from pathlib import Path
import sys
from uuid import UUID, uuid4

from sqlalchemy import select

MESSAGE = "Add created and modified times to note frontmatter (BM-010 backfill)"


def read_tsv_times(paths: list[Path]) -> dict[str, tuple[datetime, datetime]]:
    """Later files override earlier ones; naive timestamps are local wall-clock time."""
    out: dict[str, tuple[datetime, datetime]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                created = datetime.fromisoformat(row["created_at_db"]).astimezone()
                updated = datetime.fromisoformat(row["updated_at_db"]).astimezone()
                out[row["external_id"]] = (created, updated)
    return out


def backfilled(markdown: str, created: datetime, updated: datetime) -> str | None:
    """The note with both keys set, or None if it has them already.

    Raises if anything but the two keys would change.
    """
    from basic_memory.file_utils import parse_frontmatter
    from basic_memory.services.note_times import (
        CREATED_FIELD,
        MODIFIED_FIELD,
        format_note_time,
        stamp_markdown_times,
    )

    meta = parse_frontmatter(markdown)
    if meta.get(CREATED_FIELD) and meta.get(MODIFIED_FIELD):
        return None
    # An existing key is never overwritten; only the missing ones are added.
    wanted = {
        key: format_note_time(value)
        for key, value in ((CREATED_FIELD, created), (MODIFIED_FIELD, updated))
        if not meta.get(key)
    }
    result = stamp_markdown_times(markdown, wanted)
    # Proof: every original line is still there, in order, and the only new lines are the keys.
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, markdown.splitlines(), result.splitlines(), autojunk=False
    ).get_opcodes():
        added = result.splitlines()[j1:j2]
        if tag == "equal" or (
            tag == "insert" and all(line.split(":", 1)[0] in wanted for line in added)
        ):
            continue
        raise ValueError("backfill would change more than the missing time keys")
    check = parse_frontmatter(result)
    if any(str(check.get(key)) != value for key, value in wanted.items()):
        raise ValueError("backfilled times do not read back")
    return result


async def run(args: argparse.Namespace) -> int:
    from basic_memory import db
    from basic_memory.config import ConfigManager
    from basic_memory.index.git_project_ownership import own_git_project
    from basic_memory.models import Entity, Project
    from basic_memory.repository.git_project_repository import (
        GitProjectRepository,
        settle_git_publication,
    )

    config = ConfigManager().config
    tsv = read_tsv_times([Path(p) for p in args.tsv or []])
    engine, maker = await db.get_or_create_db(
        config.database_path, ensure_migrations=False, config=config
    )
    del engine
    async with maker() as session:
        projects = list((await session.scalars(select(Project).where(Project.is_active))).all())
        rows = {
            e.external_id: (e.created_at, e.updated_at)
            for e in (await session.scalars(select(Entity))).all()
        }
    failed = False
    for project in projects:
        if args.project and project.name not in args.project:
            continue
        repo_path = config.git_repositories.get(UUID(project.external_id))
        if repo_path is None:
            print(f"{project.name}: not Git-managed, skipped")
            continue
        repository = GitProjectRepository(repo_path)
        try:
            async with own_git_project(
                repository,
                project_id=project.id,
                project_external_id=project.external_id,
                project_root=Path(project.path),
                session_maker=maker,
            ):
                snapshot = await asyncio.to_thread(repository.snapshot)
                files = dict(snapshot.files)
                note_ids = dict(snapshot.manifest.note_ids)
                changed = missing = 0
                for path, content in snapshot.files.items():
                    key = str(note_ids[path])
                    times = tsv.get(key) or rows.get(key)
                    if times is None:
                        missing += 1
                        print(f"  no source for {path}")
                        continue
                    new = backfilled(content, *times)
                    if new is not None:
                        files[path] = new
                        changed += 1
                print(
                    f"{project.name}: {len(files)} notes, {changed} to change, "
                    f"{missing} without a source"
                )
                if missing:
                    raise ValueError("notes without a source; nothing published")
                if args.apply and changed:
                    prepared = await asyncio.to_thread(
                        repository.prepare,
                        transaction_id=uuid4(),
                        base_commit=snapshot.commit_id,
                        files=files,
                        note_ids=note_ids,
                        message=MESSAGE,
                    )
                    publication = await settle_git_publication(repository, prepared)
                    print(f"  published {publication.commit_id[:12]}")
        except Exception as error:  # report every project
            failed = True
            print(f"{project.name}: REFUSED {type(error).__name__}: {error}")
    return 3 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tsv", action="append", help="repeatable; later files win")
    parser.add_argument("--project", action="append", help="repeatable; default: all")
    parser.add_argument("--apply", action="store_true", help="publish (default: report only)")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
