"""One-off backfill of `created`/`modified` into the published notes of Git projects (BM-010).

Run with the service stopped (the script takes the same project ownership as the service,
so a running service makes it refuse). Without ``--apply`` it only reports.

    git_backfill_times.py [--tsv FILE ...] [--project NAME ...] [--apply]

Source of the times, per note UUID: first the given TSV files (columns ``external_id``,
``created_at_db``, ``updated_at_db``; naive values are local time), else the indexed entity row
in the active profile's database. A note that already has both keys is left alone.

``--edited-since TIME`` (with a UTC offset) says when the TSV values were taken. A note listed
there whose bytes changed in the published history after that moment was edited since, so its
``modified`` is the time of that change, not the older TSV value. A move (same bytes under a new
path) is not a change. Every such note is named in the report.

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
import os
from pathlib import Path
import subprocess
import sys
from uuid import UUID, uuid4

from typing import Any

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


def last_content_changes(repository: Any) -> dict[str, datetime]:
    """When the bytes of each note last changed in the published history (first parent).

    Only a note that already existed in the previous commit and has other bytes now counts.
    A note that first appears (the adoption, or a note created later) and a note moved to
    another path with the same bytes are not changes.
    """
    head = repository.snapshot().commit_id
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    listing = subprocess.run(
        ["git", "--git-dir", str(repository.path), "rev-list", "--first-parent", "--reverse", head],
        capture_output=True,
        check=True,
        env=environment,
        timeout=60,
    ).stdout.decode("ascii")
    changes: dict[str, datetime] = {}
    previous: dict[str, str] | None = None
    for commit in listing.split():
        snapshot = repository.read_snapshot(commit)
        committed = repository.committed_at(commit)
        current = {
            str(identity): snapshot.files[path]
            for path, identity in snapshot.manifest.note_ids.items()
            if path in snapshot.files
        }
        if previous is not None:
            for identity, content in current.items():
                if identity in previous and previous[identity] != content:
                    changes[identity] = committed
        previous = current
    return changes


def resolve_times(
    key: str,
    *,
    listed: dict[str, tuple[datetime, datetime]],
    rows: dict[str, tuple[datetime, datetime]],
    changes: dict[str, datetime],
    since: datetime | None,
) -> tuple[datetime, datetime, bool] | None:
    """(created, modified, edited_since_snapshot) for one note, or None without any source."""
    base = listed.get(key) or rows.get(key)
    if base is None:
        return None
    created, modified = base
    changed = changes.get(key)
    if key in listed and since is not None and changed is not None and changed > since:
        return created, changed.astimezone(), True
    return created, modified, False


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
    since = datetime.fromisoformat(args.edited_since) if args.edited_since else None
    if since is not None and since.tzinfo is None:
        raise SystemExit("--edited-since needs a UTC offset, e.g. 2026-11-08T06:44:29+01:00")
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
                changes = (
                    await asyncio.to_thread(last_content_changes, repository)
                    if since is not None
                    else {}
                )
                changed = missing = edited = 0
                for path, content in snapshot.files.items():
                    key = str(note_ids[path])
                    times = resolve_times(key, listed=tsv, rows=rows, changes=changes, since=since)
                    if times is None:
                        missing += 1
                        print(f"  no source for {path}")
                        continue
                    created_at, modified_at, edited_since = times
                    if edited_since:
                        edited += 1
                        print(
                            f"  edited since the snapshot: {path} -> modified "
                            f"{modified_at.isoformat(timespec='seconds')} "
                            f"(TSV had {tsv[key][1].isoformat(timespec='seconds')})"
                        )
                    new = backfilled(content, created_at, modified_at)
                    if new is not None:
                        files[path] = new
                        changed += 1
                print(
                    f"{project.name}: {len(files)} notes, {changed} to change, "
                    f"{missing} without a source, {edited} edited since the snapshot"
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
    parser.add_argument(
        "--edited-since",
        metavar="TIME",
        help="when the TSV values were taken (ISO 8601 with offset); later edits keep their time",
    )
    parser.add_argument("--apply", action="store_true", help="publish (default: report only)")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
