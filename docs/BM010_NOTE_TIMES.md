# BM-010: creation and modification times live in the note

Status: built and verified in isolation, 2026-11-08. Active only on a Git host.

In a Git-managed project the Markdown bytes are the only durable state; SQL is rebuilt from
them (restore, a new host, a peer's history). Without `created`/`modified` in the frontmatter,
every rebuild dates each note with the projection time. The Git host therefore keeps both keys
in the accepted bytes (ISO 8601 with offset, to the second):

| Event | `created` | `modified` |
|---|---|---|
| new note | now (or the writer's value) | now (or the writer's value) |
| write/edit that changes content or other frontmatter | unchanged | now |
| write that changes nothing | unchanged | unchanged |
| write that changes only `created`/`modified` | as written | as written (never renewed) |
| move (in a project, across projects, in a transaction) | unchanged | unchanged |
| note without the keys | the entity row's value | now on a change; the row's value on a move |

The two lines are inserted as text, so no other byte of the note changes.

## One-off backfill

`scripts/git_backfill_times.py` (service stopped) adds the missing keys to the published notes
from TSV files (`external_id`, `created_at_db`, `updated_at_db`; naive values are local time)
and, for notes not listed, from the entity row. Without `--apply` it only reports. For every
note it proves that nothing but the two lines was added; an existing value is never
overwritten; one commit per project.

## Evidence

* `tests/services/test_note_times.py` (rules, text insertion, backfill proof).
* `test-int/test_git_note_times_integration.py` (real Git host: write/edit/move, move across
  projects, transactions, date-only writes, no-op writes).
* Rehearsal on a copy of the real data: backfill (626 notes, 1,202 added lines, 0 changed),
  service start, SQL times unchanged to the second; then `bm git backup` and `bm git restore`
  into a fresh instance: SQL `created_at`/`updated_at` equal the frontmatter for 626 of 626
  notes.
