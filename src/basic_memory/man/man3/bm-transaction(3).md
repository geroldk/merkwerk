---
title: bm-transaction(3)
type: manpage
section: 3
name: bm-transaction
summary: prepare and publish private local Git transactions
generated: registry
tool: bm_transaction
verified: BM-006 local Git host
---

# bm-transaction(3)

## NAME

**bm-transaction** — prepare and publish private local Git transactions

## SYNOPSIS

```
bm_transaction(command, transaction_id=None, transaction_key=None,
               expected_revision=None, identifier=None, content=None,
               destination=None, operation="replace", view="draft",
               message="Publish Basic Memory transaction",
               source_transaction_id=None, source_key=None,
               source_revision=None, output_format="json", project=None)
```

## DESCRIPTION

Requires an explicitly configured local Git host and `project` on every call.
Ordinary bm_write/bm_edit/bm_move/bm_delete calls still publish implicitly;
they do not join a transaction automatically.

`start` returns `transaction_id`, `transaction_key`, a fixed `base_commit`,
`revision=0`, and a lease. Retain the private key: it is required for all subsequent
operations and is not returned by status or stored in published Git. No global
current transaction is used. Two clients can keep independent drafts.

`list` shows paths, UUIDs and checksums. `read` resolves an exact path, UUID or
permalink exclusively inside the fixed tree. `view="base"` reads the starting
tree; `view="draft"` includes accepted private changes. Titles are not addresses.
Drafts are absent from ordinary readers, graph/full text and vectors.

`write`, `move` and `delete` require `expected_revision` from a previous result.
Writes replace complete canonical Markdown by default; `append`/`prepend` use
BM's existing text-edit rules. A replacement needs nonempty `title`, `type` and
`permalink` frontmatter. Duplicate permalinks and invalid paths are refused.
Move preserves UUID and canonical permalink; change metadata explicitly when
desired. A stale draft revision is a conflict: inspect the private result before
retrying a lost response. V1 offers no general operation replay key.

`commit` validates the full tree and compares the entire project's fixed base.
It publishes one Git commit, or no empty commit when unchanged. Index work is
asynchronous. Ordinary readers may lag or mix published revisions. A repeated
commit returns `already_committed` with its original SHA, even after later commits.
`rollback` discards only private changes; it cannot rewind a published commit.

Default inactivity is 30 minutes. Successful private reads/changes and
`keep_alive` renew a live lease; `status` only observes and never renews it.
Expired drafts cannot be committed or revived. They remain available for seven
days for explicit transfer into a new transaction. Expiry never commits.
Cleanup runs on startup and transaction calls, rather than on a new daemon.
Minimal authenticated terminal status survives draft cleanup.

After a base conflict, the old draft is read-only. Start a successor with
`source_transaction_id`, `source_key` and `source_revision` to freeze and protect
that exact source. The successor starts from current Git C without copying B→D.
Read source base/draft and the successor's base; carry intended edits explicitly,
recheck factual dependencies and commit normally. Active/retained successors pin
their source against cleanup. Never silently rebase or replay append operations.

## PARAMETERS

- **command** (required) — Transaction operation.
- **transaction_id** (string | null, optional, default: None) — ID returned by start.
- **transaction_key** (string | null, optional, default: None) — Private key returned only by start.
- **expected_revision** (integer | null, optional, default: None) — Last read draft revision; required for changes/commit/rollback.
- **identifier** (string | null, optional, default: None) — Read: exact path, UUID or permalink. Changes: exact Markdown path.
- **content** (string | null, optional, default: None) — For write: full canonical Markdown, or appended/prepended text.
- **destination** (string | null, optional, default: None) — For move: new exact Markdown path; UUID/permalink are preserved.
- **operation** (string, optional, default: "replace") — Write behavior.
- **view** (string, optional, default: "draft") — For list/read: fixed basis or own draft.
- **message** (string, optional, default: "Publish Basic Memory transaction") — For commit: Git message. It must not contain 'Basic-Memory-' in any case: BM reserves that prefix for its own commit trailers.
- **source_transaction_id** (string | null, optional, default: None) — For start: freeze/protect this old source; do not copy it automatically.
- **source_key** (string | null, optional, default: None) — For start with a source: its private key.
- **source_revision** (integer | null, optional, default: None) — For start with a source: its reviewed revision.
- **output_format** (optional, default: "json")
- **project** (string | null, optional, default: None) — Required local Git project name or UUID.

## EXAMPLES

```
bm_transaction(command="start", project="main")
# Keep the returned transaction_id, transaction_key and revision.

bm_transaction(command="write", project="main", transaction_id=tid,
               transaction_key=key, expected_revision=0, identifier="plans/plan.md",
               content="---\ntitle: Plan\ntype: note\npermalink: main/plans/plan\n---\n\nDraft")
bm_transaction(command="read", project="main", transaction_id=tid,
               transaction_key=key, identifier="plans/plan.md")
bm_transaction(command="commit", project="main", transaction_id=tid,
               transaction_key=key, expected_revision=1, message="Publish reviewed plan")
```

## SEE ALSO

- see_also [[bm-project(3)]]
- see_also [[bm-write(3)]]
