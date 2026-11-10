# BM-006: opportunistic project Git synchronization

Status: isolated implementation, 2026-11-04. Requires the opt-in local Git host
and controlled adoption described in [BM006_GIT_PUBLICATION.md](BM006_GIT_PUBLICATION.md).
Production peer endpoints, credentials and cutover are not configured by this code.

## Configuration and entry points

Each project keeps its own bare repository and private shared peer. Configure an
already registered project UUID, independently of its local path:

```json
{
  "git_sync_remotes": {
    "11111111-1111-4111-8111-111111111111": {
      "url": "ssh://git@nuc1/private/example.git"
    }
  }
}
```

The mapping defaults to empty; not every project needs a peer. Unknown Git project
UUIDs are refused at host startup. Supported endpoints are absolute local paths,
`ssh://` and `https://`. SCP shorthand, executable remote helpers, HTTP, embedded
passwords, HTTPS user information, queries and fragments are refused. Configure
credentials outside the URL, using SSH configuration/agent or a trusted helper in
the dedicated bare repository's local Git configuration. Global Git configuration
is intentionally excluded. SSH uses batch mode; Git disables terminal prompting,
local hooks, recursive submodules and unapproved transport protocols.

The published branch is fixed at `refs/heads/main`. The current interface is:

- `POST /v2/projects/{project_uuid}/git/sync`: run one bounded attempt for this project.
- `GET /v2/projects/{project_uuid}/git/status`: observe indexes and `synchronization`.
- `bm_project(command="sync", project="...", output_format="json")`: typed MCP access.
- `bm_project(command="status", project="...")`: passive status, including sync.

This slice deliberately has no timer, background network loop, all-project batch
or additional native CLI command. A caller chooses when connectivity is useful.
Ordinary local writes and explicit transactions never wait for a peer. A sync
call waits for transfer/publication, then returns without waiting for indexing.

## Transfer and publication

1. Acquire only this project's sync admission. A concurrent sync request receives
   HTTP 409; other projects continue independently. Persist a `running` attempt
   before contacting the peer.
2. Advertise/fetch **only main**, without tags, pruning, `FETCH_HEAD`, mirror/all
   refspecs or recursive transfers. Retain the incoming tip under
   `refs/bm/incoming/<attempt_uuid>`. Fetch never changes published main.
3. Parse every newly received historical tree, not just its final tip. Require
   regular UTF-8 Markdown notes, complete unique path/UUID manifests, the registered
   project UUID, canonical title/type/permalink, unique permalinks and valid Markdown
   metadata/dates. An invalid intermediate revision cannot hide behind a valid tip.
4. Briefly enter normal mutation admission and classify the current local main
   against the fetched immutable tip. A remote descendant advances main with a
   single-ref compare-and-swap that also verifies the retained incoming pointer.
   An intervening external local publication is reported as `local_changed`.
5. If only local history is ahead, send the captured **published SHA** with an
   explicit non-force `SHA:refs/heads/main` push. Network waits hold no mutation
   gate. A local commit made during the transfer remains locally ahead for the
   next attempt. A concurrent remote advance may reject the push; refetch/retry
   classifies the resulting histories without overwriting the peer.
6. Notify the existing durable worker. Incoming commits follow its ordinary
   content, file, graph/full-text and vector path, including generation fences,
   structural graph invalidation and editor-file guards. They do not create an
   extra local commit or a new indexer.

An empty remote main can be initialized by a normal non-force push. A second
installation must clone that project's published history and register its portable
project UUID; creating a second independent initialization history is not a clone.
Local SQL IDs, root bindings and indexes remain specific to each installation.

Only published ancestry is pushed. Private transaction candidates, keys/JSON,
no-op receipts, local bindings, incoming-ref names and sync observations are not
transferred. Git objects reachable through published main are its public history
for that private peer, including earlier published note versions.

## Observations and failure contract

`synchronization` contains `configured`, current `local_commit`, the last safely
observed remote commit, its relationship to local (`unobserved`, `equal`, `ahead`,
`behind`, `diverged`), and `last_attempt`. This compares an observation with current
local state; it does not assert that the unreachable/live remote still has that SHA.
Index `caught_up` says nothing about peer synchronization.

Attempts are a discriminated union:

| Kind | Meaning |
| --- | --- |
| `running` | Attempt UUID/start time persisted; completion is not known. |
| `completed` | Outcome `equal`, `pushed`, `pulled`, `diverged`, `remote_rewritten`, `merged` or `needs_review` (the last two only with `git_sync_merge`). A release that does not know them cannot load a record that holds one: roll back only after the next sync has replaced it. |
| `failed` | Reason `transport`, `invalid_incoming`, `local_changed` or `interrupted`. |

`invalid_incoming` also covers incoming history that would give a note UUID to two
projects. A UUID another project still holds only because a cross-project move is
still travelling is let through, and the projection waits for it (see "Note UUIDs
across projects" in [BM006_GIT_PUBLICATION.md](BM006_GIT_PUBLICATION.md)); a real
duplicate is refused. The status of that attempt then carries `failure_detail`, which
names the UUID and both projects. It is held in memory only (it is gone after a
restart until the next attempt), so `bm-sync.json` keeps the format older releases read.

Every finished attempt also carries its finish time. Expected transport/validation
failures return HTTP 200 with `kind=failed`; callers must inspect the attempt. This
does not mean the peer/index is synchronized. Unconfigured/busy peers return 409,
legacy/unowned projects 404, and an unavailable observation store 503. Responses
and persisted errors omit raw Git stderr, endpoint URLs and credentials. Successful
sync responses use `Cache-Control: no-store`.

`<bare>/bm-sync.json` atomically stores observations, bound to project UUID and a
peer fingerprint. The pointer uses private permissions, fsync and same-directory
replacement; symlinks are refused. Changing the configured endpoint starts a new
observation history but does not remove incoming refs. On startup a retained
`running` marker becomes `failed/interrupted`; it is never mistaken for a completed
network operation.

Git remains authoritative if publication or push succeeded before cancellation,
interpreter death, a lost answer or an observation-write failure. Retry fetches
again and compares actual histories; it cannot publish an extra content commit.
Repeated cancellation settles a started transfer/ref mutation before releasing
ownership. Shutdown drains the project's active sync before releasing repository
ownership. Each Git transport subprocess has a 30-second timeout; this is not an
end-to-end latency or index-completion guarantee.

## Divergence and reviewed recovery

`diverged` preserves both published branches and the immutable incoming ref. It
does not merge, rebase, replay an append, pick a timestamp winner or force-push.
Old explicit drafts retain their fixed base; an accepted incoming commit makes
their stale commit fail normally. Use the [transaction source-transfer protocol](BM006_TRANSACTIONS.md)
to review such private changes on a current basis.

For published-history divergence, an operator/agent with explicit merge authority:

1. Records local main, the attempt UUID/incoming SHA and their merge base; inspects
   both branches in a separate review repository/worktree outside all indexed roots.
2. Reviews the Markdown delta, reading dependencies, deletions/moves and portable
   UUID manifest. A text merge alone is insufficient evidence of semantic agreement.
3. Produces a reviewed merge commit containing both heads as ancestors and validates
   the final tree. Publishes it to the shared peer with an ordinary non-force push
   against the reviewed peer basis; if that basis changes, reviews again.
4. Runs normal sync. It validates all newly received trees and accepts the reviewed
   descendant through the same CAS/worker path. No local head is reset.

This is a documented external review workflow, **not a new automatic merge API**.
The integration test constructs an explicitly reviewed two-parent tree and proves
subsequent normal convergence. With `git_sync_merge` enabled, BM does the same itself
(next section).

## Note-level merge (`git_sync_merge`)

Per project with a peer, `git_sync_merge[<project uuid>] = "auto"` makes a `diverged` sync
merge instead of stopping. The default is off and then everything above applies unchanged.
`remote_rewritten`, a history with more than one merge base and an invalid incoming
history are never merged.

**What is compared.** Each note by its UUID, as (path, bytes) in base, local and remote;
never text lines. The manifest is rebuilt from the decisions. Changed on one side, the same
change on both, deleted twice, added on one side, and move-on-one-side-plus-edit-on-the-other
all merge by themselves. Four cases need a person: the same note changed differently, deleted
against changed, moved to two places, and two notes on one path or permalink.

**How it is published.** The device that sees `diverged` builds one commit with two parents
(local first, remote second) and trailers `Basic-Memory-Merge`, `Basic-Memory-Merge-Base`
(and `Basic-Memory-Review` after a decision). A single `update-ref` transaction moves `main`
from the local tip to that commit and verifies the incoming ref still holds the merged remote
tip; then it is pushed without force. If the local tip moved meanwhile the attempt fails
`local_changed`; if the peer moved on, the push is refused (`transport`) and the next run
merges again. The observed remote commit only advances after a successful push.

**Conflicts.** Nothing is published. A review package is written to
`<bare repo>/bm-merge-review/<local12>-<remote12>/`: `konflikte.json`, and per conflict and
note `basis.md`, `lokal.md`, `fern.md` (plus `vorschlag.md`, a line-level reading aid with
conflict marks, never applied automatically). The project waits; other projects keep
syncing. `bm_project(command="status")` shows the open review (kinds and paths only).

**Deciding.** Put a `decision` into each conflict of `konflikte.json`:

| kind | decisions |
| --- | --- |
| `both_changed` | `{"action": "lokal"\|"fern"\|"beide"\|"bearbeitet"}` |
| `delete_vs_change` | `lokal`, `fern`, `loeschen`, `behalten`, `bearbeitet` |
| `moved_twice` | `lokal`, `fern`, `bearbeitet` (the latter also needs `"path"`) |
| `path_collision`, `permalink_collision` | `{"notes": {"<uuid>": {"path": "new.md", "bearbeitet": true}}}` |

`beide` keeps the local note and adds the remote version as a new note (`<name>-fern.md`,
title and permalink marked). `bearbeitet` takes the text from `bearbeitet.md` next to the
three versions. Then run `bm_project(command="merge", project=..., review=<id>)` on the local
service. It is not reachable through the gateway. A decision is bound to the exact versions
it was made for: if either tip moved, the package is rewritten, decisions whose versions
are unchanged are carried over, and the command must be run again.

A previously observed peer head must remain an ancestor of its new head. Rewinding,
deleting or replacing that branch reports `remote_rewritten` repeatedly; it never
silently resets the observation watermark or recreates the deleted branch. Inspect
the retained incoming ref and restore/reconcile the peer history deliberately.

## Evidence and limits

Tests use real local bare peers and normal API/MCP lifespans with isolated SQLite.
They exercise offline writes, private-ref exclusion, both transfer directions,
empty-peer initialization, invalid intermediate history, divergence, repeated
rewind/deletion refusal, independent projects, CAS races, cancellation, restart and
whole-interpreter death before/after fetch, publication, push and observation writes.
The existing worker checks real graph/full-text/vector convergence with deterministic
test embeddings. This is not an SSH/HTTPS deployment, TCP-server, model-performance,
network-filesystem or power-loss test.

Incoming refs are deliberately retained for review/crash evidence, including failed
validation; automated pruning and large-history transfer/validation measurements
remain operational work before rollout. Sync metadata and private drafts/receipts
need independent operational backup; branch replication is not a full backup. See
the [remaining production obligations](BM006_GIT_PUBLICATION.md#remaining-before-productive-use).
