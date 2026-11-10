# BM-006: explicit local Git transactions

Status: implemented and tested in isolated local API/MCP hosts, 2026-11-04.
This extends the opt-in host in [BM006_GIT_PUBLICATION.md](BM006_GIT_PUBLICATION.md).
It does not enable Git or migrate an existing installation.

## Public contract

The new `bm_transaction` MCP tool exposes `start`, `status`, `list`, `read`,
`write`, `move`, `delete`, `keep_alive`, `commit` and `rollback`. Its bundled
[manual](../src/basic_memory/man/man3/bm-transaction%283%29.md) gives the exact
signature and examples. The tool calls the typed API client; it does not access
the repository directly. Every call explicitly selects the project.

Ordinary `bm_write`/`bm_edit`/`bm_move`/`bm_delete` calls continue to make implicit
single-operation publications. They never join a transaction through hidden
process or conversation state. Use the explicit tool/API for a bundle.

The REST prefix is `/v2/projects/{project_uuid}/git/transactions`:

| Method/path | Purpose |
| --- | --- |
| `POST ""` | Start; body `{}` or the complete source-transfer controls below. |
| `GET /{id}` | Observe status without renewing the lease. |
| `GET /{id}/notes?view=base\|draft` | List paths, UUIDs and SHA-256 checksums. |
| `GET /{id}/note?identifier=...&view=base\|draft` | Read one exact path, UUID or permalink. |
| `POST /{id}/write` | `path`, `content`, `expected_revision`, optional `operation=replace\|append\|prepend`. |
| `POST /{id}/move` | `path`, `destination`, `expected_revision`. |
| `POST /{id}/delete` | `path`, `expected_revision`. |
| `POST /{id}/keep-alive` | Renew an unexpired lease. |
| `POST /{id}/commit` | `expected_revision`, optional `message`. |
| `POST /{id}/rollback` | `expected_revision`. |

Start returns `transaction_id`, `transaction_key`, project UUID, fixed
`base_commit`, `revision=0`, state and `expires_at`. Subsequent requests pass
`X-BM-Transaction-Key`; the MCP tool takes it as `transaction_key`. The random
256-bit capability defines draft ownership in this local host. Only its SHA-256
hash is retained locally, and neither key nor hash enters published Git. The
caller must retain the key. Status does not return it. A wrong key or unknown ID
returns 404. Transaction responses use `Cache-Control: no-store`.

This is draft ownership within the existing local host. It does not replace the
separate authenticated gateway required to expose that host remotely.

## Fixed basis, private writes and validation

Each transaction belongs to one project. A private candidate tree holds its
starting Git snapshot plus its accepted changes. Reads resolve only in that
tree or in the immutable starting snapshot; mutable SQL, editor files and search
indexes never supply a draft's factual basis. Titles are not unique addresses.
Missing or ambiguous identifiers return 404 or 409.

Draft mutations compare `expected_revision`. Accepted replacements advance the
draft revision; a stale operation returns 409 and leaves accepted content intact.
After a lost write/append response, inspect the private content/revision before
retrying. This version check is not a V2 request fingerprint or response journal.

Replacement input is complete canonical Markdown with nonempty string
`title`, `type`, and `permalink` fields. BM's existing `EntityParser` validates
Markdown/timestamps; its existing edit helper supplies append/prepend behavior.
The complete draft must have unique permalinks and normalized Markdown paths.
Invalid requests return 422 before the private pointer is advanced. UUIDs are
allocated for new notes and preserved by edits/moves. Move preserves the
permalink as authored; change metadata explicitly when desired. Replacing the
identity at an occupied published path still requires a separate delete
publication, as enforced by the existing publisher.

This first explicit API accepts authored canonical bytes rather than running
every convenience normalization of the implicit BM write facade. It does not add
draft semantic search, automatic schema editing, directory operations, a general
snapshot API or a native `bm tool transaction` command.

Commit validates the complete proposed tree again and uses the publisher's
project-wide fixed-base CAS. A newer commit in that project returns 409 and
freezes the draft as `conflicted`, retaining its base/content and observed
`conflict_commit`. A publication in another project does not change this basis.
Successful commit publishes one complete tree; an unchanged tree produces a local
terminal receipt without an empty commit. Commit messages cannot contain the
text `Basic-Memory-` anywhere and in any case: BM reserves it for its own trailers
(transaction, merge, moved note), which would otherwise forge evidence. A commit that
would publish a note UUID another project still holds (its SQL rows or its published
head) is refused with 409; ordinary drafts get fresh UUIDs and are not affected (I1).

Git success is the canonical outcome. Commit wakes the existing durable worker
and returns without waiting for SQL content, files, graph/full text or vectors.
Ordinary V1 readers may therefore lag or mix published revisions. Use project
Git progress for actual stage completion; it does not pin later reads.

## States, expiry and cleanup

| State | Private content | Further actions |
| --- | --- | --- |
| `open` | Read/write | Renew, commit or rollback with the proper revision. |
| `frozen` | Read-only transfer source | Read/renew while live; rollback after successors release it. |
| `conflicted` | Read-only source after Git CAS conflict | Review into a new transaction; never replace its base. |
| `expired` | Retained read-only preview | Start a new transaction for reviewed transfer; no revival/commit. |
| `expired_cleaned` | Private refs removed | Minimal authenticated status remains. |
| `committed` | Published SHA in status | Repeated commit returns `already_committed` and the same SHA. |
| `rolled_back` | Private refs removed | No publication or history rewind. |

Defaults are `git_transaction_idle_seconds=1800` and
`git_transaction_retention_seconds=604800`, both configurable. There is no hard
maximum total lifetime. Successful private reads/writes and keep-alive renew a
live lease; passive status does not. Expired previews do not renew or revive it.
`retained_until` is the ordinary retention boundary; a retained successor may
protect the source beyond that boundary while `draft_available` stays true.

Expiry never publishes. A short metadata/publication lock coordinates admitted
commit and cleanup: a commit already running is not discarded at a concurrent
lease boundary. Startup and transaction calls evaluate expiry and collect eligible
draft refs opportunistically. No additional daemon or periodic timer is added.
There is no Git/SQL transaction or filesystem lock between agent calls.

Cleanup retains minimal owner/project/base/revision/terminal metadata. Effective
publication evidence lives in the published commit; no-op evidence lives in its
local receipt ref. Removing private refs does not remove published history or
terminal receipts. Unreachable object reclamation remains Git's garbage
collection. Retained records are scanned by maintenance; large transaction
histories need their own operating-cost measurement before production.

## Reviewed conflict transfer

Start a successor with `source_transaction_id`, `source_key`, and
`source_revision` together. Ownership and revision are checked before freezing a
live source. A source write that wins first changes the revision and rejects a
stale transfer; a transfer that wins first rejects later source writes. No
accepted change disappears between source versions.

The successor has a new ID/key/lease and a fixed current basis C. It records its
source ID/revision and starts from **C only**. It does not copy D, replay old
append calls or rebase silently. Read the old fixed B and frozen D, compare them
with C, carry only the intended changes and reread factual dependencies through
the new transaction. Commit normally against C; another publication causes a new
conflict and preserves both drafts.

A retained successor pins its fixed source against rollback/cleanup, transitively
through a transfer chain. Expired source previews can thus survive their normal
retention deadline. A committed/rolled-back or eventually collected successor
releases its pin. The successor's own candidate is independent of later source
cleanup. A lost start response can leave an orphan successor, which expires
normally; V1 does not guarantee replay of a start operation or its returned key.

## Storage and crash boundaries

`repository/git_transaction_repository.py` stores a small validated record in
`<bare-repo>/bm-transactions/{uuid}.json`, outside every indexed root. Exact
Markdown and portable identities remain in private candidate refs. Same-directory
temporary files, fsync and atomic pointer replacement publish a draft revision.
A short local file lock serializes metadata, publication and cleanup; planning
holds no lock. The runtime's existing admission gate prevents shutdown from
releasing ownership before a filesystem mutation settles, including repeated
request cancellation.

The candidate is retained before its pointer is replaced and before publication.
Every recovery checks Git's terminal transaction evidence **before** expiry or
cleanup. Process loss after Git publication but before the record becomes
`committed` is therefore reconstructed without publishing again. Unavailable
metadata/Git I/O returns 503 with an instruction to inspect status; it does not
claim a Git rollback. Rollback after publication is refused.

The tested boundary is process loss on local Linux/SQLite, not a proven power-loss
contract for the Git object store/filesystem. Operational backup must include
private candidates, transaction records and no-op receipts; a normal branch push
does not transfer them. Independent operational-state restore and production
cutover remain separate acceptance work. Draft keys remain the caller's state.

## Evidence

The new tests cover fixed own reads, two-note publication, no-op/retry, caller
ownership, version conflicts, private CRUD/rollback, configured leases and retained
previews, cleanup, protected sources, concurrent freeze/write and
commit/cleanup, another project, later conflict, metadata failure, and cancellation.
Normal API/MCP dependencies are used in integration tests.

`test-int/fixtures/git_transaction_crash.py` ends an entire normal API interpreter
before/after draft pointer save, before/after Git publication and after terminal
metadata save. Restart either keeps the prior/private draft or resolves the
original published SHA; the actual owned worker catches up afterward. The dying
child pauses background lanes to target the foreground boundary. Parent tests
resume real graph/full text/vectors with deterministic test embeddings. This is
not a TCP, GPU/model-inference or power-loss benchmark.
