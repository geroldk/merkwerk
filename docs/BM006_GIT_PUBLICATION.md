# BM-006: Git publication, recovery and opt-in local hosting

Status: ten implementation slices, 2026-11-04. Experimental code on
`feat/bm006-git-publication`, based on `local/bm003-baseline`. API/MCP hosting can now
select Git through explicit configuration after controlled adoption. The default
configuration stays on the existing write path. No productive installation or
existing project has been switched by this implementation.

## Binding scope

Each BM project has its own private note repository. The public implementation
repository is separate. Git defines published note content; local SQLite content
reads, graph, full text and vectors may lag or mix **published** revisions in V1.
Private drafts must never leak through these readers. Explicit consistent snapshot
read APIs and general per-operation idempotency/replay are V2 work.

Effective implicit writes produce one commit; unchanged bytes produce no empty
commit. Explicit multi-note transactions have a fixed project base, keep private
drafts without holding Git/SQL locks during agent work, and publish one
commit. V1 rejects a changed base anywhere in that project. Commits in another
project do not invalidate it. Offline divergence requires explicit review; no force
push, timestamp winner or silent rebase is promised.

## Implemented primitive

`repository/git_project_repository.py` owns a **dedicated bare repository outside every
indexed note directory**. The normal implementation checkout, editor working files,
global Git configuration and user index do not participate. Creation refuses an
existing directory. An initial commit stores the project UUID and establishes the basis.

1. `prepare(transaction_id, base_commit, files, note_ids)` builds a full Markdown tree using
   a temporary Git index and immutable blobs. It validates relative note paths and
   retains the candidate under `refs/bm/candidates/<uuid>/<sha>`. Preparation does
   not change the published branch or hold a planning-time lock.
2. Effective candidates contain a `Basic-Memory-Transaction: <uuid>` commit trailer.
   `publish()` compares the fixed base and advances `refs/heads/main`. The terminal
   evidence is **in that same commit**, not dependent on a second ref update.
3. `transaction_status()` finds the transaction in published history. Retrying a
   terminal publication returns its original SHA even after subsequent commits;
   it never resets the branch or publishes the candidate a second time.
4. A no-op records a local terminal receipt pointing to its existing base. One ref
   transaction locks/verifies the branch but modifies only the receipt. Effective
   publications also verify that this no-op receipt is absent. Thus a no-op and an
   effective candidate cannot both close the same transaction.
5. `snapshot()` reads exact UTF-8 Markdown bytes from a published commit. Candidate
   commits outside published ancestry and non-regular files are rejected. This is
   an internal primitive, not a public V2 consistency API.

## Portable identities and SQL projection

Every tree contains `.basic-memory/project.json`, validated by `schemas/git_project.py`:

```json
{
  "format_version": 1,
  "project_external_id": "11111111-1111-4111-8111-111111111111",
  "notes": [
    {"external_id": "22222222-2222-4222-8222-222222222222", "path": "notes/example.md"}
  ]
}
```

The manifest and Markdown publish in the same tree. UUIDs are explicit: missing or
duplicate identities/paths, unidentified files, reserved control paths and unknown
manifest versions fail validation. An existing path cannot silently receive a new
UUID. An explicit permutation of existing UUIDs supports path swaps; inventing a
new UUID at an occupied path or replacing a disappearing UUID still requires a
separate delete publication. Canonical JSON ordering prevents a reordered identity list creating an empty
content change. The manifest contains no host paths, numeric SQL IDs or credentials.
The old first-slice repositories without a manifest are disposable experimental
stores; there is no implicit migration/fallback for that format.

`repository/git_projection_repository.py` explicitly registers a clone's project UUID
with a local root/name. Existing local names/roots or UUIDs bound elsewhere are not
silently adopted. `services/git_project_projection.py` restores exact `NoteContent`
under the manifest's note UUIDs, with short per-note SQL transactions and read-cache
invalidation. Repetition keeps the UUID and content version; an unchanged note keeps
its file state, since the files stage checks every note's file on disk itself and
rewrites a missing or interrupted one. The normal BM guarded materializer then writes
bytes; it preserves an unexpected editor file and reports conflict.

Existing SQL revisions use version CAS. An earlier absence uses a unique-key INSERT,
so a concurrently appearing content row cannot be overwritten by an initial-version
upsert. Existing UUID/path ownership and mirrored content identities are checked.
SQL bytes outside the target's published history are refused; private candidate
objects do not count as published history. Adoption now proves the UUID, old path
and exact bytes belonged together in a published ancestor. The check traverses
the selected published history, not every object in the repository.
See [git-rev-list](https://git-scm.com/docs/git-rev-list).

The projector accepts only the current published head and checks it again before
each per-note write. If Git advances it stops as superseded and the worker retries on
the latest head at once; it does not finish an older target over newer SQL. A partial pass can leave a
mixture of published revisions, which V1 permits. It does not acknowledge complete
project/index progress. Proven moves/deletions are reconciled as described below.
Unproven local notes and raw imports lacking canonical BM `title`/`type`/`permalink`
still require controlled preparation and are refused.

## Note UUIDs across projects (I1)

A note keeps its UUID when it moves to another project on a Git host (decisions I1 A
and I2 A). The move (W2, built in part B) drafts the note in the target with the old
UUID, deletes it in the source, then commits the target. Until both projects have
caught up, on this host or on a peer that syncs the two projects in any order, one UUID
can stand in two projects' published states. `repository/git_host_identities.py` sorts
every such UUID with one rule, for the projection and for the sync guard alike:

| Case | The owner (SQL holds the UUID there) | Result |
| --- | --- | --- |
| (a) | its published head no longer holds the note: only its SQL lags | wait |
| (b) | its head still holds it, and the commit that introduced the note into the claiming history carries a move trailer for it | wait: the move is still travelling |
| (c) | anything else | real duplicate, refused as before |

The proof is the trailer `Basic-Memory-Moved-Note: <note uuid> from <source project uuid>`,
which the target commit of a move carries. It proves a **move**, nothing else; this
narrows I2 A ("wait while the source has not caught up") to moves BM itself recorded.
It is read as a real Git trailer of BM's final trailer block, and only on the commit
that last introduced the note into the claiming project's current history (first
parents; a merge that keeps a note its first parent had dropped introduces it itself),
so an old trailer from an earlier round trip proves nothing. Callers cannot forge it:
the text `Basic-Memory-` in any case is refused anywhere in a commit message, both by
the transaction API and by `prepare()`, and BM's own messages quote folder names with
a non-breaking hyphen. Any well-formed trailer for the note counts, so a chain A -> B -> C
converges; but when the owner itself took the note by a move from the same source the
claim names, and no project on this host shows a move chain from the owner to that
source, two hosts moved one note to two projects (a double move): that is case (c).
The double move is also caught while the common source still owns the note (both
targets synced before it): another project's published head that already took the note
by a move from a source the claim names too, and lies on no chain to the claim, makes the
claim a duplicate whatever SQL says, so the second target is refused by name instead of
reaching SQL's unique index (sync orders b, c, a and c, b, a are tested).
Only manifests and trailers are read: one first-parent walk of the claiming history
for all notes at once (manifests in `cat-file --batch` chunks, stopping when every note
is placed) and one `git log` for their trailers; both are cached per tip and commit.
`test-int/test_git_note_ownership_scale.py` (slow) classifies 600 notes that arrived by
600 single moves into a 1000-note project in about 0.8 s (a cached pass: 0.0 s); a walk
per note took 86 s. The sync guard runs inside the project's admission.

**Projection.** With the host's identities, `project_git_content` projects everything
except the waiting notes: removals and moves of local notes run as usual, which also
releases the UUIDs this project gives away and breaks crossing moves (P6). It then
raises `GitProjectionDeferred`, a subclass of `GitProjectionConflict`, so a caller
without identities (offline tools) refuses as before. The structural graph check
compares SQL with the projected part only: repeated waiting passes do not reset the
graph and vector progress of every project. The worker counts a deferral as no
failure, polls at six poll intervals (at most 60 s, never below one interval) and is
woken when an owner's content stage completes (also its startup recovery), and once
after every runtime has started. Host startup tolerates a deferral; a **real
duplicate at startup still refuses the whole host start**, loudly, naming the UUID and
the owner (decided: no per-project blocked state).

**Writes.** A project that waits refuses writes with HTTP 409 `kind=deferred`, naming the
waiting notes, the owners and an owner's own content error (MCP: `UNAVAILABLE`); this is
not the 503 "Git published; SQL pending". A transaction commit that would publish a
UUID another project still holds is refused (409), and so is a PUT upsert that would
create one; a UUID an open cross-project move reserves is refused host-wide except for
that move (the reservations are filled by the move service). While any intent file in a
`bm-moves/` cannot be read, its move's UUID is unknown: every caller-chosen UUID (a PUT
upsert create, a new cross-project move) is refused host-wide until the file is fixed or
removed and the moves are read again (`GET /v2/git/moves`, resume, next start); new notes
with fresh UUIDs are not affected.

**Sync.** `require_note_ownership` applies the same rule to a fast-forward tip and to a
prepared merge: a waiting UUID may be published, a real duplicate stays
`invalid_incoming`, and the status of that attempt names the UUID and both projects
(`failure_detail`, held in memory; `bm-sync.json` keeps its format).

Known limits:
- Writes into a waiting project are refused until the owner releases (one sync of the
  owner on a peer); transactions there can still commit, though they do no SQL catch-up.
- The files, graph and vector stages of a waiting project do not run: files of notes it
  removed stay on disk and search rows of notes it gave away remain until it resumes.
- A double move on two hosts stays a named `invalid_incoming` with no recovery tool; a
  person removes one copy.
- A merge decision "keep" (`behalten`) for a note this host moved away (the peer edited
  it in the source) would put its UUID back into the source: the sync guard refuses it
  by name; "delete" converges. There is no re-key tool.
- A claim through a project the owner itself took the note from earlier (a round trip
  through a hub) is refused until that hub or the owner has synced; then it converges.
- A move trailer forged in a peer's history only makes a duplicate wait instead of fail;
  a UUID owned elsewhere is never inserted.
- Deployment order: both hosts need this rule before the first such move. An older
  peer refuses the target's incoming history as `invalid_incoming` until it is
  upgraded, and recovers then; it does not stop that host.

### Moving a note into another project (I1 part B)

`services/git_cross_project_move.py` moves one note and keeps its UUID. The route is
`POST /v2/projects/{source}/git/moves` with `{external_id, target_project_id,
destination_path, expected_checksum?, catch_up?}`; `bm_move(destination_project=...)`
calls it on a Git host (a folder: once per note, then one
`POST /v2/projects/{target}/git/moves/catch-up`). On a host without Git publication the
routes answer 404 `kind=not_git_host`, and the MCP tool keeps the copy-with-a-new-id move
there (W0).

The steps, under the host's `move_lock` (one move, resume or resolve at a time) and with
the **target's admission gate held from the draft to the commit** (lock order target, then
source):

1. **Checks, nothing written.** The target's SQL follows its head before each note (a
   target that waits for other notes still takes this one; the titles of its waiting
   notes are read from its head). The source note must be in SQL and in the source
   head with the same bytes; `expected_checksum`, a Markdown path, no `locked: true`, no
   open move holding the UUID, no note with the same title in the target (P2, ignoring
   case), a free path (also equivalent spellings, and on disk), the UUID not in the
   target, and no other project holding it in SQL or in its published head (a peer's
   move of it into a third project that waits here: 409 `owned_elsewhere`, since the
   target commit could never take it after the source delete). The canonical bytes come from the target's own create preparer in a rolled-back
   session: the permalink is formed from the new path (a taken one gets `-1`), and a note
   without `created`/`modified` first gets the source entity's times, never the move time.
2. **Draft (0).** A target transaction holds the note under its UUID
   (`write_identified_admitted`, never routed), the UUID is reserved host-wide, and the
   intent is saved: `<target bare repo>/bm-moves/<move_id>.json` (directory 0700, file 0600,
   temporary file, fsync, rename, fsync of the directory; symlinks refused). It holds the
   target bytes and the draft's key; `GET /v2/git/moves` never shows either.
3. **Source delete (b)** through the ordinary guarded delete (`NoteContentMutationService`
   of the request, with the checksum read in 1; the journal records it like any delete, and a
   deployment's actor resolver sees the label `api-cross-project-move`). The source's gate
   is not held between the checks and the delete, so another client may delete the note
   (or edit and then delete it) meanwhile; only this move's own removal counts. Success:
   the newest removal after the base is the move's (the reservation keeps the UUID from
   being created anew). "Git published; SQL pending": the removal whose commit the answer
   names. A definite refusal (4xx, e.g. the note is gone or changed): nothing was moved,
   whatever the history shows, so a delete by someone else is never undone in the
   target. A lost answer: the first removal after the base (`removals_since`) counts only
   if its parent held the note with exactly the bytes read. Otherwise the draft and the
   intent go and the source's refusal is answered ("Nothing was moved"; a waiting source
   answers 409 `kind=deferred`). After "Git published; SQL pending" the source is caught
   up under its own gate before the commit. The intent then says `source_deleted` with
   the removing commit. Known limit: a lost answer at the same moment as someone else's
   delete of the unchanged note cannot be told apart from the move's own delete; the
   move then completes.
4. **Target commit (c)** with the trailer `Basic-Memory-Moved-Note: <uuid> from <source>`.
   A draft that can no longer commit (the target advanced, the draft expired after 1800 s
   or was cleaned) is rebuilt from the intent: new canonical bytes through the preparer,
   the new transaction saved into the intent before the old draft is dropped. If the
   source still holds the UUID, it is caught up once more; otherwise the move is `pending`.
5. **After the commit** the intent and the reservation go, and the target's SQL catches up
   before the answer, so `bm_read` by UUID works at once. A failure there is no failed
   move: the answer is `completed` with `index_pending` and both commit ids.

The answer is `completed`, `pending` (the source let go, the target could not commit yet;
resume retries) or `stuck`. `GET /v2/git/moves` names the cause in `stuck_reason`:

| Stuck because | Choices that work |
| --- | --- |
| the planned path (or a rebuilt draft) is refused by the target | `target` with a free `destination_path`, or `source` |
| the UUID came back into the source after the delete | `source` only (`target` answers 409 `in_source`) |
| the source project is not on this host any more | `target` only (`source` answers 409 `no_source`) |
| the source's history no longer contains the commit the move read | `target` only (`source` answers 409 `no_source_bytes`) |

With the gate held, `stuck` cannot arise from local writes; it needs a crash. Nothing is lost then: the bytes are in the intent and in the source's
history, the UUID stays reserved (a PUT that would create it anywhere is refused 409
`owned_elsewhere`), and the MCP tool answers `PARTIAL_FAILURE` with `details.move_id`,
`state`, `external_id` and `target_path`.

**Resume.** At every host start, after all runtimes entered and before the first request,
the open intents reserve their UUIDs and each one is decided (oldest first, each in its own
error boundary, so one broken intent never stops the host):

| The target... / the source's history after the base | Outcome |
| --- | --- |
| target published the draft (record or trailer), or its head holds the UUID | completed: the intent goes |
| no commit dropped the UUID, and the source head holds it | abandoned: the draft goes, the note stays in the source |
| intent `drafted`, and the first removal did not remove the bytes read (an edit came first) | abandoned: someone else removed it; the move never deleted |
| a commit dropped it, and the source head holds it again | stuck (a peer or a write brought it back); never dropped as "never moved" |
| a commit dropped it | commit as in step 4, rebuilding the draft if needed |

`POST /v2/git/moves/resume` does the same online. `POST /v2/git/moves/{move_id}/resolve`
takes a person's choice for a stuck move: `{"choice": "target", "destination_path": ...}`
publishes it in the target at another path, `{"choice": "source"}` puts it back into the
source with its UUID and the bytes it had there (at another `destination_path` its
permalink is formed again there, as for any new note; or it closes the move when the
source holds the UUID again; the moved bytes stay in the source's history). Each
intent file is read on its own: a damaged or stray file is listed under `unreadable` by
project and file name with a fixed reason (no host path, no raw error), and the readable
intents next to it are still reserved and resumed. There is no CLI command;
the offline `bm git remove-project` refuses a project an open move names, and
`bm git backup` / `restore` carry `bm-moves/` (with its lock taken during the backup), so
the restored host's rebuild resumes the move.

The crash matrix (`test-int/test_git_cross_project_move_w2_integration.py`, process exits
through `test-int/fixtures/git_cross_move_crash.py`): after the draft before the intent and
after the intent before the delete the note stays in the source (an orphan draft expires);
after the source's Git publication before its SQL, after the delete before the commit,
after the target's publication before its record, and after the record before the intent
goes, the next start finishes the move. An expired or cleaned draft is rebuilt; a taken
permalink gets `-1`; a taken path makes the move stuck, visible and resolvable, and the
host still starts.

Further limits of part B:
- Moves are one at a time per host; while one runs, every write into its target waits.
- A move does a full target transaction per note (start, write, commit, each reading and
  writing the whole tree) plus the source delete. Measured with
  `test-int/test_git_cross_project_move_scale.py` (slow and benchmark; one machine, both
  projects settled first, a target of 1000 notes): a folder of 300 notes took 769 s for the
  `bm_move` call (2.6 s per note), and every stage (files, graph, vectors) had caught up
  323 s later; 600 notes took 1840 s (3.1 s per note), then 433 s. The cost per note
  grows with the target and with the background graph work that every source delete
  restarts host-wide. Two changes made this feasible: snapshots and tree writes in one Git
  call each (before: 3.5 s per snapshot and 3.8 s per tree write of 1000 notes, several of
  each per move), and content passes and draft checks that skip unchanged notes (before:
  a median of 9.7 s per move, measured over 15 moves). A call of this
  length outlives an ordinary MCP or gateway timeout; a caller that loses the answer
  finds the moved notes in the target (each has its id) and the rest still in the source.
- Every move leaves a committed transaction record next to the target repository; records
  are never pruned, and each transaction call reads all of them (measured: 0.026 s per call
  at 600 records, so a minor share of a large folder move).
- The P2 title rule is not applied again when resume rebuilds a draft.
- A merge "keep" for a note this host moved away re-creates the UUID in the source: the
  ownership guard refuses it by name (no silent duplicate); there is no re-key tool.

## Guarded moves, deletions and old-file cleanup

`repository/git_layout_repository.py` records all local note identities/content in
an immutable basis. Target notes are parsed/validated before changing that layout.
A short SQL transaction compares the basis again, claims moved/deleted NoteContent
using version/checksum CAS, and records durable old-file retirement permissions.
No Git or file I/O runs inside that transaction. Moved notes keep their UUID and
numeric ID; advancing the content version fences older materialization requests.
Their destination file lineage is cleared so it cannot authorize overwriting an
unrelated destination. SQL-only temporary names permit path/permalink permutations;
none are committed or materialized. Deleted targets leave incoming relations unresolved.

`models/git_file_retirement.py` and migration `0045` store the old path, permitted
checksums and retirement attempt UUID independently of the Entity's lifecycle.
Permissions include proven accepted bytes and known materialized lineage. Repeated
moves merge older permissions; a stale acknowledgement cannot erase a newer one.
An inconsistent accepted checksum grants no cleanup permission.

The files stage writes each note's new file before its old file goes (F2), so a
moved note is never missing from disk; for the length of one note it can be there
twice. `index/git_file_projection_plan.py` classifies every path from one look at
SQL, disk and the retirement rows and orders the steps; the stage in
`index/git_projection.py` carries them out with BM's checksum-guarded storage
deletion. Old bytes go only after the note they belong to is verified on disk (the
owner is named from the bytes, or from the last fully written commit where disk still
holds its bytes); an unknown owner waits until every other note is on disk. A path
another note still occupies is cleared right before its own write, one path at a
time; a swap is broken at one delete, so one note is absent for two deletes and two
materializations. A missing file is already retired; a reused path containing its
exact current Git bytes is kept and its row still published as synced. Unexpected
editor bytes keep the retirement and files progress pending, while the other notes
are still written. Bytes nothing in SQL explains are replaced only when Git published
them at that path. A case- or Unicode-only alias of a live file (the same directory
entry, never a symlink) is renamed, never deleted, and waits like any other old file
for the note its bytes belong to; manifests whose paths would merge on such file
systems are refused. Files and folders may trade places: folders the stage's own
deletions left empty are removed (never the root, a hidden folder or one reached
through a symlink), also on the pass after a crash, and an empty folder where a note
must go is removed before the write. A folder the system refuses to remove simply
stays. An unknown owner's old file is not deleted once any write of the pass has
failed; in a swap whose first write fails, that note stays absent until the next
pass. A final check reads every note.
The graph stage purges orphan full-text rows; the vector stage removes stale
manifest/physical vectors through the normal repository cleanup/reconciliation.
These guards do not replace the preproduction requirement to control all writers.

`test-int/test_git_layout_integration.py` covers moves, swaps, deletions through an
empty project, path reuse, coalesced moves and rapid round trips without intervening
file work. It verifies preserved identities, incoming unresolved relations, FTS and
physical vector cleanup. Child processes terminate after layout/file work,
immediately after file deletion but before retirement acknowledgement, at the first
old file to go (right after its note's new file is written), between a reused path's delete
and its write, and after a write before its publish. Restart repeats safely,
including when an old file has disappeared or a new file now owns its path. Editor
conflicts remain visible and preserve the editor's bytes.
`test-int/test_git_file_projection_integration.py` watches disk during the files
stage (no moment without a note, at most one duplicate at a time, a 110-note folder
move without empty folders left) and covers commits arriving during a pass.

The one-ref rule matters: Git documents atomic updates of individual refs while
concurrent readers can observe a subset of a multi-ref update. This implementation
does not depend on crash atomicity of multiple modified refs.
See [git-update-ref](https://git-scm.com/docs/git-update-ref).

The UUID here identifies a terminal publication. It is **not** an idempotency key
for an implicit API call, append operation, or draft update. General operation
deduplication remains V2. No-op receipts and retained private candidates are local
operational state; an ordinary branch push does not transfer them. Their private-state backup/restore contract is implemented in
[BM006_BACKUP_RESTORE.md](BM006_BACKUP_RESTORE.md); productive storage and cutover
still require operational approval.

## Real BM integration experiment

`test-int/test_git_publication_integration.py` installs the publisher **only in its
test fixture**, using `NoteContentMutationService.on_accepted_mutation`. Real API
create/edit calls still run the existing preparer and accepted-note runner.

The hook receives canonical Markdown while the SQLite transaction is uncommitted.
A second SQL connection cannot read those new bytes. The hook publishes Git first;
SQLite then commits and the existing post-commit/materialization code proceeds.
Git failure rejects and rolls back the prepared SQLite mutation.

The experiment pauses the existing production materializer, not a replacement
index callback. The write and normal content read complete while files/vectors
are still outstanding. Releasing that worker produces identical file bytes,
observation/relation rows, full-text results, and vector manifests/physical vectors
for entity, observation and relation chunks. Embeddings use the repository's normal
deterministic test provider; this is a storage/pipeline proof, not a model benchmark.

An interruption after Git publication intentionally rolls back SQLite. The test
reopens Git, resolves the terminal transaction without a second publication, restores
the original UUID/content through the projector, and calls the existing recovery
materializer. A **full normal project index API** run rebuilds graph/full text and
vectors. The full index is explicit: file recovery alone can leave matching file
checksums while graph/search have never been built.

An independent `git clone --bare --no-local` is then registered in a second empty
SQLite database. Occupied numeric IDs force different local project/entity numbers.
The normal API still reads the original UUIDs and exact contents; relations resolve
to the new numeric target IDs, and physical vectors match the expected chunks. This
proves portable identity/content recovery in the isolated SQLite experiment. The
clone keeps the original canonical project naming/settings. Normal opt-in lifecycle
wiring is tested below; configuration variation and productive migration need
their own controlled validation.

Additional cases exercise successful API edits and rejection/SQL rollback when a
different commit advances the same project's Git basis. Core tests use actual Git
repositories, concurrent publishers, and child processes that terminate with
`os._exit()` immediately before/after completed publication calls. These process
tests do not kill the whole ASGI process mid-SQL commit, inject failures inside
Git's ref implementation, or prove power-loss/filesystem durability.

## Durable catch-up worker

`index/git_projection.py` explicitly composes the existing BM guarded materializer,
single-file graph/full-text indexer and `SearchService.sync_entity_vectors`. The
opt-in host below owns their lifecycle. Indexing uses a
writer that **rejects frontmatter rewrites** before touching the published file.
Unsupported local normalization settings therefore leave graph progress pending
instead of changing Git's bytes. Safe automatic adaptation remains future work.

`models/git_projection_progress.py`, migration `0044` and
`repository/git_progress_repository.py` persist one local row per project and stage:

| Stage | Completion means |
| --- | --- |
| `content` | Exact current Git Markdown and portable UUIDs have reached SQL. |
| `files` | Every note passed BM's guarded materializer and exact byte verification. |
| `graph` | Graph/full text converged through a full pass or a proven body-only delta. |
| `vectors` | Every note completed vector synchronization or its explicit note opt-out. |

Rows retain target SHA, last completely processed SHA, running SHA/attempt UUID,
attempt count and the most recent error. Successful file recovery never implies
graph or vector completion. Missing semantic infrastructure and failed/deferred
vector results stay pending; they do not change a requested search mode. BM's
unchanged-vector outcome can count as both synced and skipped, which is accepted
without double-counting it as a failure. A vector configuration key invalidates
only the vector checkpoint when model/index/dimensions/token policy changes. Its
version must be bumped when the derived chunk/index policy changes in code.

`index/git_projection_worker.py` provides an explicit `async with worker.run()`
lifecycle. It owns two serial tasks: content → files → graph, and vectors after
graph completion. A slow embedding call cannot block newer content/file/graph
catch-up. Each lane retries failures with bounded exponential backoff (poll
interval × up to 64); publication notifications only reduce latency. A pass
superseded by a newer head is not a failure: it records `superseded`, keeps the
failure count and starts the next pass at once (vectors wait for the index lane). Polling Git
and SQL on startup and subsequent passes recovers even completely lost wakeups.
There is no per-write coroutine backlog and no new daemon. Local worker ownership
is now enforced by `index/git_project_ownership.py` before either task starts.

Work samples the latest published head and may coalesce A/B into C. A head change
supersedes an old pass: the content stage stops at once, the files stage finishes its
writes (each fenced by the materializer's version claim and the checksum-guarded
delete) and checks HEAD at its end; attempt/target guards reject late
acknowledgements. The graph stage holds the runtime's content gate per note, from its
HEAD check through the file index: indexing a file creates its entity when SQL has none,
so a delete published between the check and the index (its file is still on disk until
the files stage runs) would otherwise come back into SQL outside the head, and the next
write would refuse ("SQL content is outside the captured Git basis"). The worker
records completion only after work returns. Cancellation/process death before that
point leaves a durable running marker; a new owner repeats the stage safely.
Context exit cancels and awaits both tasks. SQL transactions do not span embedding
or Git/file I/O.

`worker.status()` reports the current source SHA plus each stage's distinct target,
completed and running SHA/error. `caught_up` also checks the active graph and vector policies.
These are **historical whole-stage completion checkpoints**, not a promise that
all rows remain on that revision during a newer partial pass. Ordinary V1 readers
can still mix published revisions; private drafts are not inputs. The host exposes
these checkpoints without adding a general V2 consistent read endpoint.

`test-int/test_git_projection_worker_integration.py` proves real API content reads,
graph relations, full-text results and physical vectors without a manual full-index
call. It tests A/B/C advancement during blocked vectors, independent vector failure,
configuration invalidation, preserved editor bytes, cancellation before/after each
stage and a fresh child interpreter terminating with `os._exit()` after each stage's
work but before checkpoint acknowledgement. Restart uses the same durable SQLite
and Git store. Both interpreters use the same deterministic test embedding methods
and provider identity; this is not an embedding benchmark or power-loss proof.

## Local root binding and exclusive ownership

The worker validates the project UUID/root against SQL and the Git manifest, and
rejects overlapping registered project roots or a bare repository inside an indexed
root. `filelock` provides two nonblocking OS ownership locks: one at the note root
and one at the bare repository. A second object or process cannot run the same
projection; another project can proceed independently. These are local filesystem
ownership locks, not long-held Git/SQL locks for an agent transaction.

Local-only binding metadata lives at `<root>/.basic-memory/git-binding.json` and
`<bare-repo>/bm-local-binding.json`; neither belongs to the published tree or an
ordinary Git clone/push. The paired markers bind project UUID, canonical root and
repository path. Both are checked before creating either. Interrupted first creation
can repair a missing matching marker; differing bindings require controlled remapping.
Control metadata symlinks are refused. SQL registration is rechecked for status,
each worker pass and each stage, so a live root change does not silently continue.

Context exit waits for the worker tasks and releases the locks. Process death also
releases the tested local OS locks; lock files remain and must not be unlinked by
cleanup. The verified environment is local Linux/SQLite. Other filesystem/platform
semantics and controlled root remapping still need their own operational validation.
This ownership does not prevent a legacy API, watcher or Cloud importer from writing;
controlling those sources remains mandatory before production enablement.

## Opt-in API/MCP host

`index/git_host.py` binds existing SQL project UUIDs to adopted bare repositories.
The new `git_repositories` configuration is empty by default. For an isolated,
already adopted project its shape is:

```json
{
  "git_repositories": {
    "11111111-1111-4111-8111-111111111111": "/absolute/private/project.git"
  }
}
```

The UUID must already match both SQL and the published manifest. This setting is
not an import command. Startup refuses mismatched roots, foreign manifests,
inactive projects, relative paths, Cloud projects and stateless/Postgres hosting.
The first supported topology requires **every configured and SQL-registered project
to be local and Git-managed**. A mixed host is refused because legacy target
changes could otherwise bypass cross-project graph invalidation. This is a current
implementation limit, not a requirement that projects share one note repository.
Two managed projects publish to separate repositories independently.

Normal API and MCP lifespans recover published SQL content before admission and
own all Git workers through shutdown. Off-lifespan writes fail with HTTP 503;
they never silently use legacy acceptance. A failed second owner restores the
first host's container, routing and database connection. Worker tasks finish or
cancel before their owner's database closes. Configuration reconciliation cannot
delete unmatched SQL projects before validation.

Normal note create/replace/edit/move/delete dependencies select the Git executor.
Post-acceptance materialization and relation scheduling notify the owned worker;
they do not start legacy jobs. API/MCP watchers and direct legacy file-index startup
are disabled. An API policy refuses unconnected imports, direct index/reindex,
directory mutations and project-registry mutations before acceptance. File-based
schema/prompt helpers are also unavailable until they read published content.
The native CLI refuses unconnected data-changing commands and suppresses automatic
upstream replacement. Agent tool calls use the normal hosted API/MCP path.

Lifecycle-owned local routing and project discovery remain local even when Cloud
credentials are present; Git project listings report direct Cloud sync unsupported.
This controls this host only: external editors, rclone/timers, older installations
and custom injected client factories still require operator control. The setting
does not stop any external process or prove the productive cutover prerequisite.

`GET /v2/projects/{project_uuid}/git/status` and
`bm_project(command="status", project="...")` return the source SHA, separate
stage targets/completions/running attempts, attempts/errors and `caught_up`.
While notes wait for another project (see "Note UUIDs across projects"), the content
stage also carries `deferred`: each waiting note with its owner project and that
owner's own content error, if it has one.
The response explicitly describes relaxed reads. It neither starts an index nor
waits for one, and a completed checkpoint does not pin concurrent readers.

SQLite deletion retains the existing targeted removal of that entity's FTS,
chunk manifest and physical sqlite-vec rows inside acceptance. There is no model
inference, project reindex or external vector request in this cleanup. Deferring
it naively could cascade away the mapping needed to remove physical vectors.
Tests delete an already indexed note while graph/vectors are blocked: the call
returns and all three owned search stores are cleared. This is bounded to the
deleted entity, not a latency guarantee for arbitrarily large notes.

`test-int/test_git_host_integration.py` uses normal lifespans and dependencies,
including real local-mode scheduling decisions, without injecting a mutation
executor. It proves ordinary API/MCP publication/read/search/progress, blocked
index return, startup recovery, topology refusal and independent repositories.
Deterministic embeddings remain a pipeline test, not a model timing benchmark.

## Explicit transactions

[BM006_TRANSACTIONS.md](BM006_TRANSACTIONS.md) documents the implemented API and
`bm_transaction` MCP tool: fixed bases, private own reads/writes, revision CAS,
capability ownership, commit/rollback/status/keep-alive, configurable 30-minute
inactivity and seven-day expired-draft retention. Source transfer freezes and pins
B/D while a new independent draft starts on current C; nothing is copied or
rebased automatically. Terminal evidence survives private-ref cleanup. Git
publication returns before projection completion; general V2 replay and snapshot
reads remain outside this interface. Existing implicit tools never join a draft
through hidden state.

## Remaining before productive use

The [opportunistic project sync](BM006_GIT_SYNC.md) is implemented through API and
`bm_project(command="sync")`: branch-only fetch/non-force push, validated incoming
history, single-ref fast-forward publication, durable observations and visible
divergence. Network waits do not hold local mutation admission. Private drafts and
local receipts stay local. Peer endpoints and production cutover are still unset.

Remaining implementation and operational obligations include:

1. Complete controlled first import/migration and operational host cutover. The isolated
   adoption primitive below preserves accepted-but-unwritten content, but does not
   stop legacy writers. Unindexed files, resources and noncanonical input remain
   protected. Define controlled root remapping. Index rewrites are refused;
   automatic normalization adaptation is not implemented. The body-only graph
   delta is isolated below; content, files and vectors still require whole-project
   traversal. Assess their measured catch-up cost before a productive pilot.
2. Extend supported controlled import/bulk/schema/prompt workflows when needed;
   they are currently refused. Validate packaging, deployment and recovery under
   the chosen operator configuration. Core runners prepare typed graph data and
   a pending-generation marker before commit; graph/FTS publication is deferred.
3. Preserve the captured Git basis before preparation and its verified cancellation,
   SQLite failure and whole-ASGI-process recovery behavior when wiring the host.
   Create/update/edit/move/delete are covered by the isolated executor; imports and
   bulk/directory operations must use the publisher or be refused before acceptance.
4. Use the implemented [offline backup/restore](BM006_BACKUP_RESTORE.md) in the
   pilot and verify an independent restore with actual pilot changes. Preserve
   caller-held keys separately. Measure retained-record maintenance
   on the intended transaction history. The explicit API is implemented, but
   branch-only cloning does not restore local operational state.
5. Validate actual peer access/deployment and clone/restore on both installations,
   operational state backup, retained-incoming-ref housekeeping and large-history
   validation cost. Divergence stops independently per project; reviewed external
   two-parent merges can then converge normally. A native merge-review API and an
   automatic network schedule are not part of this slice.
6. Before the **first productive Git publication**, control every legacy writer:
   API/MCP, active/in-flight materializers, watcher/editor imports, Cloud imports,
   timers and manual sync. Accepted but unwritten content must be included. The
   agreed initial cutover suspends direct Cloud file sync for converted projects
   and routes editor drafts through controlled import. Index drainage is not a
   substitute for write-path control. Migration and rollback retain new Git history.

Do not enable the experimental host on existing projects before these obligations
are implemented and the isolated boundary tests are extended accordingly.

## Reviewable first adoption

`services/git_project_adoption.py` provides `prepare_git_adoption()` and
`publish_git_adoption()`. They require externally stopped/drained legacy writers,
including accepted in-flight mutations, materializers, editor and Cloud imports.
The existing root/repository ownership excludes other Git workers; it does not
prove that external precondition or perform a productive cutover.

Preparation requires an unused initialization commit and preserves project/note
UUIDs and exact bytes in local `bm-adoption.json`. Accepted NoteContent takes
precedence over an older or missing materialized file. Old disk bytes and materialization
lineage are retained as migration evidence. File-first notes require the indexed
file checksum. Unknown editor bytes, unindexed files, symlinks, unreadable directories,
non-Markdown resources/entities and noncanonical frontmatter are refused before
publication. Unknown control files are not silently excluded. The reserved root
control directory contains only the owned Git binding/lock files during adoption.

The retained plan is not silently refreshed. Publication revalidates SQL/file state,
root registration and the fixed Git basis, then publishes through the same CAS.
Retry resolves its terminal transaction before comparing the old migration basis:
normal projection may already have advanced SQL/materialized files after a lost
answer. The plan is local operational evidence containing private content; retain
it with backups. An ordinary Git clone or branch push does not carry it.

`settle_git_publication()` in the Git repository module waits for publication to
settle even if the caller is cancelled repeatedly. Ownership/admission cannot be
released while a detached thread can still change the published branch. Cancellation
still reaches the caller; durable Git determines subsequent recovery.

## Opt-in startup and real API acceptance

`services/git_project_runtime.py` implements the optional mutation executor accepted
by `NoteContentMutationService`. Git-configured host dependencies select it. Its lifecycle
owns the existing Git worker and first restores SQL content before yielding admission;
it does not wait for files, graph or vectors. The exact restored startup commit is
retained, rather than incorrectly sampling a potentially newer head afterward.

Each implicit mutation acquires a short local admission gate, captures immutable Git
before calling the real BM preparer/runner, and verifies that SQL contains that
published basis. A newer peer publication first causes content recovery. Private SQL
bytes/checksums/identities cannot become the basis of a new publication. The worker
shares the gate only for SQL content projection; vectors never hold it. Explicit
agent planning must not run inside this implicit mutation gate.

Create, update, append/edit, move and delete prepare through the normal BM core.
Git publication precedes SQL commit; old-file permissions survive successful SQL
move/delete even after the old entity is gone. The returned change suppresses legacy
materialization/cleanup and the service skips its legacy post-commit graph/FTS work.
The lifecycle worker performs files/graph/FTS/vectors. Canonical normalization and
validation, typed graph preparation and a pending-generation marker remain inside
acceptance; they do not publish observation/relation/FTS rows. The blocked-worker
API test verifies those rows and vector manifests are still absent at response time.
Existing delete runners retain the targeted SQLite search/vector cleanup described
above; expensive catch-up remains owned by the worker.

A changed Git basis returns a distinct HTTP 409 publication conflict, including base
and actual source commits. It must not be converted by `write_note()` into an
`already_exists` path outcome. A failed SQL commit after successful Git publication
reports HTTP 503 with `published_commit` and an instruction to inspect status before
retrying; it never claims the Git write was rolled back. This is not V2 operation
deduplication. Cancellation/process loss recovers from Git without replaying append.

Both Git content/layout projection and this acceptance path use BM's existing
`db.scoped_session` for writes. SQLite foreign-key enforcement is connection-local;
a raw fresh session can otherwise leave an incoming relation pointing at a deleted
target. A fresh-connection regression verifies preserved links with `to_id=NULL`.
This adds no row locks or wider derived-state transaction.

`test-int/test_git_runtime_integration.py` exercises the real API with an explicitly
injected executor, blocked graph/vector work, concurrent writes, peer publication
during preparation, SQL commit failure, shutdown and repeated cancellation.
`git_runtime_crash.py` terminates the entire in-process ASGI interpreter before/after
Git publication and before/after SQLite's commit call. Restart uses the real worker
against the same Git/SQLite state. Background loops are paused only in the dying
child to target the foreground commit boundary. This is not a TCP server or a
power-loss proof. All ordinary tests use deterministic embeddings.

## Safe body-only graph catch-up

Migration `0046` adds a nullable `reusable_commit` to the local progress rows.
Historical completion alone cannot justify skipping unchanged notes: an
unacknowledged intermediate pass may already have replaced some graph rows.
The graph attempt atomically claims its observed reusable basis and clears it
**before** work begins. Only the same successful attempt restores it. Failed,
cancelled or killed attempts therefore force a full retry, even if C restores
the body from A after a partially indexed B. Late acknowledgements stay fenced.

When UUIDs, paths and all frontmatter values are unchanged, the graph stage
reindexes only notes whose Markdown bytes changed since that proven basis.
It uses the same guarded single-file indexer, generation publisher and exact-byte
checks. There is no replacement parser or independent FTS writer. Structural
changes conservatively retain the existing full graph pass.

Both SQL content/layout recovery and the optional mutation executor invalidate
graph reuse before/in the transaction changing identities or frontmatter.
Deleting and recreating an unchanged target between graph attempts thus cannot
leave incoming links permanently unresolved. This also fences active graph and
vector attempts across locally registered Git projects, since other projects'
sources can reference changed targets. Their graph/vector completion is cleared;
content/file checkpoints and independent Git publication bases are preserved.
The worker rereads progress after content recovery and CAS-checks the graph basis
when claiming work. No new row locks or planning-time locks are introduced.

A separate graph policy key covers the implementation version, permalink policy,
normalization and workspace prefix. Changing it rebuilds graph and vectors;
changing only the embedding policy continues to invalidate vectors alone.
Content, files and vectors still traverse the whole project in this slice.
Registry renames, external writers and imports must be controlled or routed
through equivalent invalidation at the host boundary. The current host refuses
mixed legacy/Git projects and unconnected writers instead of allowing an unmanaged
target to bypass invalidation of its Git sources.

`test-int/test_git_graph_delta_integration.py` exercises real graph/FTS/vectors,
a failed intermediate pass, interpreter death after a delta but before its
checkpoint, deletion before any graph attempt, target recreation and graph policy
changes. A real API append regression verifies that an unchanged peer note is
skipped. These are isolated SQLite proofs; opt-in normal host selection is tested
separately as described above.

On the same private 593-note/5,981,450-byte corpus, one body edit indexed one
graph source instead of 593. Graph/FTS stage time fell from 184.12 s to 3.23 s;
the sum of Git publication and sequential projection stages fell from 234.38 s
to 47.72 s. These are single comparison runs with deterministic vectors under
the Jina-DE provider identity, not model-inference, API-response or percentile
latency measurements. Full initial indexing still visits every note; observed
differences in that initial run are not a delta-path speedup. SQL content, files
and vectors account for the remaining whole-project work.

## Offline backup and independent restore

[BM006_BACKUP_RESTORE.md](BM006_BACKUP_RESTORE.md) defines the implemented cold
backup and independent restore. `bm git backup`, `bm git restore` and
`bm git finish-restore` preserve all private refs/transactions/receipts, configuration
and recovery evidence, remap local roots, and rebuild SQL, files, graph/FTS and
vectors. They do not initialize the caller's database or overwrite an existing
restore target. Unpublished working files remain outside the restored indexed roots.

## Upstream and migration maintenance

The fork merges upstream instead of rebasing published history. Upstream snapshot
`26d0a966` adds its own revision 0043. In the merged experimental chain the three
BM-006 migrations are 0044 (progress), 0045 (retirement) and 0046 (graph basis),
with exactly one Alembic head. Upstream later added its own `0044`
(`entity_embedded_checksum`); it collided with BM-006's `0044`, so the fork carries it as
`0047` after `0046` (released BM-006 revisions are never renumbered). Upstream's
`0045` (`drop_frontmatter_btree_indexes`, 2026-11-05) collided the same way and is carried as
`0048` after `0047`. Earlier experimental databases using BM-006's old
0043–0045 numbering cannot be upgraded through this chain automatically. Use new
disposable development databases; non-disposable databases require a separate,
backed-up recovery operation. Never infer schema identity from a numeric revision
without its matching code history.

Once applied to persistent installations, fork migration IDs must remain stable.
Future newly imported, unapplied upstream migrations must be assigned the next
available fork revisions and recorded with their original upstream identity;
already applied BM migrations must not be renumbered on every merge. Recheck the
linear graph, upgrade behavior and both parents' regressions at each merge.

## Validation

Use [DEVELOPMENT_ISOLATION.md](DEVELOPMENT_ISOLATION.md) before running development
code on an installed host. Environment variables alone are insufficient because
upstream test fixtures clear them. The launcher masks the real default profile,
makes the host read-only and gives child processes only private writable state.

Acceptance uses the full official SQLite unit and integration recipes with
`--import-mode=importlib`, including real Redis cache integration, plus the combined
Git/crash suite, changed-file Ruff/format/type checks, whole-tree typing and Doctor.
PostgreSQL, model-quality/semantic benchmarks, real peer deployment and power-loss
behavior require separate evidence. Historical focused test counts in earlier
slices were partial regression selections, not full-suite acceptance.

Shared real-host fixtures live in `tests/git_host_support.py`. The Git status
schema no longer imports the SQL repository layer during CLI registration. The
`bm_project` tool declares destructive behavior because explicit peer sync can
apply published note deletions; its interface and manual explain that behavior.

Use directory-based coverage (`--cov=src/basic_memory`), with the aggregate
coverage threshold disabled for a focused selection, then inspect coverage for
changed modules. Module-name coverage can import parent service packages too early
and trigger a Pydantic initialization assertion.
