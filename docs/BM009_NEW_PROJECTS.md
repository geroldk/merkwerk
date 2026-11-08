# BM-009: new Git projects and peers, offline

Status: built and verified in isolation, 2026-11-08. Nothing here changes a running service.

The Git host builds one runtime per project when it starts and refuses a profile in which a
registered project is not Git-managed. New projects and peers are therefore set up with the
service **stopped**, and take effect at the next start (about two minutes with a backup).

## `bm git create-project NAME [NAME=/absolute/path ...] [--root DIR] [--repositories DIR]`

Creates new, empty Git projects: database row, note root, bare repository with the first
commit, the binding markers a backup needs, and the `projects` / `git_repositories` entries.

* Names match `[a-z][a-z0-9-]{1,31}` and are unique regardless of case (`PKM` blocks `pkm`).
* Default root `<root>/NAME` (`--root` defaults to `~/bm`), repository `<repositories>/NAME.git`.
* Refused: a root in, over or equal to another project or the base, in `repositories`, `peer`,
  `operations`, `code`, `vendor`, `bin`, `instanz`, `systemd`, a non-empty existing directory,
  a repository path in use, a profile that is not a Git host, a database that is not at the
  Alembic head, a running service (it holds the project locks).
* All or nothing: the future profile is checked with the host's own start rules before the
  configuration is replaced (atomically, last). On any failure the database rows and the
  directories this call made are removed.
* **Rights of `config.json`:** the file is replaced through a temporary file that gets the rights
  of the old one (a new file starts at 0600), also when `remove-project` puts the old text back
  and for `peer-setup`. The configuration can hold credentials.
* **Signals:** SIGTERM, SIGHUP and SIGINT (a `kill`, a closed terminal, Ctrl-C) cancel the work and
  reach the rollback like any other failure; a second signal is ignored while the rollback runs. The
  command then prints `Cancelled by a signal; the changes were rolled back.` and exits 130. `remove-project`
  decides which configuration is right by the database (a cancellation during the commit cannot stop
  the commit). Only SIGKILL, a power cut or an out-of-memory kill can leave a
  half-made state: database rows the configuration does not know make the service refuse to
  start. Take a backup first (the move-day runbook does) and run the tool in tmux.
* Not touched: `sync-projects.json`, peers, and the gateway project lists. The gateway lists
  (`read_projects`, `write_projects`, `known_projects`) are changed by whoever owns the gateway.

## `bm git remove-project NAME...`

The way back for `create-project`: removes empty projects again. Service stopped, as above.

* Only a project that is still what `create-project` made: no entity rows, a repository that
  holds nothing but its initialization commit **and no other reference** (drafts under
  `refs/bm/candidates`, receipts, fetched peer references) **and no transaction pointer** in
  `bm-transactions` (a note that was written and deleted counts as
  used, because its history would go with the repository), a note root that holds nothing but the
  control directory `.basic-memory`. Everything else is refused before anything changes.
* Also refused: an unknown name, the default project, removing every project (the host needs at
  least one), a directory shared with a project that stays, a running service.
* The database rows and the configuration change together. The rows are deleted inside a
  transaction that stays open while the configuration is replaced atomically; the transaction
  commits last. If either step fails, both are as they were. Then the root and the repository
  are deleted. A directory that cannot be deleted is reported, not hidden.
* The configuration entries of the project go with it: `projects`, `git_repositories`,
  `git_sync_remotes`, `git_sync_merge`. A peer repository (the other end) is **not** touched:
  the command says so when the project had one. A project created again under the same name
  gets a new identity and must not be pointed at the old peer repository.
* Not touched, as for `create-project`: `sync-projects.json` and the gateway project lists.

## `bm git peer-setup NAME... --peer BASE [--merge auto|off]` / `--remove`

Sets the sync peer of named projects to `BASE/NAME.git` (BASE: an absolute directory, whose
bare repositories are created if missing, or `ssh://host/dir`). `--merge auto` also enables
automatic merging for them. A project not named keeps its settings, and a project may have no
peer at all. `--remove` removes peer and merge setting of the named projects. A local peer may
not be a note root or a project's own repository (nor over or inside one): the project would be
its own peer. A failed creation of a local peer removes the directories this call made.

## `bm_move` across projects: the permalink is formed again

A note moved with `destination_project` is written in the target project as a new note, then
deleted in the source. Until now the copy carried the source's frontmatter verbatim, permalink
included, so a note moved from `main` into `cobol-ana` kept `personal/main/tasks/...`. A
gateway refuses addresses that start with another project's name (or `personal`), so such a
note could not be reached remotely.

* The copy is written **without** its `permalink` key (the one line is removed from the text before
  the write). The target forms the permalink from the new path, exactly as for a note created
  there: `cobol-ana/aufgaben/jek-...`. An already owned permalink gets a `-1` suffix. Works the same
  whether the old permalink began with `personal/main/`, `main/` or anything else.
* This is part of the move, so `created` and `modified` stay as they were (BM-010 rule 4); a
  later `set_metadata` would renew `modified` on every moved note. Every other frontmatter key keeps
  its value and the body its text; the copy is written like any new note, so the frontmatter is
  re-serialised (key order and quoting may change) and leading/trailing blank lines of the body
  are trimmed. Precondition on the Git host: the note already carries the two keys (the BM-010
  backfill), else the copy is dated with the move.
* The result names both: `previous_permalink` and `permalink`, and the identity of the new note
  (`previous_external_id`, `external_id`); a directory move lists one entry per note in
  `moved_notes` (`previous_file_path`, `file_path`, `previous_permalink`, `permalink`,
  `previous_external_id`, `external_id`). If some notes of a directory fail, the `PARTIAL_FAILURE`
  refusal lists exactly what did move (JSON form: `details.moved_notes`, `details.failed`; text form: the
  message ends with `Moved: ...`), because those notes are already gone from the source. Use
  `output_format="json"` for a move you want to log: the text form of a single note names the permalinks
  only, the identities are in the JSON form. The text form ends with `; permalink OLD -> NEW`.
* A move inside one project is untouched: it keeps the permalink unless
  `update_permalinks_on_move` is on (production: off).
* Links in other notes that name the old permalink are not rewritten (as before); the move
  procedure lists them (`verweise-*.tsv`).

## Restore

`bm git restore` now clears `git_sync_merge` together with `git_sync_remotes`; a host refuses
merge settings for a project without a peer, so a restore of a merge-enabled profile would not
start.

## Evidence

* `tests/services/test_git_project_create.py` (plan rules and the running-service lock).
* `test-int/test_git_create_project_integration.py`: the host starts with the new projects, a
  write makes exactly one commit, a backup includes them, nothing is left behind after a failed
  write or a refusal.
* `test-int/test_git_peer_setup_integration.py`: peer creation, first sync `pushed`, removal,
  ssh base, refusals, rollback.
* `tests/services/test_git_project_remove.py` and `test-int/test_git_remove_project_integration.py`:
  create then remove restores the profile and the database exactly, a project with notes or with
  history stays (also when it was requested together with an empty one), the default project, the
  last project, foreign files and a running service are refused, and a failure while replacing
  the configuration, or after it, leaves rows, directories and configuration as they were.
* `tests/mcp/test_tool_bm_move_across_projects.py` and
  `test-int/test_git_cross_project_move_integration.py`: one case per source prefix
  (`personal/main/`, `main/`, other); the data stay the same (both times, every other key, the
  body, SQL instants); a taken permalink gets a suffix; a move inside a project keeps the
  permalink with `update_permalinks_on_move` off and forms it again with it on. Without the
  change seven of these fail.
* `tests/utils/test_file_utils.py`: `drop_frontmatter_key`.
* `tests/cli/test_git_projects_cli.py`: arguments and refusals of the three commands.
* Rehearsal on a copy of the real data (7 projects after `create-project haus meldungen`):
  the host starts, all projects `caught_up`, a write in each new project, `haus` syncs to its new
  peer (`pushed`), `meldungen` without a peer is skipped silently by `git_sync_watch.py`.
