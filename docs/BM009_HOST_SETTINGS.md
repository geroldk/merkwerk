# BM-009: settings per host and `bm-notify`

Status: built and verified in isolation, 2026-11-08. Nothing here is installed or switched on.

Everything about the scheduled Git sync that differs from one machine to the next is
configuration, not code: the host name, which projects are visited, where a message goes. It
lives in one file **outside** `config.json`. `bm git backup` saves `config.json` only, so a
restore on another host never inherits these settings.

## The settings file

Looked for at `--settings FILE`, else `$MERKWERK_HOST_SETTINGS`, else
`~/.config/merkwerk/host.json`. Every key is optional; unknown keys, wrong types and a bad
value are **refused** (a misspelt key never falls back silently).

```json
{
  "host_name": "ryzen",
  "bm_url": "http://127.0.0.1:8765/mcp",
  "notify": {
    "project": "meldungen-ryzen",
    "directory": "abgleich",
    "spool": "~/.local/state/bm-git-sync/spool",
    "stale_hours": 24,
    "max_attempts": 5
  },
  "sync": {
    "projects": ["main", "pkm"],
    "tolerate": 3,
    "state": "~/.local/state/bm-git-sync/state.json"
  }
}
```

| Key | Meaning | Default |
|---|---|---|
| `host_name` | name in every message (`[a-z0-9-]`, up to 31) | the machine's short name |
| `bm_url` | the local service | `http://127.0.0.1:8765/mcp` |
| `notify.project` | project that receives one note per message; `null` = journal only | `null` |
| `notify.directory` | folder inside that project | `abgleich` |
| `notify.spool` | directory of waiting messages | `~/.local/state/bm-git-sync/spool` |
| `notify.stale_hours` | a waiting message older than this is called out on every run | 24 |
| `notify.max_attempts` | refusals before a message is set aside in `rejected/` | 5 |
| `sync.projects` | projects the schedule visits; `null` = all the service knows | `null` |
| `sync.tolerate` | failures in a row before "unreachable" is reported | 3 |
| `sync.state` | state file of the schedule | `~/.local/state/bm-git-sync/state.json` |

`python bm_host_settings.py [--settings FILE]` prints the effective settings (or the refusal).
Command-line flags of `git_sync_watch.py` win over the file, the file over the defaults.

**Variant A or B** (decision N1) is not a switch but a consequence of which projects have a peer
(`bm git peer-setup`) and which `notify.project` each host names: one shared, synced project
for all hosts (A), or one per host with only the Ryzen's synced to nuc1 (B). A machine that
only needs A sets nothing or `notify.project: null`. Which projects the schedule visits is
`sync.projects`; a project without a peer is skipped by the sync itself (`meldungen` on nuc1).

## `bm-notify`

`git_sync_watch.py --notify-cmd bm-notify --notify-always` hands the report to it on standard
input, one message per line, **on every pass** (empty when nothing is new, so that waiting
messages are delivered as soon as the service is back). For each message:

1. the line goes to standard error (the journal), first;
2. it is written to the spool, one small file per message (mode 0600, directory 0700), named
   by time, a running number (the order within one report) and the message id;
3. a note is written with `bm_write` into `notify.project`/`notify.directory`, and the spool file
   is deleted. Title: `Abgleich <host> <YYYYMMDD-HHMMSS> <project> <id>`; the body starts with
   `Meldung: <line>` and has the observations `host`, `zeit`, `projekt`, `problem` and
   `status offen` (acknowledge with `bm_edit`: `offen` -> `erledigt`); type `meldung`, tags
   `abgleich` and the host, frontmatter `host`. `created`/`modified` come from BM-010.

* **Service away:** the messages stay in the spool and go out in order on the next run (N4 A).
  The run exits 0; a message is lost only if it cannot even be spooled (exit 1, and it is in the
  journal already).
* **Service refuses a note** (project missing, read-only): retried each run up to
  `max_attempts`, then moved to `<spool>/rejected/` and reported on standard error. It never
  blocks the messages behind it.
* **Idempotent:** the note name carries the id, and "already exists" counts as delivered, so a
  run cut off between the note and the deletion does not create a second note.
* **Safe text:** control characters are removed, a line is cut at 2000 characters, the body
  never starts with `---` and `[[ ]]` are broken up, so a report line cannot open frontmatter or
  make a link.
* `bm-notify --status` shows waiting/stale/rejected and exits 2 if something is stale or
  rejected.

Why a directory and not one spool file: the schedule must be able to add a message while another
run delivers; with one file a rewrite could drop the new line, and the sync reports a problem only
once.

## Fixes to `git_sync_watch.py` on the way

* **The service not answering at all** crashed the script with a traceback (the documented
  "transport failure for every project" never happened), so no message could ever come. It now
  counts a transport failure for every project to visit (the configured ones, else those seen on
  earlier passes, else the one name `dienst`) and reports after `--tolerate` passes.
* The notify command can run on every pass (`--notify-always`).

## Installation (later, with Gerold's word)

Fixed copies, like `git_sync_watch.py` today: `~/bm/bin/` holds `git_sync_watch.py`,
`bm_host_settings.py`, `bm_notify.py` and `bm-notify` (mode 755, with a SHA256 line). The
template `deploy/systemd/bm-git-sync.service` already points there. The wrapper `bm-notify`
starts `bm_notify.py` with the Python of the BM uv tool (`BM_PYTHON` overrides), which has
fastmcp. No restart of Basic Memory is needed. The notify project itself comes from
`bm git create-project` (BM-009).

## Evidence

* `tests/test_bm_host_settings.py`, `tests/test_bm_notify.py`, `tests/test_git_sync_watch.py`.
* Rehearsal on a copy of the real data with the real scripts: report -> note in the notify project;
  service address wrong -> both messages spooled, exit 0, the next empty run delivers them in
  order, a repeat run is empty; the watch with an unreachable service -> `unreachable:transport`
  for each project -> `bm-notify` -> notes; the service back -> "wieder in Ordnung" note.
