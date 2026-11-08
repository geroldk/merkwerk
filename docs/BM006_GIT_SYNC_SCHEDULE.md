# BM-006: scheduled peer sync and the divergence report

The Git sync itself has no timer ([BM006_GIT_SYNC.md](BM006_GIT_SYNC.md)). This is the external
schedule. Nothing here is installed automatically.

## Parts

- `scripts/git_sync_watch.py`: one pass. For each project it calls `bm_project(command="sync")`
  on the running local service (`http://127.0.0.1:8765/mcp`), so the repositories are never opened
  by a second process. Projects without a peer (`409 No Git peer`) are skipped, a sync already in
  progress for the project is skipped until the next pass.
- `deploy/systemd/bm-git-sync.{service,timer}`: templates for a user timer, every 5 minutes.
- Per-machine settings (host name, projects to visit, notify project) and `bm-notify`, which turns
  a report into notes and keeps a queue while the service is away: see
  [BM009_HOST_SETTINGS.md](BM009_HOST_SETTINGS.md).

## What is reported

| Result | Reported | Why |
| --- | --- | --- |
| `equal`, `pushed`, `pulled` | never | normal |
| `diverged`, `remote_rewritten` | at once | both sides published different history; a person reviews and publishes a merge (see "Divergence and reviewed recovery") |
| `needs_review` | at once | merging is on (`git_sync_merge`), but notes conflict; the report names kinds and paths. Decide in the review package, then `bm_project(command="merge")` (see "Note-level merge"). `merged` is routine and silent |
| `invalid_incoming`, `local_changed` | at once | the peer sent something invalid, or the local side moved during the attempt |
| `transport`, `interrupted`, or the service not answering | after 3 passes in a row (`--tolerate`) | normal while offline |

A problem is reported when it appears and once more when it clears, not on every pass. State
is kept in `~/.local/state/bm-git-sync/state.json`. Exit status 3 means a project needs a person.

The report goes to stderr (the journal) and, with `--notify-cmd`, to that command's standard
input. The channel (mail, push message, file) is a decision of the operator; no default is set.

## Stop rule

Never run the schedule for a project whose peer is not yet set up or whose history was not
cloned from the peer; and never together with an rclone mirror of the same project.
