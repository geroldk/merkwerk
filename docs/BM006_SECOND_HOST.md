# BM-006: installing a second host (for example the Ryzen)

Status: exercised end to end on 2026-11-07 with synthetic projects (wheel `c3456e5c`): install on the
second host, `bm git backup` on a copy of the first host, `bm git restore`, peer enabled on both sides,
sync in both directions, diverged history with `git_sync_merge` off (the project stops, others continue)
and `auto` (simple merge and a review package). Findings: a restored instance places each project's
notes in a directory named by its project UUID; the restore leaves `git_sync_remotes` empty;
`git_sync_remotes` and `git_sync_merge` are read only at service start; a review decision must be
an object (`{"action": "beide"}`), a bare word is reported as `invalid_incoming`. Real-size restore
timing on the second host is not yet measured.

## Order matters

1. The first host (nuc1) is migrated to this fork and its projects are adopted and published.
2. The first host's shared peer (one bare repository per project) is created and filled.
3. Only then the second host is installed. A second history initialized independently is **not**
   a clone and will never sync.
4. A host never mixes: every project on it is local and Git-managed, and no rclone mirror runs.

## Steps on the second host

1. **Install the fork at a pinned commit** (never a floating branch; automatic updates stay off):

   ```sh
   uv tool install "basic-memory @ git+https://github.com/geroldk/merkwerk@<commit>" --prerelease=allow
   ```

   Then set `auto_update` to false in the profile before the first run.

2. **Bootstrap the data from a backup of the first host.** There is no native "clone from peer"
   command; `bm git restore` is the supported bootstrap and keeps all project and note UUIDs:

   ```sh
   # on the first host, services stopped (see BM006_BACKUP_RESTORE.md)
   bm git backup /path/private/memory.bm.zip --config <first host config.json>
   # copy the archive over a trusted channel, then on the second host:
   bm git restore /path/private/memory.bm.zip <new instance dir> --timeout 900
   ```

   The restored profile has the peer disabled. Select the new instance explicitly:
   `BASIC_MEMORY_CONFIG_DIR=<new instance dir>/config`.

3. **Enable the peer after review.** Add `git_sync_remotes` for each project UUID, pointing at the
   first host's bare repository (`ssh://…` with a dedicated key that may only run Git; see
   BM006_GIT_SYNC.md). Credentials live in the SSH configuration, never in the URL.

4. **Run the service** as a user unit on `127.0.0.1:8765` (same shape as the first host's unit,
   environment `BASIC_MEMORY_CONFIG_DIR` set to the restored profile) and enable the schedule
   from [BM006_GIT_SYNC_SCHEDULE.md](BM006_GIT_SYNC_SCHEDULE.md).

5. **Point the clients at it:** `bm-local` (`http://127.0.0.1:8765/mcp`) for Claude Code and
   Codex, and deny the Cloud connectors there.

## Not in this procedure

Tailscale/Funnel (the second host stays off the internet path), rclone, the Cloud account.
