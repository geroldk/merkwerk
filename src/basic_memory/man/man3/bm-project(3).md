---
title: bm-project(3)
type: manpage
section: 3
name: bm-project
summary: list, create, and show recent activity or Git progress in projects
generated: registry
tool: bm_project
verified: 0.24.1 mcp
---

# bm-project(3)

## NAME

**bm-project** — list, create, and show recent activity or Git progress in projects

## SYNOPSIS

```
bm_project(command, name=None, path=None, set_default=False, workspace=None,
           since="7d", review=None, items=None, page=1, page_size=10,
           output_format="text", project=None)
```

## DESCRIPTION

`command` picks what to do:

- `list` — the projects you can address, grouped by workspace.
- `create` — add a project: `name`, `path`, optional `set_default` and
  `workspace`.
- `recent` — what changed lately in the required `project`: `since`, `items`, paging. JSON
  output carries `page`, `page_size` and `has_more`, plus `guidance` for any
  project that was never indexed.
- `status` — local Git publication and index progress for the required `project`.
  Reports source revision and separate SQL content, file, graph/full-text and
  vector completion. This observes progress; it does not wait or pin a snapshot.
  Available on an explicitly configured local Git host. While notes of the project
  wait for another project to let go of their ids (a move between projects still in
  transit), the content stage lists them in `deferred` (note, owner project, and the
  owner's own error when it fails); `caught_up` stays false and writes to the project
  are refused as `UNAVAILABLE` until the owner has caught up or synced.
- `sync` — explicitly fetch/push the configured private Git peer for `project`.
  Local writes work offline. Incoming fast-forwards are validated before publication;
  divergence stops for review. JSON returns the last attempt and observed relationship.
  Inspect `last_attempt.kind`: HTTP success can carry `failed` after a transport error.
  `status` includes the same passive synchronization observation. No timer is created.

Git `sync` can apply note deletions recorded in published peer history; the tool
declares destructive behavior. `bm_project_delete` removes whole projects and is
available only when the server enables it.

**Hosted server only.** The hosted Basic Memory Cloud server replaces this
tool with one that has two more commands and three more parameters. A local
server has neither.

- `info` — how to register for Basic Memory Cloud and start a trial. If
  project or note tools fail, send the person to the Basic Memory app, then
  retry.
- `usage` — the workspace's credit usage, for owners and admins. `workspace`
  picks the workspace and `project` must be left out, because usage is
  counted per workspace. `since` is the first UTC day (`YYYY-MM-DD`) or a
  lookback such as `30d`. For `usage` it defaults to the start of the current
  billing period, not the `7d` shown in PARAMETERS above, which applies to
  `recent`. `until` is the last UTC day (`YYYY-MM-DD`) and
  defaults to today. `group_by` is `actor` (who used credits, and background
  job types; the default), `day` (credits per day and who used them) or
  `job_type` (background work by job type, per day). `customer` is for
  partner staff only: a managed customer's slug, id or name, or `all` for
  every managed customer this month.

While the local server is running, its watcher picks up a new project's folder
at the next project reload. Once the watch is attached, a one-time index pass
includes files copied in before attachment; later changes use normal file events.

The shell spelling of `recent` is `bm tool project recent`.

The experimental Git host owns all configured local projects and disables the
legacy watcher, imports, registry edits and directory mutations. External editors
and Cloud file sync must be controlled before enabling it. Ordinary reads may mix
published revisions while indexes catch up.

## PARAMETERS

- **command** (required) — What to do.
- **name** (string | null, optional, default: None) — For create: the new project's name.
- **path** (string | null, optional, default: None) — For create: the folder that holds the project's notes.
- **set_default** (boolean, optional, default: False) — For create: make the new project the default.
- **workspace** (string | null, optional, default: None) — In Cloud: the workspace slug, name or id that holds the project.
- **since** (string, optional, default: "7d") — How far back to look: '7d', '24h', '2 days ago', 'last week', or a date. Windows shorter than one day are widened to one day.
- **review** (string | null, optional, default: None) — For merge: the review package id.
- **items** (array | null, optional, default: None) — What to return: notes, observations, relations, or a mix. Default: notes.
- **page** (integer, optional, default: 1) — Page number.
- **page_size** (integer, optional, default: 10) — Results per page (1-100).
- **output_format** (optional, default: "text")
- **project** (string | null, optional, default: None) — Project name, 'workspace/project', or external id. Required for one-project operations; search may instead use `projects` or `all_projects=true`. Project discovery needs no scope.

## EXAMPLES

```
bm_project(command="create", name="demo", path="/Users/me/demo", set_default=True)
# → ✓ Project 'demo' added successfully
#   ...
#   Project is now available for use in tool calls; indexing completed.

bm_project(command="list")
# → Available projects:
#   - demo (local) [b2a21302…]
#   - main (local) [9b4d5944…]

bm_project(command="recent", since="1d", project="demo")
# → ## Recent Activity: demo (1d)
#   **📄 Recent Notes & Documents (1):**
#     • launch-plan (plans) [id: 686e696d…]
```

## SEE ALSO

- see_also [[bm-project-delete(3)]]
- see_also [[bm-list(3)]]
