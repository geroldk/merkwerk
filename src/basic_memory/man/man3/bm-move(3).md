---
title: bm-move(3)
type: manpage
section: 3
name: bm-move
summary: move or rename a note or a directory, within a project or into another
generated: registry
tool: bm_move
verified: 0.24.1 mcp
---

# bm-move(3)

## NAME

**bm-move** — move or rename a note or a directory, within a project or into another

## SYNOPSIS

```
bm_move(destination, project, identifier=None, path=None,
        expected_checksum=None, destination_project=None,
        output_format="text")
```

## DESCRIPTION

Moves a note (`identifier`) or a directory (`path`). `destination` is the new
project-relative path; ending it with `/` moves into that folder under the
current name (`archive/`). A destination that names only the project
(`demo/`) is its root.

A note keeps its file extension (`.markdown` may become `.md`). A folder that
differs from an existing one only in case or Unicode form lands in the existing
folder, and the result names it.

Pass `expected_checksum` (from bm_read) to move a note only if it is still the
revision you read.

`destination_project` moves into another project; `destination` is then a path
in that project. Links in either project are not rewritten. Its permalink is
formed again from its new path, as for a note created there (a permalink that
carried the source project's prefix would point into the wrong project; a
permalink already taken there gets a `-1` suffix). Every other frontmatter key,
including `created` and `modified`, keeps its value; a note without those keys
gets its indexed times written in, never the time of the move. The note is
written like any new note, so the frontmatter is re-serialised and the body's
outer blank lines are trimmed. In the JSON form the result names the old and
the new permalink and the identity (`previous_permalink`, `permalink`,
`previous_external_id`, `external_id`; for a directory, one entry each in
`moved_notes`). A note already at the destination is never replaced, a note
whose title the target already has (ignoring case) is refused, and a locked
note is refused. A directory moves note by note; files that are not notes
stay, and the result lists whatever did not move. If some notes of a directory
fail, the `PARTIAL_FAILURE` refusal lists what did move (in the JSON form as
`details.moved_notes`), because those notes are already gone from the source.

On a host that publishes its projects through Git, the note keeps its id:
`external_id` equals `previous_external_id` (kept in the result for callers of
the older form), and the text form says `id kept`. The server writes a private
draft in the target under the old id, deletes the note in the source (only if
it is still the revision read), then publishes the draft with a proof of the
move that peers read. Every other write to the target waits meanwhile. If the
source refuses the delete, the draft is dropped and nothing has moved. A move
that stops after the source let go (a crash, or the target could not publish
yet) is kept with its content and finishes at the next start of the host or
with `POST /v2/git/moves/resume`; until then the call answers
`PARTIAL_FAILURE` with `details.move_id`, `details.state` (`pending` or
`stuck`) and `details.external_id`, and the id stays reserved (no other note
may take it). `stuck` means the planned path was taken meanwhile: a person
chooses with `POST /v2/git/moves/{move_id}/resolve`, `{"choice": "target",
"destination_path": ...}` or `{"choice": "source"}` (back where it came from).
`GET /v2/git/moves` lists open moves. A directory moves note by note through the
same route, with every id kept, and the target's index catches up once at the
end; the text form lists every note, one line each, without an upper limit. As a
rough guide, a folder of 300 notes moved into a project of 1000 notes in about
13 minutes (2.6 s per note) and one of 600 in about 31 minutes, on one machine
with both projects idle; graph and search indexing follow for some minutes
more. A caller whose answer is lost finds the moved notes in the target, each
with its id, and the others still in the source.
On any other host the note is written in the target as a new note (a new id and
its own version history) and then deleted here; if that delete is refused, the
copy is removed again and nothing has moved.

On a host that publishes its projects through Git, a directory moves within
its project as one commit, all or nothing. Every note keeps its id, its
permalink and its `created` and `modified`; a note that lacks those keys gets
its indexed times written in, as for a single move. There is no upper limit on
the number of notes, and the result lists every one: old path, new path, id and
permalink, one line each in the text form, and in the JSON form `moved_notes`
(`previous_file_path`, `file_path`, `external_id`, `permalink`) next to
`moved_files` (the new paths) and `commit_id`. The answer grows with the
folder: with paths of about 95 characters it takes about 380 bytes per note as
text and 530 as JSON, so 600 notes answer with about 225 KB of text or 320 KB
of JSON. While the move runs, every other write to the project waits. As a
rough guide, a move of 600 notes took about 15 seconds in a project with a short
history and about 23 seconds in one with 770 commits and 900 notes; these are
whole-call times on one machine with no other load on the project (graph and
search indexing held back), not a guarantee for a busy host.

Nothing changes, and the refusal names every offender (in the JSON form in
`details.conflicts`), when a destination path is taken (by a note outside the
folder, by a note whose name differs only in case, Unicode form or the
permalink it would get, or by a file on disk); when the folder holds anything
that is not a published note (an attachment, `.DS_Store`, an editor's swap
file, a symlink) or a note that was changed on disk outside Basic Memory; when
the destination lies inside the source; when the folder has no notes; when the
project changed during the move (repeat it); or when the host sets
`update_permalinks_on_move`, because a folder move never changes a permalink
and would otherwise disagree with a single move. A folder may move up into its
parent (`a/b` to `a`), even when a moved note lands where another one was.

Links inside notes are not rewritten. A link by permalink (`[[folder/note]]`)
or by title still finds the moved note; a link by file path
(`[[/folder/note.md]]`, `[[../other.md]]`) names the old place and no longer
resolves.

A `destination` that starts with another project's name, without
`destination_project`, is refused rather than becoming a folder of that name.

## PARAMETERS

- **destination** (string, required) — New path, e.g. 'archive/plan.md'. A trailing '/' ('archive/') moves the note or directory into that folder under its current name.
- **project** (string, required) — Required project name, 'workspace/project', or project external id. Discover projects with bm_project(command='list'). Paths are relative to this project; a qualified path must agree with it.
- **identifier** (string | null, optional, default: None) — The note to move: a path, permalink, memory:// URL or exact title.
- **path** (string | null, optional, default: None) — The directory to move, with everything in it, e.g. 'drafts/2025'.
- **expected_checksum** (string | null, optional, default: None) — Move the note only while it is still this revision (from bm_read).
- **destination_project** (string | null, optional, default: None) — Move into this project instead: a project name, 'workspace/project', or project external id. `destination` is then a path in that project.
- **output_format** (optional, default: "text")

## EXAMPLES

```
bm_move(identifier="plans/release-checklist.md", destination="archive/", project="demo")
# → Moved plans/Release Checklist.md to archive/Release Checklist.md

bm_move(path="archive", destination="old", project="demo")
# → Moved directory archive to old (1 files)

# The same on a Git host: one commit, every note listed.
bm_move(path="drafts", destination="archive/", project="demo")
# → Moved directory drafts to archive/drafts (2 notes, one commit 3f2a9c1d0b7e)
#   - drafts/a.md -> archive/drafts/a.md (id 0b6e5d1c-…, permalink demo/drafts/a)
#   - drafts/b.md -> archive/drafts/b.md (id 5c1d7a2e-…, permalink demo/drafts/b)

bm_move(identifier="plans/launch.md", destination="inbox/", project="demo",
        destination_project="research")
# → Moved plans/launch.md (project demo) to inbox/launch.md (project research)
# On a Git host: ..., id kept (0b6e5d1c-…); permalink demo/plans/launch -> research/inbox/launch
```

## GOTCHAS

- A destination with no file extension is refused for a note; end it with `/`
  to move into a folder.
- On a Git host, a move into another project that answers `PARTIAL_FAILURE`
  with a `move_id` has left the source already. Do not write the note again:
  it finishes on its own (next start, or `POST /v2/git/moves/resume`), or, when
  `stuck`, by a person's choice; its id stays reserved meanwhile.
- A target project that waits for notes another project still holds takes the
  move anyway; the answer then says its index is pending. A source that waits
  refuses (`UNAVAILABLE`) until the owner it waits for has synced.
- Off a Git host a directory move is file by file. If some files fail, the
  error lists which moved and which did not; bm_list the destination before
  retrying.
- On a Git host, a folder that differs from the destination only in case
  (`archive` to `Archive`) is refused: move it twice, through a different name.
- On a Git host the files on disk follow the commit within moments. Each
  note's new file is written before its old file is removed, so a moved note
  is not missing on disk; for a moment it can be there twice, with the same
  permalink. One exception: when moves that disk has not caught up with yet
  make two notes trade paths (folder moves onto paths whose old files are
  still waiting to go), one of the two notes is briefly missing. Folders the
  move leaves empty are removed; a folder that still holds anything else (an
  attachment, a hidden folder) stays, and so does one the system refuses to
  remove (for example while it is open in another program).
- A note moved away keeps its permalink, so a new note written later at the old
  path gets a permalink with a suffix (`drafts/a-1`).
- If a folder move gets no answer, do not repeat it blindly: a repeat that
  finds the folder empty is refused as not found. bm_list the destination.
- On a Git host a note edited on disk outside Basic Memory blocks the folder
  move, and saving the editor again does not help: Basic Memory does not take
  in disk edits there. Undo the edit, or make it with bm_edit or bm_write.
- If the server adopts an existing spelling of the folder (`Drafts` for
  `drafts`), the text form says so in `requested source:` and
  `requested destination:` lines, the JSON form in `requested_path` and
  `requested_destination`.

## SEE ALSO

- see_also [[bm-delete(3)]]
- see_also [[bm-write(3)]]
- see_also [[bm-list(3)]]
