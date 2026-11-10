"""The folder-move planner and its disk checks (V1): pure inputs, every offender named."""

import hashlib
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from basic_memory.services.git_directory_moves import (
    GitDirectoryMoveConflict,
    GitDirectoryMoveDisk,
    GitDirectoryMovePlan,
    GitDirectoryMoveRefused,
    GitPlannedNoteMove,
    disk_conflicts,
    normalize_note_directory,
    note_directories,
    plan_git_directory_move,
    unpublished_source_files,
)


def note(path: str) -> str:
    return f"---\ntitle: {path}\ntype: note\npermalink: p/{path.removesuffix('.md')}\n---\n\nBody\n"


def tree(*paths: str) -> tuple[dict[str, str], dict[str, UUID]]:
    return {path: note(path) for path in paths}, {path: uuid4() for path in paths}


def refused(files, ids, source: str, destination: str) -> GitDirectoryMoveRefused:
    with pytest.raises(GitDirectoryMoveRefused) as error:
        plan_git_directory_move(files, ids, source, destination)
    return error.value


# --- Plan ---


def test_every_note_under_the_folder_moves_with_its_id_and_permalink():
    files, ids = tree("drafts/a.md", "drafts/sub/b.md", "drafts-other/c.md", "keep/d.md")

    plan = plan_git_directory_move(files, ids, "/drafts/", "archive/drafts")

    assert plan == GitDirectoryMovePlan(
        "drafts",
        "archive/drafts",
        (
            GitPlannedNoteMove(
                "drafts/a.md", "archive/drafts/a.md", ids["drafts/a.md"], "p/drafts/a"
            ),
            GitPlannedNoteMove(
                "drafts/sub/b.md",
                "archive/drafts/sub/b.md",
                ids["drafts/sub/b.md"],
                "p/drafts/sub/b",
            ),
        ),
    )
    assert plan.message == "Move drafts/ to archive/drafts/ (2 notes)"


def test_a_folder_known_only_through_notes_deeper_down_moves():
    files, ids = tree("a/b/c/x.md")

    plan = plan_git_directory_move(files, ids, "a", "z")

    assert [move.file_path for move in plan.moves] == ["z/b/c/x.md"]
    assert note_directories(files) == {"a", "a/b", "a/b/c"}


@pytest.mark.parametrize(
    "source,destination,reason",
    [
        ("", "x", "project_root"),
        ("/", "x", "project_root"),
        ("drafts", "/", "project_root"),
        ("drafts", " archive", "invalid_path"),
        ("drafts ", "archive", "invalid_path"),
        ("drafts", "a/../b", "invalid_path"),
        ("drafts", ".git/x", "invalid_path"),
        ("drafts", "a/.basic-memory", "invalid_path"),
        ("drafts", "a:b", "invalid_path"),
        ("drafts", "a\\b", "invalid_path"),
        ("drafts", "a//b", "invalid_path"),
        ("drafts", "a\nb", "invalid_path"),
        ("missing", "archive", "not_found"),
        ("drafts", "drafts", "same_path"),
        ("drafts", "Drafts", "case_only"),
        ("drafts", "drafts/inner", "inside_source"),
        ("drafts", "DRAFTS/inner", "inside_source"),
    ],
)
def test_refusals_change_nothing_and_say_why(source, destination, reason):
    files, ids = tree("drafts/a.md")

    error = refused(files, ids, source, destination)

    assert error.reason == reason
    assert error.status_code == {"not_found": 404}.get(reason, 422)


def test_whitespace_is_refused_rather_than_stripped():
    files, ids = tree("drafts/a.md", " drafts/b.md")

    assert refused(files, ids, "drafts", "archive ").reason == "invalid_path"
    # ' drafts' is a different folder; it is addressed only by its exact name.
    assert refused(files, ids, " drafts", "archive").reason == "invalid_path"


def test_the_source_takes_the_spelling_of_the_one_equivalent_folder():
    files, ids = tree("Drafts/a.md")

    plan = plan_git_directory_move(files, ids, "drafts", "archive")

    assert plan.source == "Drafts"
    assert plan.moves[0].file_path == "archive/a.md"


def test_two_case_variant_folders_are_ambiguous_instead_of_merged():
    files, ids = tree("Drafts/a.md", "drafts/b.md")

    error = refused(files, ids, "DRAFTS", "archive")

    assert (error.status_code, error.reason) == (409, "ambiguous_source")
    assert "Drafts, drafts" in error.message
    # The exact spelling still names one of them.
    assert plan_git_directory_move(files, ids, "drafts", "archive").moves[0].file_path == (
        "archive/b.md"
    )


def test_the_destination_takes_an_existing_folders_spelling():
    files, ids = tree("drafts/a.md", "Archive/seed.md")

    plan = plan_git_directory_move(files, ids, "drafts", "archive/drafts")

    assert plan.destination == "Archive/drafts"


def test_the_destination_resolves_against_the_sources_own_parents():
    """'a' exists only through 'a/b/...'; 'A/c' must not become a case-variant sibling."""
    files, ids = tree("a/b/x.md")

    plan = plan_git_directory_move(files, ids, "a/b", "A/c")

    assert plan.destination == "a/c"


def test_a_case_only_rename_is_refused_by_the_portable_key():
    files, ids = tree("archive/a.md", "other/Café.md")

    error = refused(files, ids, "archive", "Archive")

    assert error.reason == "case_only" and "differ only in case" in error.message
    # NFC and NFD spellings of one name are the same folder too.
    files, ids = tree("café/a.md")
    assert refused(files, ids, "café", "Café").reason == "case_only"


def test_a_folder_may_move_up_into_its_parent_and_swap_paths():
    files, ids = tree("a/b/x.md", "a/b/b/x.md")

    plan = plan_git_directory_move(files, ids, "a/b", "a")

    assert {move.previous_file_path: move.file_path for move in plan.moves} == {
        "a/b/b/x.md": "a/b/x.md",
        "a/b/x.md": "a/x.md",
    }


def test_every_taken_target_is_named_in_one_refusal():
    files, ids = tree(
        "src/a.md",
        "src/B.md",
        "src/Foo Bar.md",
        "src/Straße.md",
        "src/x.md/y.md",
        "src/z.md",
        "dst/a.md",
        "dst/b.md",
        "dst/foo-bar.md",
        "dst/STRASSE.md",
        "dst/x.md",
        "dst/z.md/inner.md",
    )

    error = refused(files, ids, "src", "dst")

    assert (error.status_code, error.reason) == (409, "occupied")
    assert error.conflicts == (
        GitDirectoryMoveConflict("dst/B.md", "equivalent", "dst/b.md"),
        GitDirectoryMoveConflict("dst/Foo Bar.md", "equivalent", "dst/foo-bar.md"),
        GitDirectoryMoveConflict("dst/Straße.md", "equivalent", "dst/STRASSE.md"),
        GitDirectoryMoveConflict("dst/a.md", "occupied", "dst/a.md"),
        GitDirectoryMoveConflict("dst/x.md/y.md", "file_folder_clash", "dst/x.md"),
        GitDirectoryMoveConflict("dst/z.md", "file_folder_clash", "dst/z.md/inner.md"),
    )
    assert "6 destination path(s) are taken" in error.message
    assert "dst/a.md is already a note" in error.message


def test_a_note_outside_the_folder_occupies_a_target():
    files, ids = tree("src/a.md", "elsewhere/a.md", "dst/a.md")

    error = refused(files, ids, "src", "dst")

    assert error.conflicts == (GitDirectoryMoveConflict("dst/a.md", "occupied", "dst/a.md"),)


def test_two_notes_in_the_source_that_name_one_note_refuse_the_folder():
    files, ids = tree("src/X.md", "src/x.md")

    error = refused(files, ids, "src", "dst")

    assert error.conflicts == (GitDirectoryMoveConflict("dst/x.md", "equivalent", "dst/X.md"),)


def test_a_published_note_without_permalink_is_named():
    files, ids = tree("src/a.md")
    files["src/a.md"] = "---\ntitle: a\ntype: note\n---\n"

    error = refused(files, ids, "src", "dst")

    assert (error.reason, error.status_code) == ("invalid_draft", 422)
    assert "src/a.md" in error.message


def test_a_published_note_with_unreadable_frontmatter_is_named():
    files, ids = tree("src/a.md")
    files["src/a.md"] = "---\ntitle: [unclosed\n---\n"

    error = refused(files, ids, "src", "dst")

    assert error.reason == "invalid_draft" and "src/a.md" in error.message


def test_normalize_keeps_the_folder_as_given():
    assert normalize_note_directory("/a/b/", role="source") == "a/b"
    assert normalize_note_directory("a/~b", role="destination") == "a/~b"


# --- Disk ---


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def disk(tmp_path: Path, files: dict[str, str], retired=None) -> GitDirectoryMoveDisk:
    return GitDirectoryMoveDisk(
        tmp_path,
        {path: frozenset({checksum(text)}) for path, text in files.items()},
        {path: frozenset(values) for path, values in (retired or {}).items()},
    )


def write(root: Path, path: str, text: str) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


def test_clean_published_notes_and_pending_retirements_pass(tmp_path):
    files, ids = tree("src/a.md", "src/sub/b.md")
    for path, text in files.items():
        write(tmp_path, path, text)
    # A note moved out earlier whose old file BM has not deleted yet.
    write(tmp_path, "src/gone.md", "old bytes")
    # Another one at a target path.
    write(tmp_path, "dst/a.md", "older bytes")
    plan = plan_git_directory_move(files, ids, "src", "dst")

    unclean, taken = disk_conflicts(
        disk(
            tmp_path,
            files,
            {"src/gone.md": {checksum("old bytes")}, "dst/a.md": {checksum("older bytes")}},
        ),
        plan,
    )

    assert (unclean, taken) == ([], [])


def test_every_file_in_the_source_that_is_not_a_clean_note_is_named(tmp_path):
    files, ids = tree("src/a.md", "src/b.md")
    write(tmp_path, "src/a.md", files["src/a.md"])
    write(tmp_path, "src/b.md", files["src/b.md"] + "edited in Obsidian\n")
    write(tmp_path, "src/img.png", "png")
    write(tmp_path, "src/.DS_Store", "x")
    write(tmp_path, "src/deep/.b.md.swp", "swap")
    write(tmp_path, "src/stale.md", "not the retired bytes")
    os.symlink(tmp_path / "src/a.md", tmp_path / "src/link.md")
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", tmp_path / "src/linked-folder")
    plan = plan_git_directory_move(files, ids, "src", "dst")

    unclean, taken = disk_conflicts(
        disk(tmp_path, files, {"src/stale.md": {checksum("retired")}}), plan
    )

    assert taken == []
    assert unclean == [
        GitDirectoryMoveConflict("src/.DS_Store", "unpublished_file"),
        GitDirectoryMoveConflict("src/b.md", "edited_on_disk"),
        GitDirectoryMoveConflict("src/deep/.b.md.swp", "unpublished_file"),
        GitDirectoryMoveConflict("src/img.png", "unpublished_file"),
        GitDirectoryMoveConflict("src/link.md", "symlink"),
        GitDirectoryMoveConflict("src/linked-folder", "symlink"),
        GitDirectoryMoveConflict("src/stale.md", "unpublished_file"),
    ]
    assert unclean[1].describe() == "src/b.md was changed on disk outside Basic Memory"


def test_a_source_folder_that_is_a_symlink_or_under_a_file_is_refused(tmp_path):
    files, ids = tree("src/a.md", "top/src/a.md")
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "src")
    (tmp_path / "top").write_text("a file where a folder should be")

    assert unpublished_source_files(disk(tmp_path, files), "src") == [
        GitDirectoryMoveConflict("src", "symlink")
    ]
    assert unpublished_source_files(disk(tmp_path, files), "top/src") == [
        GitDirectoryMoveConflict("top/src", "file_folder_clash", "top")
    ]
    # A folder that is not on disk yet (files stage behind) is clean.
    assert unpublished_source_files(disk(tmp_path, files), "elsewhere") == []


def test_targets_blocked_on_disk_are_named(tmp_path):
    files, ids = tree("src/a.md", "src/b.md", "src/c.md", "src/d/e.md", "src/f/g.md")
    write(tmp_path, "dst/a.md", "an unindexed file")
    (tmp_path / "dst/b.md").mkdir(parents=True)
    os.symlink(tmp_path / "src", tmp_path / "dst/c.md")
    write(tmp_path, "dst/d", "a file named like the folder")
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "dst/f")
    plan = plan_git_directory_move(files, ids, "src", "dst")

    unclean, taken = disk_conflicts(disk(tmp_path, files), plan)

    assert unclean == []
    assert taken == [
        GitDirectoryMoveConflict("dst/a.md", "occupied"),
        GitDirectoryMoveConflict("dst/b.md", "file_folder_clash"),
        GitDirectoryMoveConflict("dst/c.md", "symlink"),
        GitDirectoryMoveConflict("dst/d/e.md", "file_folder_clash", "dst/d"),
        GitDirectoryMoveConflict("dst/f", "symlink"),
    ]
    assert [conflict.describe() for conflict in taken[:2]] == [
        "dst/a.md already exists on disk",
        "dst/b.md is a folder on disk",
    ]


def test_a_target_another_moving_note_leaves_is_not_occupied(tmp_path):
    files, ids = tree("a/b/x.md", "a/b/b/x.md")
    for path, text in files.items():
        write(tmp_path, path, text)
    plan = plan_git_directory_move(files, ids, "a/b", "a")

    assert disk_conflicts(disk(tmp_path, files), plan) == ([], [])


def test_a_file_that_vanishes_during_the_scan_counts_as_absent(tmp_path, monkeypatch):
    # The files stage runs beside the move: it stages each note as '.<name>.<hex>.tmp' and
    # renames it over the note, and it deletes retired files, so a listed file may be gone
    # by the time the scan looks at it.
    files, ids = tree("src/a.md", "src/b.md")
    for path, text in files.items():
        write(tmp_path, path, text)
    write(tmp_path, "src/.a.md.0123456789abcdef.tmp", files["src/a.md"])
    write(tmp_path, "src/gone-before-read.md", "retired bytes")
    write(tmp_path, "dst/b.md", "retired target")
    plan = plan_git_directory_move(files, ids, "src", "dst")
    real_walk = os.walk

    def walk_then_rename(top, *args, **kwargs):
        for current, folders, names in real_walk(top, *args, **kwargs):
            # Listed, then renamed over the note before the scan reaches it.
            if ".a.md.0123456789abcdef.tmp" in names:
                os.replace(Path(current) / ".a.md.0123456789abcdef.tmp", Path(current) / "a.md")
            yield current, folders, names

    module = "basic_memory.services.git_directory_moves"
    monkeypatch.setattr(f"{module}.os.walk", walk_then_rename)
    real_checksum = __import__(module, fromlist=["_file_checksum"])._file_checksum

    def checksum_after_delete(path: str) -> str:
        # The lstat saw the file; it is deleted before it is read.
        if path.endswith(("gone-before-read.md", "dst/b.md")):
            os.unlink(path)
        return real_checksum(path)

    monkeypatch.setattr(f"{module}._file_checksum", checksum_after_delete)

    assert disk_conflicts(disk(tmp_path, files), plan) == ([], [])


def test_conflicts_describe_themselves():
    assert GitDirectoryMoveConflict("d/x.md", "unpublished_file").describe() == (
        "d/x.md is not a published note"
    )
    assert GitDirectoryMoveConflict("d/x.md", "symlink").describe() == "d/x.md is a symlink"
    assert GitDirectoryMoveConflict("d/x.md", "equivalent", "d/X.md").describe() == (
        "d/x.md names the same note as d/X.md"
    )
    assert GitDirectoryMoveConflict("d/x/y.md", "file_folder_clash", "d/x").describe() == (
        "d/x/y.md clashes with d/x (a file where a folder is needed)"
    )
