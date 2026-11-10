"""Folders the files stage's own deletions left empty go, nothing else (F2, C8)."""

import errno
import os
from pathlib import Path

from basic_memory.index.git_projection import (
    clear_empty_folder,
    drop_second_name,
    one_regular_file,
    prune_empty_folders,
)


def test_empty_folders_go_deepest_first_up_to_the_first_that_holds_something(tmp_path: Path):
    (tmp_path / "a/b/c").mkdir(parents=True)
    (tmp_path / "a/keep.md").write_text("x")
    prune_empty_folders(tmp_path, "a/b/c/gone.md")
    assert not (tmp_path / "a/b").exists()
    assert (tmp_path / "a/keep.md").exists()


def test_the_root_and_hidden_folders_stay(tmp_path: Path):
    prune_empty_folders(tmp_path, "gone.md")
    assert tmp_path.is_dir()
    (tmp_path / ".hidden/inner").mkdir(parents=True)
    prune_empty_folders(tmp_path, ".hidden/inner/gone.md")
    # The visible empty folder inside goes; the hidden one stays, and so does the root.
    assert not (tmp_path / ".hidden/inner").exists()
    assert (tmp_path / ".hidden").is_dir()


def test_a_folder_that_is_already_gone_stops_the_walk(tmp_path: Path):
    prune_empty_folders(tmp_path, "never/there/gone.md")
    assert list(tmp_path.iterdir()) == []


def test_a_folder_the_system_refuses_to_remove_stays_and_the_pass_goes_on(
    tmp_path: Path, monkeypatch
):
    """The file is already gone; pruning is best effort and never aborts the pass."""
    (tmp_path / "a/b").mkdir(parents=True)

    def refused(self):
        raise PermissionError(errno.EACCES, "in use", str(self))

    monkeypatch.setattr(Path, "rmdir", refused)
    prune_empty_folders(tmp_path, "a/b/gone.md")
    assert (tmp_path / "a/b").is_dir()


def test_a_symlinked_folder_is_never_followed_out_of_the_project(tmp_path: Path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    root.mkdir()
    (root / "linked").symlink_to(outside)
    prune_empty_folders(root, "linked/sub/gone.md")
    assert (outside / "sub").is_dir() and (root / "linked").is_symlink()


# --- A folder where a file must go (C8) ---


def test_an_empty_folder_tree_where_a_file_must_go_is_removed(tmp_path: Path):
    (tmp_path / "a.md/b/c").mkdir(parents=True)
    (tmp_path / "a.md/d").mkdir()
    assert clear_empty_folder(tmp_path, "a.md")
    assert not (tmp_path / "a.md").exists()


def test_a_folder_holding_a_file_a_hidden_folder_or_a_link_stays(tmp_path: Path):
    (tmp_path / "file.md/sub").mkdir(parents=True)
    (tmp_path / "file.md/sub/x.png").write_bytes(b"png")
    (tmp_path / "hidden.md/.cache/inner").mkdir(parents=True)
    (tmp_path / "link.md").mkdir()
    (tmp_path / "link.md/to").symlink_to(tmp_path / "file.md")
    for path in ("file.md", "hidden.md", "link.md"):
        assert not clear_empty_folder(tmp_path, path)
    assert (tmp_path / "file.md/sub/x.png").exists()
    # Nothing inside a hidden folder is touched, not even an empty folder.
    assert (tmp_path / "hidden.md/.cache/inner").is_dir()
    assert (tmp_path / "link.md/to").is_symlink()


def test_a_symlink_to_a_folder_where_a_file_must_go_is_never_entered(tmp_path: Path):
    (tmp_path / "target/empty").mkdir(parents=True)
    (tmp_path / "a.md").symlink_to(tmp_path / "target")
    assert not clear_empty_folder(tmp_path, "a.md")
    assert (tmp_path / "target/empty").is_dir()


# --- Two names, one file (C1) ---


def test_one_file_means_one_inode_of_a_regular_file(tmp_path: Path):
    (tmp_path / "Foo.md").write_text("x")
    os.link(tmp_path / "Foo.md", tmp_path / "foo.md")
    (tmp_path / "copy.md").write_text("x")
    (tmp_path / "link.md").symlink_to("Foo.md")
    assert one_regular_file(tmp_path, "Foo.md", "foo.md")
    assert not one_regular_file(tmp_path, "Foo.md", "copy.md")
    # A symlink reads the same bytes, but it is a separate entry Basic Memory never wrote.
    assert not one_regular_file(tmp_path, "link.md", "Foo.md")
    assert not one_regular_file(tmp_path, "missing.md", "Foo.md")


def test_a_second_name_of_the_live_file_goes_only_when_it_is_its_own_entry(tmp_path: Path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/source.md").write_text("x")
    os.link(tmp_path / "notes/source.md", tmp_path / "notes/Source.md")
    drop_second_name(tmp_path, "notes/Source.md", "notes/source.md")
    assert sorted(os.listdir(tmp_path / "notes")) == ["source.md"]
    # Only one name is listed (as on macOS after the rename): nothing is removed.
    drop_second_name(tmp_path, "notes/Source.md", "notes/source.md")
    assert (tmp_path / "notes/source.md").read_text() == "x"


def test_the_same_name_through_an_aliased_folder_is_one_entry(tmp_path: Path):
    """`Notes/a.md` and `notes/a.md` on macOS: same folder, same name, one entry."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/a.md").write_text("x")
    (tmp_path / "Notes").symlink_to("notes")
    drop_second_name(tmp_path, "Notes/a.md", "notes/a.md")
    assert (tmp_path / "notes/a.md").read_text() == "x"
