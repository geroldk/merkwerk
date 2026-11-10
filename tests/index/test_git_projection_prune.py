"""Folders the files stage's own deletions left empty go, nothing else (F2, C8)."""

import errno
from pathlib import Path

import pytest

from basic_memory.index.git_projection import prune_empty_folders


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


def test_an_unexpected_error_is_not_swallowed(tmp_path: Path, monkeypatch):
    (tmp_path / "a").mkdir()

    def refused(self):
        raise PermissionError(errno.EACCES, "refused", str(self))

    monkeypatch.setattr(Path, "rmdir", refused)
    with pytest.raises(PermissionError):
        prune_empty_folders(tmp_path, "a/gone.md")
