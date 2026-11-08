"""BM-009: a project may only be removed while it is still exactly what create-project made."""

from pathlib import Path
from uuid import uuid4

import pytest

from basic_memory.repository.git_project_repository import (
    GitProjectRepository,
    settle_git_publication,
)
from basic_memory.services.git_project_create import GitProjectCreateError
from basic_memory.services.git_project_remove import (
    RemovedGitProject,
    _remove_directories,
    _require_empty_root,
    _require_unused_repository,
)


def test_a_missing_root_or_one_with_only_the_control_directory_is_empty(tmp_path):
    _require_empty_root("haus", tmp_path / "missing")
    root = tmp_path / "haus"
    (root / ".basic-memory").mkdir(parents=True)
    (root / ".basic-memory" / "git-binding.json").write_text("{}")
    _require_empty_root("haus", root)


def test_a_root_with_anything_else_is_refused(tmp_path):
    root = tmp_path / "haus"
    root.mkdir()
    (root / "notiz.md").write_text("x")
    with pytest.raises(GitProjectCreateError, match="holds files"):
        _require_empty_root("haus", root)


def test_links_and_plain_files_are_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(GitProjectCreateError, match="symbolic link"):
        _require_empty_root("haus", link)
    plain = tmp_path / "plain"
    plain.write_text("x")
    with pytest.raises(GitProjectCreateError, match="not a directory"):
        _require_empty_root("haus", plain)
    root = tmp_path / "root"
    root.mkdir()
    (root / ".basic-memory").symlink_to(target)
    with pytest.raises(GitProjectCreateError, match="Control directory"):
        _require_empty_root("haus", root)


@pytest.mark.asyncio
async def test_a_repository_with_history_is_not_unused(tmp_path):
    repository = GitProjectRepository.create(tmp_path / "haus.git")
    _require_unused_repository("haus", repository.path)  # only the initialization commit
    snapshot = repository.snapshot()
    prepared = repository.prepare(
        transaction_id=uuid4(),
        base_commit=snapshot.commit_id,
        files={"a.md": "# a\n"},
        note_ids={"a.md": uuid4()},
        message="note",
    )
    await settle_git_publication(repository, prepared)
    with pytest.raises(GitProjectCreateError, match="has been used"):
        _require_unused_repository("haus", repository.path)


def test_a_directory_that_cannot_be_removed_is_reported(tmp_path, monkeypatch):
    root, repository = tmp_path / "haus", tmp_path / "haus.git"
    root.mkdir()
    repository.mkdir()
    item = RemovedGitProject("haus", uuid4(), root, repository, had_peer=False)

    def refuse(path: Path) -> None:
        if path == repository:
            raise PermissionError(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr("basic_memory.services.git_project_remove.shutil.rmtree", refuse)
    assert _remove_directories(item).leftovers == (repository,)
