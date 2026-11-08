"""BM-009: where a local peer may live, and what a failed creation leaves behind."""

import subprocess
from pathlib import Path

import pytest

from basic_memory.services import git_peer_setup
from basic_memory.services.git_peer_setup import (
    _create_local_peer,
    _require_peer_apart_from_projects,
)
from basic_memory.services.git_project_create import GitProjectCreateError

BASE = Path("/base/bm")
ROOTS = [BASE / "main", BASE / "pkm"]
REPOSITORIES = [BASE / "repositories" / "main.git", BASE / "repositories" / "pkm.git"]


@pytest.mark.parametrize(
    "peer",
    [
        BASE / "repositories" / "main.git",  # a project's own repository: its own peer
        BASE / "repositories",  # the directory over the repositories
        BASE / "main" / "peer" / "main.git",  # inside a note root
        BASE,  # over everything
    ],
)
def test_a_peer_may_not_be_a_project_place(peer):
    with pytest.raises(GitProjectCreateError, match="overlaps"):
        _require_peer_apart_from_projects(str(peer), ROOTS, REPOSITORIES, "main")


def test_a_peer_of_its_own_is_fine():
    _require_peer_apart_from_projects(str(BASE / "peer" / "main.git"), ROOTS, REPOSITORIES, "main")
    _require_peer_apart_from_projects("/mnt/other/peer/main.git", ROOTS, REPOSITORIES, "main")


def test_a_failed_git_init_removes_what_the_call_made(tmp_path, monkeypatch):
    target = tmp_path / "new" / "deeper" / "main.git"

    def broken(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"git init failed")

    monkeypatch.setattr(git_peer_setup.subprocess, "run", broken)
    with pytest.raises(GitProjectCreateError, match="git init failed"):
        _create_local_peer(str(target))
    assert not (tmp_path / "new").exists()  # the parents this call made are gone too
    assert tmp_path.exists()


def test_an_existing_peer_is_left_alone(tmp_path):
    existing = tmp_path / "peer" / "main.git"
    existing.mkdir(parents=True)
    assert _create_local_peer(str(existing)) is None
    assert existing.exists()


def test_creating_a_peer_reports_the_topmost_directory_it_made(tmp_path):
    (tmp_path / "peer").mkdir()
    made = _create_local_peer(str(tmp_path / "peer" / "side" / "deeper" / "main.git"))
    assert made == tmp_path / "peer" / "side"
    assert (tmp_path / "peer" / "side" / "deeper" / "main.git" / "HEAD").exists()


def test_a_timeout_of_git_init_also_removes_what_the_call_made(tmp_path, monkeypatch):
    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired("git init", 30)

    monkeypatch.setattr(git_peer_setup.subprocess, "run", hangs)
    with pytest.raises(subprocess.TimeoutExpired):
        _create_local_peer(str(tmp_path / "new" / "main.git"))
    assert not (tmp_path / "new").exists()
