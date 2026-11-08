"""BM-009: the plan for new Git projects rejects everything that could confuse the host."""

from pathlib import Path

import pytest

from basic_memory.services.git_project_create import (
    GitProjectCreateError,
    NewGitProject,
    plan_new_projects,
    require_host_stopped,
)

BASE = Path("/base/bm")


def plan(
    *requests: NewGitProject, existing_roots=(), names=("main", "PKM"), permalinks=("main", "pkm")
):
    return plan_new_projects(
        requests,
        base_dir=BASE,
        repositories_dir=BASE / "repositories",
        existing_names=names,
        existing_permalinks=permalinks,
        existing_roots=[BASE / "main", BASE / "PKM", *existing_roots],
    )


def test_defaults_put_the_root_below_the_base_and_the_repository_next_to_the_others():
    assert plan(NewGitProject("haus"), NewGitProject("meldungen-ryzen")) == [
        ("haus", BASE / "haus", BASE / "repositories" / "haus.git"),
        (
            "meldungen-ryzen",
            BASE / "meldungen-ryzen",
            BASE / "repositories" / "meldungen-ryzen.git",
        ),
    ]
    explicit = plan(NewGitProject("ake", BASE / "arbeit" / "ake"))
    assert explicit[0][1] == BASE / "arbeit" / "ake"


@pytest.mark.parametrize("name", ["", "a", "Haus", "1haus", "ha us", "ha_us", "x" * 33, "ha/us"])
def test_names_must_be_short_lower_case_words(name):
    with pytest.raises(GitProjectCreateError, match="must match"):
        plan(NewGitProject(name))


def test_names_differing_only_in_case_or_repeated_are_refused():
    with pytest.raises(GitProjectCreateError, match="already taken"):
        plan(NewGitProject("pkm"))  # PKM exists
    with pytest.raises(GitProjectCreateError, match="already taken"):
        plan(NewGitProject("main"))
    with pytest.raises(GitProjectCreateError, match="already taken"):
        plan(NewGitProject("haus"), NewGitProject("haus"))


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (BASE, "new directory below"),
        (BASE.parent, "new directory below"),
        (BASE / "main" / "sub", "overlaps"),
        (BASE / "PKM", "overlaps"),
        (BASE / "repositories" / "x", "reserved"),
        (BASE / "peer", "reserved"),
        (BASE / "operations" / "haus", "reserved"),
        (BASE / "code", "reserved"),
        (Path("relative/haus"), "absolute"),
    ],
)
def test_roots_may_not_overlap_projects_or_infrastructure(path, message):
    with pytest.raises(GitProjectCreateError, match=message):
        plan(NewGitProject("haus", path))


def test_an_existing_directory_must_be_empty_and_an_existing_repository_is_refused(tmp_path):
    base = tmp_path / "bm"
    (base / "used").mkdir(parents=True)
    (base / "used" / "note.md").write_text("x")
    (base / "empty").mkdir()
    (base / "repositories" / "taken.git").mkdir(parents=True)

    def go(name):
        return plan_new_projects(
            [NewGitProject(name)],
            base_dir=base,
            repositories_dir=base / "repositories",
            existing_names=[],
            existing_permalinks=[],
            existing_roots=[],
        )

    with pytest.raises(GitProjectCreateError, match="not an empty directory"):
        go("used")
    assert go("empty")[0][1] == (base / "empty").resolve()
    with pytest.raises(GitProjectCreateError, match="already in use"):
        go("taken")


def test_nothing_requested_is_an_error():
    with pytest.raises(GitProjectCreateError, match="No project requested"):
        plan()


def test_a_service_holding_the_locks_blocks_creation(tmp_path):
    from filelock import FileLock

    root, repository = tmp_path / "root", tmp_path / "repo.git"
    (root / ".basic-memory").mkdir(parents=True)
    repository.mkdir()
    require_host_stopped([(root, repository)])  # nothing running
    require_host_stopped([(tmp_path / "missing", tmp_path / "missing.git")])  # not bound yet
    with FileLock(root / ".basic-memory" / "git-projection.lock", thread_local=False):
        with pytest.raises(GitProjectCreateError, match="stop it first"):
            require_host_stopped([(root, repository)])
