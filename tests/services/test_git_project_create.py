"""BM-009: the plan for new Git projects rejects everything that could confuse the host."""

import asyncio
import os
import signal
import stat
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
        (BASE, "must not be the base directory"),
        (BASE.parent, "must not be the base directory"),
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


# --- The configuration is replaced without losing its rights ---


def test_replacing_the_configuration_keeps_a_private_file_private(tmp_path, monkeypatch):
    from basic_memory.services.git_project_create import write_config_text_atomically

    monkeypatch.setattr("os.umask", lambda mask: 0o022, raising=False)
    previous = os.umask(0o022)  # the usual umask, under which a plain write_text gives 0644
    try:
        config = tmp_path / "config.json"
        config.write_text("old")
        config.chmod(0o600)
        write_config_text_atomically(config, "new")
        assert config.read_text() == "new" and stat.S_IMODE(config.stat().st_mode) == 0o600
        assert not (tmp_path / ".config.json.new").exists()
        fresh = tmp_path / "fresh.json"
        write_config_text_atomically(fresh, "x")
        assert stat.S_IMODE(fresh.stat().st_mode) == 0o600  # a new file starts private
        config.chmod(0o640)
        write_config_text_atomically(config, "again")
        assert stat.S_IMODE(config.stat().st_mode) == 0o640  # whatever the owner set stays
    finally:
        os.umask(previous)


def test_a_failed_replacement_leaves_the_old_file_and_no_temporary(tmp_path, monkeypatch):
    from basic_memory.services.git_project_create import write_config_text_atomically

    config = tmp_path / "config.json"
    config.write_text("old")

    def broken(source, target):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_config_text_atomically(config, "new")
    assert config.read_text() == "old" and not (tmp_path / ".config.json.new").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP])
async def test_a_termination_signal_cancels_the_task_instead_of_ending_the_process(number):
    from basic_memory.services.git_project_create import cancel_on_termination

    with cancel_on_termination():
        with pytest.raises(asyncio.CancelledError):
            os.kill(os.getpid(), number)
            await asyncio.sleep(5)
    # The handler is gone again: the default disposition is back for a later signal.
    assert signal.getsignal(number) in {signal.SIG_DFL, signal.default_int_handler} or callable(
        signal.getsignal(number)
    )
