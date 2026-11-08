"""BM-009: the three offline project commands parse their arguments and report refusals."""

from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from basic_memory.cli.main import app
from basic_memory.services.git_peer_setup import PeerChange
from basic_memory.services.git_project_create import CreatedGitProject, GitProjectCreateError
from basic_memory.services.git_project_remove import RemovedGitProject


def test_create_project_passes_names_paths_and_directories(tmp_path, monkeypatch):
    calls = []

    async def created(requests, *, base_dir, repositories_dir):
        calls.append((list(requests), base_dir, repositories_dir))
        return [
            CreatedGitProject(r.name, uuid4(), tmp_path / r.name, tmp_path / f"{r.name}.git")
            for r in requests
        ]

    monkeypatch.setattr("basic_memory.services.git_project_create.create_git_projects", created)
    result = CliRunner().invoke(
        app, ["git", "create-project", "haus", f"ake={tmp_path}/ake", "--root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    requests, base_dir, repositories_dir = calls[0]
    assert [(r.name, r.path) for r in requests] == [("haus", None), ("ake", tmp_path / "ake")]
    assert (base_dir, repositories_dir) == (tmp_path, tmp_path / "repositories")
    assert "haus:" in result.output and "ake:" in result.output


def test_create_project_reports_a_refusal(monkeypatch):
    async def refused(requests, *, base_dir, repositories_dir):
        raise GitProjectCreateError("Project name 'haus' is already taken")

    monkeypatch.setattr("basic_memory.services.git_project_create.create_git_projects", refused)
    result = CliRunner().invoke(app, ["git", "create-project", "haus"])
    assert result.exit_code == 1
    assert "Refused: Project name 'haus' is already taken" in result.output


def test_peer_setup_sets_and_removes(monkeypatch):
    calls = []

    async def changed(names, *, peer_base, merge, manager=None):
        calls.append((list(names), peer_base, merge))
        url = None if peer_base is None else f"{peer_base}/{names[0]}.git"
        return [PeerChange(names[0], uuid4(), url, merge)]

    monkeypatch.setattr("basic_memory.services.git_peer_setup.setup_git_peers", changed)
    result = CliRunner().invoke(
        app, ["git", "peer-setup", "haus", "--peer", "ssh://nuc1/peer", "--merge", "auto"]
    )
    assert result.exit_code == 0, result.output
    assert "haus: ssh://nuc1/peer/haus.git merge=auto" in result.output
    result = CliRunner().invoke(app, ["git", "peer-setup", "haus", "--remove"])
    assert result.exit_code == 0, result.output
    assert "haus: no peer" in result.output
    assert calls == [(["haus"], "ssh://nuc1/peer", "auto"), (["haus"], None, "off")]


def test_peer_setup_needs_exactly_one_of_peer_and_remove_and_a_known_merge_mode():
    for arguments in (
        ["haus"],
        ["haus", "--peer", "/p", "--remove"],
        ["haus", "--peer", "/p", "--merge", "sometimes"],
    ):
        result = CliRunner().invoke(app, ["git", "peer-setup", *arguments])
        assert result.exit_code == 1, arguments
        assert "Refused" in result.output


def test_peer_setup_reports_a_refusal(monkeypatch):
    async def refused(names, *, peer_base, merge, manager=None):
        raise ValueError("Unknown project(s): haus")

    monkeypatch.setattr("basic_memory.services.git_peer_setup.setup_git_peers", refused)
    result = CliRunner().invoke(app, ["git", "peer-setup", "haus", "--peer", "/p"])
    assert result.exit_code == 1
    assert "Refused: Unknown project(s): haus" in result.output


def test_remove_project_reports_removals_peers_and_leftovers(tmp_path, monkeypatch):
    async def removed(names):
        assert list(names) == ["haus", "ake"]
        return [
            RemovedGitProject("haus", uuid4(), tmp_path / "haus", tmp_path / "h.git", True),
            RemovedGitProject(
                "ake",
                uuid4(),
                tmp_path / "ake",
                tmp_path / "a.git",
                False,
                (Path(tmp_path / "ake"),),
            ),
        ]

    monkeypatch.setattr(
        "basic_memory.services.git_project_remove.remove_empty_git_projects", removed
    )
    result = CliRunner().invoke(app, ["git", "remove-project", "haus", "ake"])
    assert result.exit_code == 0, result.output
    assert "haus: removed" in result.output and "ake: removed" in result.output
    assert "peer repository was not touched" in result.output
    assert f"Could not delete {tmp_path / 'ake'}" in result.output


def test_remove_project_reports_a_refusal(monkeypatch):
    async def refused(names):
        raise GitProjectCreateError("Project(s) with notes are not removed: haus")

    monkeypatch.setattr(
        "basic_memory.services.git_project_remove.remove_empty_git_projects", refused
    )
    result = CliRunner().invoke(app, ["git", "remove-project", "haus"])
    assert result.exit_code == 1
    assert "Refused: Project(s) with notes are not removed: haus" in result.output


def test_create_and_remove_report_a_start_rule_refusal_instead_of_a_traceback(monkeypatch):
    from basic_memory.repository.git_projection_repository import GitProjectionConflict

    async def conflict(*args, **kwargs):
        raise GitProjectionConflict("Every configured and registered project must be Git-managed")

    monkeypatch.setattr("basic_memory.services.git_project_create.create_git_projects", conflict)
    monkeypatch.setattr(
        "basic_memory.services.git_project_remove.remove_empty_git_projects", conflict
    )
    for arguments in (["create-project", "haus"], ["remove-project", "haus"]):
        result = CliRunner().invoke(app, ["git", *arguments])
        assert result.exit_code == 1, arguments
        assert (
            "Refused: Every configured and registered project must be Git-managed" in result.output
        )
