"""The scheduled sync reports a person-needed problem once, tolerates offline, and clears."""

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "git_sync_watch.py"
SPEC = importlib.util.spec_from_file_location("git_sync_watch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
watch = importlib.util.module_from_spec(SPEC)
sys.modules["git_sync_watch"] = watch
SPEC.loader.exec_module(watch)


def completed(outcome: str) -> dict[str, Any]:
    return {"last_attempt": {"kind": "completed", "outcome": outcome}}


def failed(reason: str) -> dict[str, Any]:
    return {"last_attempt": {"kind": "failed", "reason": reason}}


class Peer:
    """Scripted answers per project; an Exception instance is raised instead."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    async def __call__(self, project: str) -> dict[str, Any]:
        answer = self.answers[project]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.mark.parametrize("outcome", ["equal", "pushed", "pulled", "merged"])
async def test_healthy_outcomes_say_nothing(outcome: str) -> None:
    state: dict[str, Any] = {}
    messages, attention = await watch.run_once(["a"], Peer({"a": completed(outcome)}), state)
    assert messages == [] and attention == 0 and "last_ok" in state["a"]


@pytest.mark.parametrize("outcome", ["diverged", "remote_rewritten"])
async def test_divergence_is_reported_at_once_and_only_once(outcome: str) -> None:
    state: dict[str, Any] = {}
    peer = Peer({"a": completed(outcome)})
    messages, attention = await watch.run_once(["a"], peer, state)
    assert attention == 1 and outcome in messages[0]
    again, attention = await watch.run_once(["a"], peer, state)
    assert again == [] and attention == 1


async def test_needs_review_names_kinds_and_paths_but_no_contents() -> None:
    answer = completed("needs_review") | {
        "open_review": {
            "id": "aaaaaaaaaaaa-bbbbbbbbbbbb",
            "conflicts": {"both_changed": 2},
            "paths": ["b.md", "x/a.md"],
        }
    }
    state: dict[str, Any] = {}
    messages, attention = await watch.run_once(["a"], Peer({"a": answer}), state)
    assert attention == 1
    assert "needs_review" in messages[0] and "both_changed=2" in messages[0]
    assert "aaaaaaaaaaaa-bbbbbbbbbbbb" in messages[0] and "b.md, x/a.md" in messages[0]
    cleared, attention = await watch.run_once(["a"], Peer({"a": completed("merged")}), state)
    assert attention == 0 and "wieder in Ordnung" in cleared[0]


@pytest.mark.parametrize("reason", ["invalid_incoming", "local_changed"])
async def test_validation_failures_are_reported_at_once(reason: str) -> None:
    messages, attention = await watch.run_once(["a"], Peer({"a": failed(reason)}), {})
    assert attention == 1 and reason in messages[0]


async def test_offline_is_tolerated_until_the_third_failure() -> None:
    state: dict[str, Any] = {}
    peer = Peer({"a": failed("transport")})
    for _ in range(2):
        messages, attention = await watch.run_once(["a"], peer, state)
        assert messages == [] and attention == 0
    messages, attention = await watch.run_once(["a"], peer, state)
    assert attention == 1 and "unreachable:transport" in messages[0]


async def test_a_dead_service_counts_as_transport_failure() -> None:
    state: dict[str, Any] = {}
    peer = Peer({"a": ConnectionError("down")})
    for _ in range(3):
        messages, attention = await watch.run_once(["a"], peer, state)
    assert attention == 1 and "unreachable" in messages[0]


async def test_recovery_is_announced_once() -> None:
    state: dict[str, Any] = {}
    await watch.run_once(["a"], Peer({"a": completed("diverged")}), state)
    messages, attention = await watch.run_once(["a"], Peer({"a": completed("pulled")}), state)
    assert attention == 0 and "wieder in Ordnung" in messages[0]
    messages, _ = await watch.run_once(["a"], Peer({"a": completed("pulled")}), state)
    assert messages == []


async def test_a_success_resets_the_failure_count() -> None:
    state: dict[str, Any] = {}
    await watch.run_once(["a"], Peer({"a": failed("transport")}), state)
    await watch.run_once(["a"], Peer({"a": completed("equal")}), state)
    assert state["a"]["failures"] == 0


async def test_projects_without_a_peer_and_busy_ones_are_skipped() -> None:
    state: dict[str, Any] = {}
    peer = Peer({"a": watch.NotConfigured(), "b": watch.Busy(), "c": completed("equal")})
    messages, attention = await watch.run_once(["a", "b", "c"], peer, state)
    assert messages == [] and attention == 0 and "a" not in state


async def test_status_relationship_diverged_without_an_attempt_is_reported() -> None:
    messages, attention = await watch.run_once(
        ["a"], Peer({"a": {"relationship": "diverged", "last_attempt": None}}), {}
    )
    assert attention == 1 and "diverged" in messages[0]


async def test_an_unknown_answer_shape_is_not_an_alarm() -> None:
    messages, attention = await watch.run_once(["a"], Peer({"a": {"last_attempt": None}}), {})
    assert messages == [] and attention == 0


def test_state_survives_a_restart_and_a_corrupt_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "state.json"
    watch.save_state(path, {"a": {"problem": "diverged", "failures": 0}})
    assert watch.load_state(path)["a"]["problem"] == "diverged"
    path.write_text("not json")
    assert watch.load_state(path) == {}
    path.write_text("[1]")
    assert watch.load_state(path) == {}
    assert watch.load_state(tmp_path / "missing.json") == {}


def test_notify_pipes_the_report_to_the_command(tmp_path: Path, capsys) -> None:
    out = tmp_path / "got.txt"
    watch.notify(f"cat > {out}", ["one", "two"])
    assert out.read_text() == "one\ntwo"
    assert "one" in capsys.readouterr().err
    watch.notify(f"cat > {out}", [])  # nothing to say: the command is not run
    assert out.read_text() == "one\ntwo"


async def test_an_offline_hiccup_keeps_an_open_review_open() -> None:
    state: dict[str, Any] = {}
    messages, _ = await watch.run_once(["a"], Peer({"a": completed("needs_review")}), state)
    assert "needs_review" in messages[0]
    messages, attention = await watch.run_once(["a"], Peer({"a": failed("transport")}), state)
    assert messages == [] and attention == 1 and state["a"]["problem"] == "needs_review"
    # Back online with the review still open: no repeated report either.
    messages, attention = await watch.run_once(["a"], Peer({"a": completed("needs_review")}), state)
    assert messages == [] and attention == 1


@pytest.mark.parametrize(
    ("text", "busy"),
    [
        ("409 Conflict", True),
        ("sync is busy", True),
        ("A sync attempt is already running for this project", True),
        ("connection refused", False),
    ],
)
def test_the_services_busy_wording_is_recognized(text: str, busy: bool) -> None:
    assert watch.is_busy(text) is busy


def test_the_notify_command_runs_for_news_and_with_always_also_for_none(tmp_path, capsys) -> None:
    seen = tmp_path / "seen"
    command = f"cat > {seen}; echo ran >> {tmp_path}/runs"

    watch.notify(command, [])  # nothing new, no `always`: the command stays quiet
    assert not (tmp_path / "runs").exists()

    watch.notify(command, ["[BM sync] main: needs_review (x)"])
    assert seen.read_text() == "[BM sync] main: needs_review (x)"
    assert capsys.readouterr().err == "[BM sync] main: needs_review (x)\n"

    watch.notify(command, [], always=True)  # a queue-keeping command gets an empty pass
    assert seen.read_text() == ""
    assert (tmp_path / "runs").read_text() == "ran\nran\n"
    assert capsys.readouterr().err == ""

    watch.notify(None, ["[BM sync] pkm: diverged (y)"], always=True)  # no command, nothing runs
    assert (tmp_path / "runs").read_text() == "ran\nran\n"


@pytest.mark.asyncio
async def test_a_service_that_does_not_answer_is_reported_once_per_project_after_the_limit() -> (
    None
):
    state: dict[str, Any] = {}
    names = ["main", "pkm"]
    for expected in (0, 0, 2):
        messages, attention = await watch.run_once(names, watch.service_down, state, tolerate=3)
        assert len(messages) == expected
    assert "[BM sync] main: unreachable:transport (3 failures in a row)" in messages
    assert attention == 2
    again, _ = await watch.run_once(names, watch.service_down, state, tolerate=3)
    assert again == []  # reported once, not on every pass


def test_the_projects_to_visit_fall_back_to_what_was_seen_before() -> None:
    state = {"main": {"problem": None, "failures": 1}, "other": "not a project entry"}
    assert watch.projects_to_visit(("a", "b"), ["x"], state) == ["a", "b"]  # configured wins
    assert watch.projects_to_visit(None, ["x", "y"], state) == ["x", "y"]  # what the service lists
    assert watch.projects_to_visit(None, None, state) == ["main"]  # service down: the history
    assert watch.projects_to_visit(None, None, {}) == [watch.SERVICE]  # nothing known at all


def test_entries_of_names_that_are_not_visited_any_more_are_retired() -> None:
    state: dict[str, Any] = {
        "main": {"problem": None, "failures": 0},
        watch.SERVICE: {"problem": "unreachable:transport", "failures": 3},
        "gone": {"problem": None, "failures": 0},
        "junk": "not an entry",
    }
    messages = watch.retire_unvisited(state, ["main"])
    assert messages == [f"[BM sync] {watch.SERVICE}: wieder in Ordnung (wird nicht mehr geprüft)"]
    assert set(state) == {"main", "junk"}
    assert watch.retire_unvisited(state, ["main"]) == []
