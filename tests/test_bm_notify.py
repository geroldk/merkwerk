"""BM-009 (V5): every report line becomes one note; nothing is lost while the service is away."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bm_host_settings as hs  # noqa: E402
import bm_notify as bn  # noqa: E402

NOW = datetime(2026, 11, 8, 12, 51, 10, tzinfo=timezone(timedelta(hours=1)))
REPORT = "[BM sync] main: needs_review (outcome=needs_review review=r1 content=1 [a.md])"


def settings_for(tmp_path: Path, **notify: object) -> hs.HostSettings:
    return hs.parse_settings(
        {
            "host_name": "ryzen",
            "notify": {"project": "meldungen-ryzen", "spool": str(tmp_path / "spool"), **notify},
        }
    )


class Service:
    """A scripted `bm_write`: each call takes the next answer; an Exception is raised."""

    def __init__(self, *answers: Exception | None) -> None:
        self.answers = list(answers)
        self.notes: list[bn.NoteRequest] = []

    def __call__(self, settings: hs.HostSettings) -> "Service":
        return self

    async def __aenter__(self):
        return self.write

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def write(self, note: bn.NoteRequest) -> None:
        self.notes.append(note)
        answer = self.answers.pop(0) if self.answers else None
        if answer is not None:
            raise answer


class Away(Service):
    """The service cannot even be opened."""

    async def __aenter__(self):
        raise bn.Unreachable("connection refused")


def spooled(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "spool").glob("*.json"))


# --- Messages and notes ---


def test_a_report_becomes_one_message_per_line_without_control_characters():
    text = f"{REPORT}\n\n  second\x07 line  \r\n" + "x" * 5000
    lines = bn.split_report(text)
    assert lines[:2] == [REPORT, "second line"]
    assert len(lines) == 3 and len(lines[2]) == bn.MAX_LINE


def test_identical_lines_in_one_second_still_get_different_ids():
    first, second = bn.make_messages(["same", "same"], "ryzen", NOW)
    assert first.id != second.id and first.at == second.at
    assert bn.make_messages(["same", "same"], "ryzen", NOW)[0].id == first.id  # stable


def test_project_and_problem_are_read_from_the_watch_report():
    message = bn.make_messages([REPORT], "ryzen", NOW)[0]
    assert (message.project, message.problem) == ("main", "needs_review")
    other = bn.make_messages(["something else"], "ryzen", NOW)[0]
    assert (other.project, other.problem) == (None, None)


def test_the_note_names_host_time_project_and_id_and_is_safe_to_store(tmp_path):
    notify = settings_for(tmp_path).notify
    message = bn.make_messages([REPORT], "ryzen", NOW)[0]
    note = bn.note_for(message, notify)
    assert note.title == f"Abgleich ryzen {NOW.astimezone():%Y%m%d-%H%M%S} main {message.id[:6]}"
    assert note.directory == "abgleich" and note.note_type == "meldung"
    assert note.tags == ["abgleich", "ryzen"] and note.metadata == {"host": "ryzen"}
    assert note.content.startswith("Meldung: [BM sync] main: needs_review")
    for line in (
        "- [host] ryzen",
        "- [projekt] main",
        "- [problem] needs_review",
        "- [status] offen",
    ):
        assert line in note.content.splitlines()
    assert bn.note_for(message, notify) == note  # a replay builds the very same note


@pytest.mark.parametrize("line", ["---", "--- title: x", "[[Other Note]]", "x [[a]] y"])
def test_a_report_line_cannot_open_frontmatter_or_make_a_link(tmp_path, line):
    message = bn.make_messages([line], "ryzen", NOW)[0]
    content = bn.note_for(message, settings_for(tmp_path).notify).content
    assert content.startswith("Meldung: ")
    assert "[[" not in content and "]]" not in content


def test_a_line_that_is_not_a_watch_report_is_filed_under_meldung(tmp_path):
    message = bn.make_messages(["Backup fertig"], "nuc1", NOW)[0]
    note = bn.note_for(message, settings_for(tmp_path).notify)
    assert " meldung " in note.title and "- [projekt]" not in note.content


# --- The spool ---


def test_a_message_is_spooled_privately_and_listed_oldest_first(tmp_path):
    spool = tmp_path / "spool"
    later = bn.make_messages(["later"], "ryzen", NOW + timedelta(minutes=5))
    earlier = bn.make_messages(["earlier"], "ryzen", NOW)
    bn.enqueue(spool, later)
    bn.enqueue(spool, earlier)
    assert [entry["text"] for _, entry in bn.pending(spool)] == ["earlier", "later"]
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    for path, _ in bn.pending(spool):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(spool.glob(".*.tmp"))


def test_an_unreadable_spool_file_is_set_aside_not_retried_forever(tmp_path, capsys):
    spool = tmp_path / "spool"
    bn.enqueue(spool, bn.make_messages(["ok"], "ryzen", NOW))
    (spool / "20261108T000000-broken.json").write_text("{not json")
    (spool / "20261108T000001-partial.json").write_text('{"id": "x"}')
    assert [entry["text"] for _, entry in bn.pending(spool)] == ["ok"]
    assert sorted(p.name for p in (spool / bn.REJECTED).iterdir()) == [
        "20261108T000000-broken.json",
        "20261108T000001-partial.json",
    ]
    assert "REJECTED" in capsys.readouterr().err
    assert bn.pending(tmp_path / "missing") == []


def test_stale_messages_are_those_older_than_the_limit(tmp_path):
    spool = tmp_path / "spool"
    bn.enqueue(spool, bn.make_messages(["old"], "ryzen", NOW - timedelta(hours=30)))
    bn.enqueue(spool, bn.make_messages(["new"], "ryzen", NOW - timedelta(hours=2)))
    assert [p.name[-5 - 12 :] for p in bn.stale(spool, NOW, 24)] == [
        bn.pending(spool)[0][1]["id"] + ".json"
    ]


# --- Delivery ---


@pytest.mark.asyncio
async def test_delivered_messages_leave_the_spool(tmp_path):
    settings = settings_for(tmp_path)
    service = Service()
    code = await bn.run(f"{REPORT}\nsecond", settings, service, NOW)
    assert code == 0 and len(service.notes) == 2
    assert spooled(tmp_path) == []


@pytest.mark.asyncio
async def test_while_the_service_is_away_messages_wait_and_the_next_run_delivers_them(
    tmp_path, capsys
):
    settings = settings_for(tmp_path)
    assert await bn.run(REPORT, settings, Away(), NOW) == 0
    assert len(spooled(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "[bm-notify ryzen] [BM sync] main" in err and "service not reachable" in err
    # Later, with nothing new to report (the watch runs bm-notify on every pass):
    service = Service()
    assert await bn.run("", settings, service, NOW + timedelta(minutes=5)) == 0
    assert [note.content.splitlines()[0][:18] for note in service.notes] == ["Meldung: [BM sync]"]
    assert spooled(tmp_path) == []


@pytest.mark.asyncio
async def test_delivery_stops_at_the_first_unreachable_and_keeps_the_order(tmp_path):
    settings = settings_for(tmp_path)
    bn.enqueue(settings.notify.spool, bn.make_messages(["one", "two", "three"], "ryzen", NOW))
    service = Service(None, bn.Unreachable("gone"))
    report = await bn.deliver(settings.notify.spool, settings.notify, service.write, NOW)
    assert report == bn.DeliveryReport(delivered=1, rejected=0, waiting=2, unreachable=True)
    assert [entry["text"] for _, entry in bn.pending(settings.notify.spool)] == ["two", "three"]
    assert len(service.notes) == 2  # "three" was not even tried


@pytest.mark.asyncio
async def test_a_refused_message_is_retried_then_set_aside_and_blocks_nobody(tmp_path, capsys):
    settings = settings_for(tmp_path, max_attempts=2)
    spool = settings.notify.spool
    bn.enqueue(spool, bn.make_messages(["bad", "good"], "ryzen", NOW))
    refuse = bn.Rejected("project not found")
    first = await bn.deliver(spool, settings.notify, Service(refuse).write, NOW)
    assert first == bn.DeliveryReport(delivered=1, rejected=0, waiting=1)
    assert [e["attempts"] for _, e in bn.pending(spool)] == [1]
    second = await bn.deliver(spool, settings.notify, Service(refuse).write, NOW)
    assert second == bn.DeliveryReport(delivered=0, rejected=1, waiting=0)
    assert len(list((spool / bn.REJECTED).glob("*.json"))) == 1
    assert "refused 2 times: project not found" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_cut_off_run_does_not_create_a_second_note(tmp_path):
    """The note exists but the spool file is still there: the replay builds the same title."""
    settings = settings_for(tmp_path)
    spool = settings.notify.spool
    bn.enqueue(spool, bn.make_messages(["once"], "ryzen", NOW))
    first, second = Service(), Service()
    path, _ = bn.pending(spool)[0]
    kept = path.read_bytes()
    await bn.deliver(spool, settings.notify, first.write, NOW)
    path.write_bytes(kept)  # the delete never happened
    await bn.deliver(spool, settings.notify, second.write, NOW + timedelta(minutes=5))
    assert first.notes == second.notes


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("ALREADY_EXISTS: A note already exists at 'x'. Nothing was written.", None),
        ("A note already exists", None),
        ("HTTP 409 conflict", bn.Unreachable),
        ("another sync is busy", bn.Unreachable),
        ("NOT_FOUND: project 'meldungen' does not exist", bn.Rejected),
        ("FORBIDDEN: project is read-only", bn.Rejected),
    ],
)
def test_tool_errors_are_sorted_into_delivered_busy_and_refused(text, verdict):
    assert bn.classify_error(text) is verdict


# --- The whole run ---


@pytest.mark.asyncio
async def test_without_a_notify_project_the_report_stays_in_the_journal(tmp_path, capsys):
    settings = hs.parse_settings({"host_name": "ryzen", "notify": {"spool": str(tmp_path / "s")}})
    service = Service()
    assert await bn.run(REPORT, settings, service, NOW) == 0
    assert service.notes == [] and not (tmp_path / "s").exists()
    assert "[bm-notify ryzen] [BM sync] main" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_spool_that_cannot_be_written_fails_loudly_with_the_message_in_the_journal(
    tmp_path, capsys
):
    blocker = tmp_path / "spool"
    blocker.write_text("a file where the directory should be")
    code = await bn.run(REPORT, settings_for(tmp_path), Service(), NOW)
    assert code == 1
    err = capsys.readouterr().err
    assert "[BM sync] main" in err and "cannot spool" in err


@pytest.mark.asyncio
async def test_old_waiting_messages_are_called_out_on_every_run(tmp_path, capsys):
    settings = settings_for(tmp_path, stale_hours=1)
    bn.enqueue(settings.notify.spool, bn.make_messages(["old"], "ryzen", NOW - timedelta(hours=3)))
    await bn.run("", settings, Away(), NOW)
    assert "WARNING 1 message(s) waiting for more than 1 h" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_an_empty_run_with_an_empty_spool_does_nothing(tmp_path, capsys):
    service = Service()
    assert await bn.run("", settings_for(tmp_path), service, NOW) == 0
    assert service.notes == [] and capsys.readouterr().err == ""


def test_status_reports_waiting_stale_and_rejected(tmp_path, capsys):
    settings = settings_for(tmp_path, stale_hours=24)
    assert bn.status(settings, NOW) == 0
    bn.enqueue(settings.notify.spool, bn.make_messages(["old"], "ryzen", NOW - timedelta(hours=30)))
    assert bn.status(settings, NOW) == 2
    out = capsys.readouterr().out
    assert "1 waiting (1 older than 24 h), 0 rejected" in out
