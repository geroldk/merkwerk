"""BM-010: the rules for `created`/`modified` in accepted frontmatter."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

from basic_memory.file_utils import parse_frontmatter
from basic_memory.services import note_times
from basic_memory.services.note_preparation import PreparedEntityMove
from basic_memory.services.note_times import (
    format_note_time,
    keep_move_times,
    stamp_markdown_times,
    wanted_note_times,
)

NOW = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone(timedelta(hours=1)))
OLD_C, OLD_M = "2026-09-01T10:00:00+02:00", "2026-09-23T11:31:45+02:00"


def times_for(*args, **kwargs) -> dict[str, str]:
    result = wanted_note_times(*args, **kwargs)
    assert result is not None
    return result


def note(extra: str = "", body: str = "# T\n\ntext\n") -> str:
    return f"---\ntitle: T\ntype: note\npermalink: p/t\n{extra}---\n\n{body}"


def keys(extra_c: str = OLD_C, extra_m: str = OLD_M) -> str:
    return f"created: '{extra_c}'\nmodified: '{extra_m}'\n"


def test_format_is_iso_with_offset_to_the_second():
    assert format_note_time(NOW.replace(microsecond=7)) == "2026-11-08T12:00:00+01:00"
    assert (
        format_note_time(datetime(2026, 1, 2, 3, 4, 5)).count("+")
        + format_note_time(datetime(2026, 1, 2, 3, 4, 5)).count("-")
        >= 3
    )  # a naive value is read as local time and gets an offset


def test_a_new_note_gets_now_for_both_and_keeps_writer_supplied_times():
    assert times_for(note(), now=NOW, current_markdown=None) == {
        "created": "2026-11-08T12:00:00+01:00",
        "modified": "2026-11-08T12:00:00+01:00",
    }
    assert times_for(note(keys()), now=NOW, current_markdown=None) == {
        "created": OLD_C,
        "modified": OLD_M,
    }


def test_a_changed_note_renews_modified_and_never_changes_created():
    current = note(keys())
    changed = note(keys(), body="# T\n\nmore\n")
    wanted = times_for(changed, now=NOW, current_markdown=current)
    assert wanted == {"created": OLD_C, "modified": "2026-11-08T12:00:00+01:00"}
    # A writer cannot rewrite created through an ordinary change.
    forged = note(keys("2000-01-01T00:00:00+01:00"), body="# T\n\nmore\n")
    assert times_for(forged, now=NOW, current_markdown=current)["created"] == OLD_C
    # Other frontmatter counts as a change too.
    other = note(keys() + "status: x\n")
    assert times_for(other, now=NOW, current_markdown=current)["modified"].startswith("2026-11-08")


def test_a_note_without_times_takes_created_from_the_row():
    row_c = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    wanted = times_for(
        note(body="# T\n\nmore\n"),
        now=NOW,
        current_markdown=note(),
        fallback_created=row_c,
        fallback_modified=row_c,
    )
    assert wanted["created"] == "2026-10-01T08:00:00+00:00"
    assert wanted["modified"] == "2026-11-08T12:00:00+01:00"


def test_a_write_of_only_the_time_fields_never_renews_modified():
    current = note(keys())
    # Backfill: explicit values stand.
    assert times_for(
        note(keys("2025-01-01T00:00:00+01:00", "2025-02-02T00:00:00+01:00")),
        now=NOW,
        current_markdown=current,
    ) == {"created": "2025-01-01T00:00:00+01:00", "modified": "2025-02-02T00:00:00+01:00"}
    # Nothing changed: the old times stay.
    assert times_for(current, now=NOW, current_markdown=current) == {
        "created": OLD_C,
        "modified": OLD_M,
    }
    # The writer dropped the keys: the old ones come back.
    assert times_for(note(), now=NOW, current_markdown=current) == {
        "created": OLD_C,
        "modified": OLD_M,
    }
    # A note that never had them gets the row's times.
    row = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    assert times_for(
        note(), now=NOW, current_markdown=note(), fallback_created=row, fallback_modified=row
    ) == {"created": "2026-10-01T08:00:00+00:00", "modified": "2026-10-01T08:00:00+00:00"}
    assert times_for(note(), now=NOW, current_markdown=note())["created"].startswith("2026-11-08")


def test_unreadable_frontmatter_is_left_to_the_caller():
    broken = "---\ntitle: [unclosed\n---\n\nbody\n"
    assert wanted_note_times(broken, now=NOW, current_markdown=None) is None
    assert wanted_note_times(note(), now=NOW, current_markdown=broken) is None
    assert stamp_markdown_times(broken, {"created": OLD_C, "modified": OLD_M}) == broken


def test_stamping_touches_only_the_time_lines():
    original = note().replace("title: T", "title: Strategie - KI-Boom Teilnahme - X")
    stamped = stamp_markdown_times(original, {"created": OLD_C, "modified": OLD_M})
    assert stamped == original.replace("---\n\n#", f"{keys()}---\n\n#")
    assert stamp_markdown_times(stamped, {"created": OLD_C, "modified": OLD_M}) is stamped
    # An existing line is replaced in place, CRLF notes stay CRLF.
    crlf = "---\r\ntitle: T\r\ncreated: 2020-01-01\r\n---\r\nBody\r\n"
    out = stamp_markdown_times(crlf, {"created": OLD_C, "modified": OLD_M})
    assert out == (
        f"---\r\ntitle: T\r\ncreated: '{OLD_C}'\r\nmodified: '{OLD_M}'\r\n---\r\nBody\r\n"
    )
    assert parse_frontmatter(out)["modified"] == OLD_M


def test_cases_the_text_path_cannot_handle_fall_back_to_a_yaml_rewrite():
    wanted = {"created": OLD_C, "modified": OLD_M}
    multi = "---\ntitle: T\ncreated:\n  - a\n---\n\nbody\n"
    assert parse_frontmatter(stamp_markdown_times(multi, wanted))["created"] == OLD_C
    no_close = "---\ntitle: T\nbody without a closing fence"
    assert note_times._set_time_lines(no_close, wanted) is None
    no_front = "just a body\n"
    assert stamp_markdown_times(no_front, wanted).startswith("---\n")


def test_a_plain_move_keeps_both_times_or_takes_the_rows():
    moved = PreparedEntityMove(
        file_path=Path("b/T.md"), markdown_content=note(), search_content="", permalink="p/t"
    )
    row_c = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    row_m = datetime(2026, 10, 2, 9, 0, 0, tzinfo=timezone.utc)
    stamped = keep_move_times(
        moved, current_markdown=note(), entity_created_at=row_c, entity_updated_at=row_m
    )
    assert parse_frontmatter(stamped.markdown_content)["created"] == "2026-10-01T08:00:00+00:00"
    assert parse_frontmatter(stamped.markdown_content)["modified"] == "2026-10-02T09:00:00+00:00"
    keep = PreparedEntityMove(
        file_path=Path("b/T.md"),
        markdown_content=note(keys()),
        search_content="",
        permalink="p/t",
    )
    assert (
        keep_move_times(
            keep, current_markdown=note(keys()), entity_created_at=row_c, entity_updated_at=row_m
        )
        is keep
    )
    broken = PreparedEntityMove(
        file_path=Path("b/T.md"),
        markdown_content="---\ntitle: [x\n---\n",
        search_content="",
        permalink="p/t",
    )
    assert (
        keep_move_times(
            broken, current_markdown=note(), entity_created_at=row_c, entity_updated_at=row_m
        )
        is broken
    )


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "git_backfill_times.py"
SPEC = importlib.util.spec_from_file_location("git_backfill_times", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
backfill = importlib.util.module_from_spec(SPEC)
sys.modules["git_backfill_times"] = backfill
SPEC.loader.exec_module(backfill)


def test_backfill_adds_exactly_the_missing_keys(tmp_path):
    created = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    updated = datetime(2026, 9, 23, 11, 31, 45, tzinfo=timezone(timedelta(hours=2)))
    result = backfill.backfilled(note(), created, updated)
    assert result == note(keys())
    assert backfill.backfilled(result, created, updated) is None
    only_created = note(f"created: '{OLD_C}'\n")
    out = backfill.backfilled(only_created, created, updated)
    assert out == note(keys())  # the existing value is never overwritten


def test_backfill_refuses_a_rewrite_that_would_change_other_lines(monkeypatch):
    created = updated = datetime(2026, 9, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(note_times, "stamp_markdown_times", lambda md, wanted: md + "extra\n")
    with pytest.raises(ValueError, match="more than the missing"):
        backfill.backfilled(note(), created, updated)
    monkeypatch.setattr(note_times, "stamp_markdown_times", lambda md, wanted: md)
    with pytest.raises(ValueError, match="read back"):
        backfill.backfilled(note(), created, updated)


def test_backfill_reads_tsv_with_later_files_winning(tmp_path):
    first, second = tmp_path / "a.tsv", tmp_path / "b.tsv"
    head = "external_id\tcreated_at_db\tupdated_at_db\n"
    first.write_text(head + "x\t2026-01-01 10:00:00.5\t2026-01-02 10:00:00\n")
    second.write_text(head + "x\t2026-02-01 10:00:00\t2026-02-02 10:00:00\n")
    times = backfill.read_tsv_times([first, second])
    assert times["x"][0].month == 2 and times["x"][1].day == 2
