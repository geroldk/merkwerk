"""Tests for structural section scanning and read-time slicing (SPEC-47 / #1403).

Pure-function tests over dedented strings: no fixtures, no I/O. The scanner and
slicers are re-run at read time on the exact content being served, so these
boundary cases pin canonical-content correctness.
"""

import random
from textwrap import dedent

import pytest

from basic_memory.markdown.sections import (
    LineRange,
    NoteSlice,
    NoteSliceError,
    SectionSelector,
    SectionSpan,
    TRUNCATION_MARKER_TEMPLATE,
    _frontmatter_line_count,
    _token_budget_line_count,
    locate_section,
    parse_line_range,
    parse_section_selector,
    scan_sections,
    slice_note_content,
)

# --- scan_sections ---


def test_scan_empty_body_returns_no_sections():
    assert scan_sections("") == ()


def test_scan_body_without_headings_returns_no_sections():
    assert scan_sections("just prose\n\nmore prose\n") == ()


def test_scan_nested_headings_build_paths_and_nested_spans():
    body = dedent("""\
        # Top
        intro
        ## Sub
        sub body
        ## Sub2
        more
        # Next
        tail""")

    sections = scan_sections(body)

    by_heading = {section.heading: section for section in sections}
    assert [section.heading for section in sections] == ["Top", "Sub", "Sub2", "Next"]

    top = by_heading["Top"]
    assert top.level == 1
    assert top.path == ("Top",)
    assert top.heading_path == "Top"
    assert (top.start_line, top.end_line) == (1, 6)

    sub = by_heading["Sub"]
    assert sub.level == 2
    assert sub.path == ("Top", "Sub")
    assert sub.heading_path == "Top/Sub"
    assert (sub.start_line, sub.end_line) == (3, 4)

    sub2 = by_heading["Sub2"]
    assert sub2.path == ("Top", "Sub2")
    assert (sub2.start_line, sub2.end_line) == (5, 6)

    nxt = by_heading["Next"]
    assert nxt.path == ("Next",)
    assert (nxt.start_line, nxt.end_line) == (7, 8)

    # Subsections nest inside their parent: overlapping spans are correct.
    assert top.start_line <= sub.start_line and sub.end_line <= top.end_line


def test_scan_skipped_heading_level_keeps_ancestor_path():
    body = "# A\n### Deep\ncontent\n"

    sections = scan_sections(body)

    assert sections[1].path == ("A", "Deep")
    assert sections[1].level == 3


def test_scan_duplicate_headings_same_path_count_duplicate_index():
    body = dedent("""\
        # Spec
        ## Auth
        first
        ## Auth
        second""")

    sections = scan_sections(body)

    auths = [section for section in sections if section.heading == "Auth"]
    assert [section.duplicate_index for section in auths] == [0, 1]
    assert [section.path for section in auths] == [("Spec", "Auth"), ("Spec", "Auth")]
    assert (auths[0].start_line, auths[0].end_line) == (2, 3)
    assert (auths[1].start_line, auths[1].end_line) == (4, 5)


def test_scan_same_heading_under_different_parents_gets_distinct_paths():
    body = dedent("""\
        # Auth
        ## Decisions
        a
        # Ops
        ## Decisions
        b""")

    sections = scan_sections(body)

    decisions = [section for section in sections if section.heading == "Decisions"]
    assert [section.heading_path for section in decisions] == ["Auth/Decisions", "Ops/Decisions"]
    assert [section.duplicate_index for section in decisions] == [0, 0]


def test_scan_ignores_headings_inside_fenced_code_blocks():
    body = dedent("""\
        # Real
        ```
        # not a heading
        ```
        ~~~
        ## also not one
        ~~~
        done""")

    sections = scan_sections(body)

    assert [section.heading for section in sections] == ["Real"]
    assert sections[0].end_line == 8


def test_scan_setext_headings_span_their_underline():
    body = dedent("""\
        Title
        =====
        body

        Sub
        ---
        more""")

    sections = scan_sections(body)

    assert [(section.heading, section.level) for section in sections] == [("Title", 1), ("Sub", 2)]
    title, sub = sections
    assert (title.start_line, title.end_line) == (1, 7)
    assert title.path == ("Title",)
    assert (sub.start_line, sub.end_line) == (5, 7)
    assert sub.path == ("Title", "Sub")


def test_scan_empty_section_ends_on_its_own_heading_line():
    body = "# A\n# B"

    sections = scan_sections(body)

    a, b = sections
    assert (a.start_line, a.end_line) == (1, 1)
    assert (b.start_line, b.end_line) == (2, 2)


def test_scan_byte_offsets_round_trip_unicode_content():
    body = "# Héllo ☕\ncafé costs 5€\n# End\nfin"

    sections = scan_sections(body)

    encoded = body.encode("utf-8")
    hello, end = sections
    assert encoded[hello.start_offset : hello.end_offset].decode("utf-8") == (
        "# Héllo ☕\ncafé costs 5€\n"
    )
    assert encoded[end.start_offset : end.end_offset].decode("utf-8") == "# End\nfin"
    assert end.end_offset == len(encoded)


def test_scan_trailing_newline_is_included_in_final_section_bytes():
    body = "# Only\ntext\n"

    (only,) = scan_sections(body)

    assert (only.start_line, only.end_line) == (1, 2)
    assert body.encode("utf-8")[only.start_offset : only.end_offset].decode("utf-8") == body


def test_scan_lone_carriage_returns_count_as_line_boundaries():
    # markdown-it normalizes "\r\n?|\n" to "\n" before assigning token line maps,
    # so a lone \r is a line boundary; counting by split("\n") shifted every later
    # heading and indexed past the line table (issue #1403 review).
    body = "# One\ralpha\r# Two\rbeta"

    one, two = scan_sections(body)

    assert (one.start_line, one.end_line) == (1, 2)
    assert (two.start_line, two.end_line) == (3, 4)
    encoded = body.encode("utf-8")
    assert encoded[one.start_offset : one.end_offset].decode("utf-8") == "# One\ralpha\r"
    assert encoded[two.start_offset : two.end_offset].decode("utf-8") == "# Two\rbeta"
    assert two.end_offset == len(encoded)


def test_scan_crlf_terminators_round_trip_original_bytes():
    body = "# A\r\nbody\r\n# B\r\ntail\r\n"

    a, b = scan_sections(body)

    assert (a.start_line, a.end_line) == (1, 2)
    assert (b.start_line, b.end_line) == (3, 4)
    encoded = body.encode("utf-8")
    assert encoded[a.start_offset : a.end_offset].decode("utf-8") == "# A\r\nbody\r\n"
    assert encoded[b.start_offset : b.end_offset].decode("utf-8") == "# B\r\ntail\r\n"
    assert b.end_offset == len(encoded)


# --- parse_section_selector ---


@pytest.mark.parametrize(
    ("selector", "segments", "duplicate_index"),
    [
        ("Decisions", ("Decisions",), 0),
        ("Auth/Decisions", ("Auth", "Decisions"), 0),
        ("Heading[1]", ("Heading",), 1),
        ("Auth/Decisions[2]", ("Auth", "Decisions"), 2),
        ("## Auth", ("Auth",), 0),
        ("# Auth / ### Decisions ", ("Auth", "Decisions"), 0),
        ("#tag", ("#tag",), 0),
    ],
)
def test_parse_section_selector_accepted_forms(selector, segments, duplicate_index):
    parsed = parse_section_selector(selector)

    assert parsed is not None
    assert parsed.raw == selector
    assert parsed.segments == segments
    assert parsed.duplicate_index == duplicate_index


@pytest.mark.parametrize("selector", ["", "   ", "[2]"])
def test_parse_section_selector_rejects_empty_forms(selector):
    assert parse_section_selector(selector) is None


@pytest.mark.parametrize(
    ("selector", "heading", "segments"),
    [
        ("Decisions", "Decisions", ("Decisions",)),
        ("Ein/Aus", "Ein/Aus", ("Ein", "Aus")),
        ("  ## Ein/Aus [1] ", "Ein/Aus", ("Ein", "Aus")),
        ("KI / ML", "KI / ML", ("KI", "ML")),
        ("Fragen/Antworten[1][0]", "Fragen/Antworten[1]", ("Fragen", "Antworten[1]")),
        # Malformed as a path, but possibly a heading's whole text.
        ("A//B", "A//B", None),
        ("/Auth", "/Auth", None),
        ("Auth/", "Auth/", None),
        ("A/ /B", "A/ /B", None),
    ],
)
def test_parse_section_selector_keeps_both_readings(selector, heading, segments):
    parsed = parse_section_selector(selector)

    assert parsed is not None
    assert parsed.heading == heading
    assert parsed.segments == segments


# --- parse_line_range ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2-4", LineRange(start=2, end=4)),
        ("3-", LineRange(start=3, end=None)),
        ("5", LineRange(start=5, end=5)),
        (" 2 - 4 ", LineRange(start=2, end=4)),
        ("7-7", LineRange(start=7, end=7)),
    ],
)
def test_parse_line_range_accepted_forms(value, expected):
    assert parse_line_range(value) == expected


@pytest.mark.parametrize("value", ["", "0", "-5", "a", "2-1", "1-b", "1.5-2", "0-"])
def test_parse_line_range_rejects_malformed_forms(value):
    assert parse_line_range(value) is None


# --- _frontmatter_line_count ---


def test_frontmatter_line_count_counts_well_formed_block():
    text = "---\ntitle: T\ntags: [a]\n---\n\nbody"
    assert _frontmatter_line_count(text) == 4


def test_frontmatter_line_count_zero_without_opening_fence():
    assert _frontmatter_line_count("# Heading\n---\n") == 0
    assert _frontmatter_line_count("") == 0


def test_frontmatter_line_count_zero_for_unterminated_block():
    assert _frontmatter_line_count("---\ntitle: T\nbody without closing fence") == 0


def test_frontmatter_line_count_counts_carriage_return_terminated_block():
    # Lone \r terminators count as line boundaries, matching the document line
    # table the count indexes into.
    assert _frontmatter_line_count("---\rtitle: T\r---\rbody") == 3


# --- slice_note_content: section selection ---

_DOC = dedent("""\
    ---
    title: Spec
    ---

    # Auth
    line a
    ## Decisions
    d1
    d2
    ## Ops
    o1
    # Tail
    z""")


def _slice(full_text: str, **kwargs) -> NoteSlice:
    result = slice_note_content(full_text, **kwargs)
    assert isinstance(result, NoteSlice)
    return result


def _slice_error(full_text: str, **kwargs) -> NoteSliceError:
    result = slice_note_content(full_text, **kwargs)
    assert isinstance(result, NoteSliceError)
    return result


def test_slice_by_section_returns_document_absolute_span():
    selector = parse_section_selector("Decisions")
    assert selector is not None

    sliced = _slice(_DOC, section=selector, note_title="Spec")

    assert sliced.content == "## Decisions\nd1\nd2"
    assert sliced.section == "Auth/Decisions"
    assert (sliced.start_line, sliced.end_line) == (7, 9)
    assert sliced.total_lines == 13
    assert not sliced.truncated
    assert sliced.continue_line is None
    # The returned coordinates address the same lines in the full document.
    assert "\n".join(_DOC.split("\n")[6:9]) == sliced.content


def test_slice_by_qualified_path_disambiguates():
    doc = dedent("""\
        # Auth
        ## Decisions
        a
        # Ops
        ## Decisions
        b""")
    selector = parse_section_selector("Ops/Decisions")
    assert selector is not None

    sliced = _slice(doc, section=selector)

    assert sliced.content == "## Decisions\nb"
    assert sliced.section == "Ops/Decisions"
    assert (sliced.start_line, sliced.end_line) == (5, 6)


def test_slice_ambiguous_suffix_lists_qualified_paths():
    doc = "# Auth\n## Decisions\na\n# Ops\n## Decisions\nb"
    selector = parse_section_selector("Decisions")
    assert selector is not None

    error = _slice_error(doc, section=selector, note_title="Spec")

    assert "'Decisions' is ambiguous in note 'Spec'" in error.message
    assert "Auth/Decisions, Ops/Decisions" in error.message


def test_slice_duplicate_headings_addressed_by_bracket_index():
    doc = dedent("""\
        # Spec
        ## Auth
        first
        ## Auth
        second""")
    selector = parse_section_selector("Auth[1]")
    assert selector is not None

    sliced = _slice(doc, section=selector)

    assert sliced.content == "## Auth\nsecond"
    assert sliced.section == "Spec/Auth[1]"
    assert (sliced.start_line, sliced.end_line) == (4, 5)


def test_slice_duplicate_index_out_of_range_reports_count():
    doc = "# Spec\n## Auth\nfirst\n## Auth\nsecond"
    selector = parse_section_selector("Auth[5]")
    assert selector is not None

    error = _slice_error(doc, section=selector)

    assert "2 section(s) share the path 'Spec/Auth'" in error.message
    assert "duplicate indexes 0-1" in error.message


def test_slice_unknown_section_lists_available_headings_with_duplicates_bracketed():
    doc = "# Spec\n## Auth\nfirst\n## Auth\nsecond\n## Ops\nz"
    selector = parse_section_selector("Nope")
    assert selector is not None

    error = _slice_error(doc, section=selector, note_title="Spec")

    assert "'Nope' not found in note 'Spec'" in error.message
    assert "Available sections: Spec, Spec/Auth[0], Spec/Auth[1], Spec/Ops" in error.message


def test_slice_unknown_section_in_heading_less_note():
    selector = parse_section_selector("Anything")
    assert selector is not None

    error = _slice_error("plain prose only", section=selector)

    assert "(note has no headings)" in error.message
    assert " in note " not in error.message


def test_slice_by_section_after_stray_carriage_return_serves_the_heading():
    # Regression: split("\n") counting let a stray \r shift the section span so
    # this selector silently served only "tail" (issue #1403 review).
    selector = parse_section_selector("H")
    assert selector is not None

    sliced = _slice("one\rtwo\n# H\ntail", section=selector)

    assert sliced.content == "# H\ntail"
    assert (sliced.start_line, sliced.end_line) == (3, 4)
    assert sliced.total_lines == 4


# --- slice_note_content: line ranges ---


def test_slice_by_line_range_addresses_full_document():
    sliced = _slice(_DOC, lines=LineRange(start=2, end=4))

    assert sliced.content == "title: Spec\n---\n"
    assert (sliced.start_line, sliced.end_line) == (2, 4)
    assert sliced.total_lines == 13
    assert sliced.section is None


def test_slice_open_ended_line_range_runs_to_end():
    sliced = _slice(_DOC, lines=LineRange(start=12, end=None))

    assert sliced.content == "# Tail\nz"
    assert (sliced.start_line, sliced.end_line) == (12, 13)


def test_slice_line_range_end_clamps_to_total_lines():
    sliced = _slice(_DOC, lines=LineRange(start=13, end=999))

    assert sliced.content == "z"
    assert (sliced.start_line, sliced.end_line) == (13, 13)


def test_slice_line_range_starting_past_end_is_an_error():
    """A range past the last line fails instead of serving an inverted 500-13 range (#1634)."""
    error = _slice_error(_DOC, lines=LineRange(start=500, end=510))

    assert error.message == "start_line 500 is past the end of the document (13 lines)"


def test_slice_single_line_range():
    sliced = _slice(_DOC, lines=LineRange(start=5, end=5))

    assert sliced.content == "# Auth"


@pytest.mark.parametrize(
    "arguments",
    [
        {"lines": LineRange(start=1, end=None)},
        {"lines": LineRange(start=1, end=5)},
        {"max_tokens": 5},
    ],
)
def test_slice_empty_document_yields_the_empty_range(arguments):
    """An empty document has no line 1, so the slice is 0-0 of 0, never 1-0."""
    sliced = _slice("", **arguments)

    assert sliced.content == ""
    assert (sliced.start_line, sliced.end_line, sliced.total_lines) == (0, 0, 0)


def test_slice_empty_document_rejects_ranges_past_line_one():
    error = _slice_error("", lines=LineRange(start=2, end=None))

    assert error.message == "start_line 2 is past the end of the document (0 lines)"


def test_slice_section_and_lines_are_mutually_exclusive():
    selector = parse_section_selector("Auth")
    assert selector is not None
    with pytest.raises(ValueError, match="mutually exclusive"):
        slice_note_content(_DOC, section=selector, lines=LineRange(start=1, end=2))


def test_slice_requires_at_least_one_slice_parameter():
    with pytest.raises(ValueError, match="requires section, lines, or max_tokens"):
        slice_note_content(_DOC)


# --- slice_note_content: max_tokens ---


def test_slice_max_tokens_alone_serves_body_after_frontmatter():
    sliced = _slice(_DOC, max_tokens=10_000)

    assert sliced.content == "\n".join(_DOC.split("\n")[3:])
    assert (sliced.start_line, sliced.end_line) == (4, 13)
    assert not sliced.truncated
    assert sliced.continue_line is None


def test_slice_max_tokens_cuts_at_section_boundary_with_marker():
    # Body: one small section, then a second section; a budget that fits the
    # first section but not both must cut exactly at the second heading.
    doc = "# One\n" + "a" * 40 + "\n# Two\n" + "b" * 400
    sliced = _slice(doc, max_tokens=20)  # budget: 80 chars

    kept_lines = sliced.content.split("\n")
    assert kept_lines[-1] == TRUNCATION_MARKER_TEMPLATE.format(max_tokens=20, continue_line=3)
    assert kept_lines[:-1] == ["# One", "a" * 40]
    assert sliced.truncated
    assert sliced.continue_line == 3
    assert (sliced.start_line, sliced.end_line) == (1, 2)


def test_slice_max_tokens_continue_read_reconstructs_document():
    doc = "# One\n" + "a" * 40 + "\n# Two\n" + "b" * 40
    sliced = _slice(doc, max_tokens=20)
    assert sliced.truncated and sliced.continue_line is not None

    remainder = _slice(doc, lines=LineRange(start=sliced.continue_line, end=None))

    kept = "\n".join(sliced.content.split("\n")[:-1])
    assert kept + "\n" + remainder.content == doc


def test_slice_max_tokens_falls_back_to_paragraph_boundary():
    # No section boundary after the start: the cut lands on the blank line
    # between paragraphs instead.
    doc = "# Only\n" + "a" * 40 + "\n\n" + "b" * 400
    sliced = _slice(doc, max_tokens=20)  # budget: 80 chars

    kept_lines = sliced.content.split("\n")
    assert kept_lines[:-1] == ["# Only", "a" * 40]
    assert sliced.truncated
    assert sliced.continue_line == 3


def test_slice_max_tokens_line_backstop_within_one_paragraph():
    # One giant paragraph: the backstop cuts at the last whole line in budget.
    doc = "x" * 30 + "\n" + "y" * 30 + "\n" + "z" * 30
    sliced = _slice(doc, max_tokens=20)  # budget: 80 chars, fits two lines (61)

    kept_lines = sliced.content.split("\n")
    assert kept_lines[:-1] == ["x" * 30, "y" * 30]
    assert sliced.continue_line == 3


def test_slice_max_tokens_applies_after_section_selection():
    doc = "# Big\n" + "a" * 40 + "\n\n" + "b" * 400 + "\n# Other\nz"
    selector = parse_section_selector("Big")
    assert selector is not None

    sliced = _slice(doc, section=selector, max_tokens=20)

    kept_lines = sliced.content.split("\n")
    assert kept_lines[:-1] == ["# Big", "a" * 40]
    assert sliced.section == "Big"
    assert sliced.truncated
    assert sliced.continue_line == 3
    assert sliced.end_line == 2


def test_slice_max_tokens_within_budget_is_not_truncated():
    sliced = _slice("# A\nshort", max_tokens=100)

    assert sliced.content == "# A\nshort"
    assert not sliced.truncated


# --- _token_budget_line_count backstops ---


def test_token_budget_no_cut_when_text_fits():
    assert _token_budget_line_count("short", ["short"], budget_chars=100) is None


def test_token_budget_single_over_budget_line_is_served_whole():
    text = "x" * 500
    assert _token_budget_line_count(text, [text], budget_chars=10) is None


def test_token_budget_keeps_first_line_even_when_it_exceeds_budget():
    lines = ["x" * 500, "tail"]
    assert _token_budget_line_count("\n".join(lines), lines, budget_chars=10) == 1


# --- Headings whose own text contains "/" (decision B3 A) ---
#
# A heading named exactly as the selector is written, "/" included, wins over the
# path reading. Without such a heading the path reading runs exactly as before.

_SLASH_DOC = dedent("""\
    ---
    title: Slash
    ---

    # Top
    ## Ein/Aus
    literal
    ## Ein
    ### Aus
    nested
    ## Log
    one
    ## Log
    two
    ## Log[1]
    bracketed""")


def _selector(text: str) -> SectionSelector:
    selector = parse_section_selector(text)
    assert selector is not None, f"{text!r} did not parse"
    return selector


def _read_section(full_text: str, selector_text: str) -> NoteSlice | NoteSliceError:
    return slice_note_content(full_text, section=_selector(selector_text))


def _read_section_ok(full_text: str, selector_text: str) -> NoteSlice:
    sliced = _read_section(full_text, selector_text)
    assert isinstance(sliced, NoteSlice), sliced
    return sliced


def _read_section_error(full_text: str, selector_text: str) -> NoteSliceError:
    sliced = _read_section(full_text, selector_text)
    assert isinstance(sliced, NoteSliceError), sliced
    return sliced


def test_selector_naming_a_slash_heading_whole_wins_over_the_path():
    sliced = _read_section_ok(_SLASH_DOC, "Ein/Aus")

    assert sliced.content == "## Ein/Aus\nliteral"
    assert (sliced.start_line, sliced.end_line) == (6, 7)
    # "Top/Ein/Aus" would read back as heading "Aus" under "Ein".
    assert sliced.section == "Ein/Aus"


@pytest.mark.parametrize("selector", ["Ein / Aus", "Top/Ein/Aus", "Aus", "# Ein/### Aus"])
def test_path_reading_stays_reachable_when_both_readings_exist(selector):
    """Regression: spaces around the "/" or an added ancestor still mean the path."""
    sliced = _read_section_ok(_SLASH_DOC, selector)

    assert sliced.content == "### Aus\nnested"
    assert (sliced.start_line, sliced.end_line) == (9, 10)


@pytest.mark.parametrize(
    ("selector", "content"),
    [
        ("Ein", "## Ein\n### Aus\nnested"),
        # [n] stays the duplicate index even beside a heading literally named "Log[1]".
        ("Log[1]", "## Log\ntwo"),
        ("Log[1][0]", "## Log[1]\nbracketed"),
        ("## Log[0]", "## Log\none"),
    ],
)
def test_selectors_without_a_slash_keep_their_meaning(selector, content):
    """Regression: the one-heading and the path reading coincide without a "/"."""
    assert _read_section_ok(_SLASH_DOC, selector).content == content


def test_slash_selector_without_such_heading_reads_as_a_path():
    """Regression: a "## Ein/Aus" line inside a code fence is no heading."""
    doc = "# Top\n## Ein\n### Aus\nnested\n\n```\n## Ein/Aus\n```\n# Tail\nz"

    sliced = _read_section_ok(doc, "Ein/Aus")

    assert sliced.content == "### Aus\nnested\n\n```\n## Ein/Aus\n```"
    assert sliced.section == "Top/Ein/Aus"


@pytest.mark.parametrize(
    ("doc", "selector", "content", "label"),
    [
        ("# Ein/Aus\nbody", "Ein/Aus", "# Ein/Aus\nbody", "Ein/Aus"),
        ("# A\n## B\n### Ein/Aus\nbody", "Ein/Aus", "### Ein/Aus\nbody", "Ein/Aus"),
        ("## Ein/Aus ##\nbody", "Ein/Aus", "## Ein/Aus ##\nbody", "Ein/Aus"),
        ("Ein/Aus\n=======\nbody", "Ein/Aus", "Ein/Aus\n=======\nbody", "Ein/Aus"),
        ("## Ein/Aus\nbody", "  ## Ein/Aus  ", "## Ein/Aus\nbody", "Ein/Aus"),
        ("## KI / ML\nbody", "KI / ML", "## KI / ML\nbody", "KI / ML"),
        (
            "## Fragen/Antworten\nbody",
            "Fragen/Antworten[0]",
            "## Fragen/Antworten\nbody",
            "Fragen/Antworten",
        ),
        ("## Q/A[1]\nbody", "Q/A[1][0]", "## Q/A[1]\nbody", "Q/A[1][0]"),
    ],
)
def test_slash_heading_is_found_at_any_level_and_in_any_spelling(doc, selector, content, label):
    sliced = _read_section_ok(doc, selector)

    assert sliced.content == content
    assert sliced.section == label


@pytest.mark.parametrize("heading", ["/etc", "Ende/", "http://x//y", "A/ /B", "/"])
def test_heading_with_an_empty_path_segment_is_found_by_its_whole_text(heading):
    doc = f"# Top\n## {heading}\nbody\n## Next\nz"

    sliced = _read_section_ok(doc, heading)

    assert sliced.content == f"## {heading}\nbody"


@pytest.mark.parametrize("selector", ["KI/ML", "ki / ml", "KI /ML"])
def test_slash_heading_text_compares_exactly(selector):
    """Regression: case and the spacing around "/" must match the heading."""
    error = _read_section_error("# Top\n## KI / ML\nbody", selector)

    assert error.message.startswith(f"Section '{selector}' not found.")


@pytest.mark.parametrize("selector", ["A//B", "/Auth", "Auth/", "A/ /B"])
def test_malformed_path_without_such_heading_is_an_invalid_selector(selector):
    error = _read_section_error(_SLASH_DOC, selector)

    assert error.invalid_selector
    assert error.message == f"Invalid section selector: '{selector}'"


def test_slash_heading_duplicates_count_under_one_parent():
    doc = "# Top\n## Ein/Aus\none\n## Ein/Aus\ntwo\n# Tail\nz"

    first = _read_section_ok(doc, "Ein/Aus")
    second = _read_section_ok(doc, "Ein/Aus[1]")

    assert (first.content, first.section) == ("## Ein/Aus\none", "Ein/Aus[0]")
    assert (second.content, second.section) == ("## Ein/Aus\ntwo", "Ein/Aus[1]")


def test_slash_heading_index_out_of_range_does_not_fall_back_to_the_path():
    doc = "# Top\n## Ein/Aus\nliteral\n## Ein\n### Aus\nfirst\n### Aus\nsecond"

    error = _read_section_error(doc, "Ein/Aus[1]")

    assert error.message == (
        "Section 'Ein/Aus[1]' not found: 1 heading(s) named 'Ein/Aus' (duplicate indexes 0-0). "
        "For the path (heading 'Aus' under 'Ein') write Top/Ein/Aus[0], Top/Ein/Aus[1]."
    )
    # The path reading is still reachable by spelling it with spaces.
    assert _read_section_ok(doc, "Ein / Aus[1]").content == "### Aus\nsecond"


def test_slash_heading_under_two_parents_is_ambiguous_and_names_each_location():
    doc = "# A\n## Ein/Aus\na\n# B\n## Ein/Aus\nb"

    error = _read_section_error(doc, "Ein/Aus")

    # Decision B3 C: the locations read back, so the refusal offers them as names.
    assert error.message == (
        "Section 'Ein/Aus' is ambiguous: 2 paths end in a heading named 'Ein/Aus'. "
        "Qualify it: A > Ein/Aus, B > Ein/Aus."
    )
    assert _read_section_ok(doc, "B > Ein/Aus").content == "## Ein/Aus\nb"


def test_slash_paths_that_join_to_one_string_stay_apart():
    """("A/B", "X/Y") and ("A", "B", "X/Y") both join to "A/B/X/Y"."""
    doc = "# A/B\n## X/Y\none\n# A\n## B\n### X/Y\ntwo"

    error = _read_section_error(doc, "X/Y")

    assert "is ambiguous: 2 paths end in a heading named 'X/Y'" in error.message


# --- locate_section: the edit side of the same selector ---


def test_locate_section_names_a_slash_heading_whole():
    span = locate_section(_SLASH_DOC, _selector("Ein/Aus"))

    assert isinstance(span, SectionSpan)
    assert (span.heading, span.label) == ("Ein/Aus", "Ein/Aus")
    # 0-indexed whole-document lines: "## Ein/Aus" is line 6, its body ends before line 8.
    assert (span.heading_start, span.body_start, span.end) == (5, 6, 7)


def test_locate_section_offers_slash_heading_duplicates_by_their_own_text():
    """The joined path "Top/Ein/Aus[1]" would read back as heading "Aus" under "Ein"."""
    doc = "# Top\n## Ein/Aus\none\n## Ein/Aus\ntwo\n## Ein\n### Aus\nx\n### Aus\ny"

    refused = locate_section(doc, _selector("Ein/Aus"))
    assert isinstance(refused, NoteSliceError)
    assert refused.message.endswith("Name one: Ein/Aus[0], Ein/Aus[1].")

    # Following the hint reaches the second "Ein/Aus", not the second "Aus".
    span = locate_section(doc, _selector("Ein/Aus[1]"))
    assert isinstance(span, SectionSpan)
    assert (span.heading, span.heading_start) == ("Ein/Aus", 3)


def test_locate_section_path_hints_keep_their_old_spelling():
    """Regression: duplicates found by the path reading are offered by full path."""
    doc = "# Top\n## Log\none\n## Log\ntwo"

    refused = locate_section(doc, _selector("Log"))

    assert isinstance(refused, NoteSliceError)
    assert refused.message == (
        "Multiple sections found with header 'Log'. Name one: Top/Log[0], Top/Log[1]."
    )


# --- Every name a read returns or a message offers reads back to its section ---
#
# A read returns the section's name, and the messages offer names to choose from. A
# caller passes such a name on to an edit, so it must reach the same heading again,
# never a neighbour that happens to spell the same string (review of B3, finding 1).

# "Plan/KI/ML[1]" spells the second "ML" under "KI", not the second "KI/ML".
_PLAN_TWICE = "# Plan\n## KI/ML\none\n## KI/ML\ntwo\n## KI\n### ML\nx\n### ML\ny\n"
# The heading "Top/Dec" takes the selector "Top/Dec" from heading "Dec" under "Top".
_LITERAL_SPELLS_A_PATH = "# Top\n## Dec\nx\n# Top/Dec\ny\n"
# "Top/Log[1]" reads as the second "Log", not the heading "Log[1]".
_BRACKETED_HEADING = "# Top\n## Log\none\n## Log\ntwo\n## Log[1]\nb\n"
# ("R", "A/B", "X") and ("R", "A", "B", "X") both join to "R/A/B/X"; only the
# location "R > A/B > X" names the first "X".
_PATHS_JOIN_ALIKE = (
    "# R\n\n## A/B\n\n### X\n\nx-under-slash\n\n## A\n\n### B\n\n#### X\n\nx-under-path\n"
)
# The selector "KI/ML" ends in both "KI/ML"; only their locations tell them apart.
_NESTED_IN_ITSELF = "## KI/ML\nouter\n### KI/ML\ninner\n"
# A top heading "Top" that recurs deeper: "Top" and "Top/Dec" end in both (B3 C).
_TOP_RECURS = "# Top\n## Dec\nx\n# Other\n## Top\n### Dec\ny\n"


def _assert_reads_back(doc: str, name: str, start_line: int) -> None:
    """The name reaches the heading on start_line, for a read and for an edit."""
    assert _read_section_ok(doc, name).start_line == start_line
    span = locate_section(doc, _selector(name))
    assert isinstance(span, SectionSpan), span
    assert span.heading_start + 1 == start_line


@pytest.mark.parametrize(
    ("doc", "selector", "content", "label"),
    [
        (_PLAN_TWICE, "KI/ML", "## KI/ML\none", "KI/ML[0]"),
        (_PLAN_TWICE, "KI/ML[1]", "## KI/ML\ntwo", "KI/ML[1]"),
        (_PLAN_TWICE, "KI / ML[1]", "### ML\ny", "Plan/KI/ML[1]"),
        (_LITERAL_SPELLS_A_PATH, "Dec", "## Dec\nx", "Dec"),
        (_LITERAL_SPELLS_A_PATH, "Top/Dec", "# Top/Dec\ny", "Top/Dec"),
        (_BRACKETED_HEADING, "Log[1][0]", "## Log[1]\nb", "Top/Log[1][0]"),
        (_BRACKETED_HEADING, "Log[1]", "## Log\ntwo", "Top/Log[1]"),
        (_PATHS_JOIN_ALIKE, "B/X", "#### X\n\nx-under-path", "R/A/B/X"),
        (_PATHS_JOIN_ALIKE, "A/B", "## A/B\n\n### X\n\nx-under-slash\n", "A/B"),
    ],
)
def test_returned_section_name_reads_back_to_the_same_section(doc, selector, content, label):
    sliced = _read_section_ok(doc, selector)

    assert sliced.content == content
    assert sliced.section == label
    _assert_reads_back(doc, label, sliced.start_line)


def test_returned_name_falls_back_to_the_callers_spelling():
    """No canonical spelling reaches heading "Aus" under "Ein" here, the caller's does.

    Headings named like each spelling take it over: the path, the spaced path, the
    location "Ein > Aus" and its "> " form.
    """
    doc = (
        "# Ein/Aus\nliteral\n# Ein / Aus\nspaced\n# Ein > Aus\ngt\n# > Ein > Aus\nrooted\n"
        "# Ein\n## Aus\nnested\n# Aus\nroot"
    )

    sliced = _read_section_ok(doc, " Ein /Aus ")

    assert sliced.content == "## Aus\nnested"
    assert sliced.section == "Ein /Aus[0]"
    _assert_reads_back(doc, "Ein /Aus[0]", sliced.start_line)


def _offered(message: str, intro: str) -> tuple[list[str], list[str]]:
    """Split an offer into its names and the locations no selector reaches."""
    offer = message.split(intro, 1)[1].removesuffix(".")
    names, _, unreachable = offer.partition(
        ". Not reachable by a section selector (read by lines): "
    )
    return (
        [] if names == "(none)" else names.split(", "),
        unreachable.split(", ") if unreachable else [],
    )


@pytest.mark.parametrize(
    ("doc", "available"),
    [
        (
            _PLAN_TWICE,
            "Plan, KI/ML[0], KI/ML[1], Plan/KI, Plan/KI/ML[0], Plan/KI/ML[1].",
        ),
        (_LITERAL_SPELLS_A_PATH, "Top, Dec, Top/Dec."),
        (_BRACKETED_HEADING, "Top, Top/Log[0], Top/Log[1], Top/Log[1][0]."),
        (_PATHS_JOIN_ALIKE, "R, A/B, R > A/B > X, R/A, R/A/B, R/A/B/X."),
        (_NESTED_IN_ITSELF, "> KI/ML, KI/ML > KI/ML."),
        (_TOP_RECURS, "> Top, Top > Dec, Other, Other/Top, Other/Top/Dec."),
        (
            # An empty heading has no name, nor does "X" under it: "X" is the top "X",
            # and a location cannot spell the empty text.
            "# Top\n#\nempty\n## X\nx\n# X\nroot\n",
            "Top, > X. Not reachable by a section selector (read by lines): "
            "'' (lines 2-5), 'X' (lines 4-5).",
        ),
    ],
)
def test_available_sections_offers_only_names_that_read_back(doc, available):
    error = _read_section_error(doc, "Nope")

    assert error.message == f"Section 'Nope' not found. Available sections: {available}"
    names, unreachable = _offered(error.message, "Available sections: ")
    reached = [_read_section_ok(doc, name).start_line for name in names]
    for name, start_line in zip(names, reached):
        _assert_reads_back(doc, name, start_line)
    # Every heading is offered once: by a name of its own, or by its lines.
    assert len(set(reached)) == len(names)
    assert len(names) + len(unreachable) == len(scan_sections(doc))


def test_paths_that_join_to_one_string_are_ambiguous():
    """Finding 5: two headings "X" whose paths join alike were taken as one path."""
    error = _read_section_error(_PATHS_JOIN_ALIKE, "X")

    assert error.message == "Section 'X' is ambiguous. Qualify it: R > A/B > X, R/A/B/X."
    # An edit refuses as well instead of changing the first "X" it meets.
    assert isinstance(locate_section(_PATHS_JOIN_ALIKE, _selector("X")), NoteSliceError)
    # "X[1]" names no second section of one path; it is the same ambiguity.
    assert "is ambiguous" in _read_section_error(_PATHS_JOIN_ALIKE, "X[1]").message


@pytest.mark.parametrize(
    ("doc", "selector", "message", "path_names"),
    [
        (
            "# A\n## KI/ML\na\n# B\n## KI/ML\nb\n## KI\n### ML\nnested\n",
            "KI/ML",
            "Section 'KI/ML' is ambiguous: 2 paths end in a heading named 'KI/ML'. "
            "Qualify it: A > KI/ML, B > KI/ML. For the path (heading 'ML' under 'KI') "
            "write B/KI/ML.",
            ["B/KI/ML"],
        ),
        (
            "# Q\n## Fragen/Antworten\nf\n## Fragen\n### Antworten\nf0\n### Antworten\nf1\n",
            "Fragen/Antworten[1]",
            "Section 'Fragen/Antworten[1]' not found: 1 heading(s) named 'Fragen/Antworten' "
            "(duplicate indexes 0-0). For the path (heading 'Antworten' under 'Fragen') "
            "write Q/Fragen/Antworten[0], Q/Fragen/Antworten[1].",
            ["Q/Fragen/Antworten[0]", "Q/Fragen/Antworten[1]"],
        ),
        (
            _NESTED_IN_ITSELF,
            "KI/ML",
            "Section 'KI/ML' is ambiguous: 2 paths end in a heading named 'KI/ML'. "
            "Qualify it: > KI/ML, KI/ML > KI/ML.",
            [],
        ),
        (
            # "/etc" spells no path at all, so there is none to point to.
            "# A\n## /etc\na\n# B\n## /etc\nb\n",
            "/etc",
            "Section '/etc' is ambiguous: 2 paths end in a heading named '/etc'. "
            "Qualify it: A > /etc, B > /etc.",
            [],
        ),
    ],
)
def test_refused_slash_heading_names_the_way_to_the_path(doc, selector, message, path_names):
    """Finding 4: the refusal says how to reach the path the selector also spells."""
    error = _read_section_error(doc, selector)

    assert error.message == message
    for name in path_names:
        assert _read_section_ok(doc, name).content.startswith("### ")


def test_offered_duplicate_names_read_back_for_an_edit():
    refused = locate_section(_PLAN_TWICE, _selector("ML"))

    assert isinstance(refused, NoteSliceError)
    assert refused.message.endswith("Name one: Plan/KI/ML[0], Plan/KI/ML[1].")
    _assert_reads_back(_PLAN_TWICE, "Plan/KI/ML[1]", 9)


# --- Locations: a heading's full path from the top (decision B3 C) ---
#
# Messages wrote where an unnamed heading sits as "A > KI/ML"; that location is now a
# selector itself, read from the heading with no heading above it down to the
# heading, "/" being plain text inside it. "> Top" is the location of a top heading.

_TWO_PARENTS = "# A\n## KI/ML\na\n# B\n## KI/ML\nb\n## KI\n### ML\nnested\n"


@pytest.mark.parametrize(
    ("selector", "location", "duplicate_index"),
    [
        ("A > KI/ML", ("A", "KI/ML"), 0),
        ("> Top", ("Top",), 0),
        ("Top  >  Dec", ("Top", "Dec"), 0),
        ("## Top > ## Dec", ("Top", "Dec"), 0),
        ("> A > Log[1]", ("A", "Log"), 1),
        ("A > Log[1][0]", ("A", "Log[1]"), 0),
        ("R > A/B > X", ("R", "A/B", "X"), 0),
    ],
)
def test_parse_section_selector_reads_a_location(selector, location, duplicate_index):
    parsed = _selector(selector)

    assert parsed.location == location
    assert parsed.duplicate_index == duplicate_index


@pytest.mark.parametrize(
    "selector", ["a>b", "a >b", "a> b", "x -> y", "A >= B", ">", ">Top", "Top >", "A >  > B"]
)
def test_parse_section_selector_bare_gt_or_a_gap_is_no_location(selector):
    parsed = _selector(selector)

    assert parsed.location is None
    assert parsed.heading == selector


@pytest.mark.parametrize(
    ("doc", "selector", "content", "label"),
    [
        (_TWO_PARENTS, "A > KI/ML", "## KI/ML\na", "A > KI/ML"),
        (_TWO_PARENTS, "B > KI/ML", "## KI/ML\nb", "B > KI/ML"),
        (_NESTED_IN_ITSELF, "> KI/ML", "## KI/ML\nouter\n### KI/ML\ninner", "> KI/ML"),
        (_NESTED_IN_ITSELF, "KI/ML > KI/ML", "### KI/ML\ninner", "KI/ML > KI/ML"),
        (_TOP_RECURS, "> Top", "# Top\n## Dec\nx", "> Top"),
        (_TOP_RECURS, "Top > Dec", "## Dec\nx", "Top > Dec"),
        (_TOP_RECURS, "  Top  >  Dec ", "## Dec\nx", "Top > Dec"),
        (_PATHS_JOIN_ALIKE, "R > A/B > X", "### X\n\nx-under-slash\n", "R > A/B > X"),
    ],
)
def test_location_reaches_headings_no_other_selector_names(doc, selector, content, label):
    sliced = _read_section_ok(doc, selector)

    assert sliced.content == content
    assert sliced.section == label
    _assert_reads_back(doc, label, sliced.start_line)


@pytest.mark.parametrize(
    ("doc", "selector", "content", "label"),
    [
        # A note whose first heading is "## Intro": nothing is above it, so it starts
        # a location, like the "# Main" after it.
        ("## Intro\nx\n# Main\n## Intro\ny\n", "> Intro", "## Intro\nx", "> Intro"),
        ("## Intro\nx\n# Main\n## Intro\ny\n", "Main > Intro", "## Intro\ny", "Main/Intro"),
        # Levels skipped on the way down do not count, only the headings above.
        ("# A\n### Deep\nd\n", "A > Deep", "### Deep\nd", "A/Deep"),
        # A location that names every heading works in an ordinary note too; the name
        # returned stays the usual path.
        (
            "# Plan\n## Auth\n### Decisions\nd\n",
            "Plan > Auth > Decisions",
            "### Decisions\nd",
            "Plan/Auth/Decisions",
        ),
    ],
)
def test_location_is_read_from_the_heading_with_none_above_it(doc, selector, content, label):
    sliced = _read_section_ok(doc, selector)

    assert (sliced.content, sliced.section) == (content, label)


@pytest.mark.parametrize("selector", ["KI > ML", "top > dec", "Top > dec", "> Dec"])
def test_location_is_no_suffix_and_compares_exactly(selector):
    doc = _TWO_PARENTS if selector == "KI > ML" else _TOP_RECURS

    error = _read_section_error(doc, selector)

    assert error.message.startswith(
        f"Section '{selector}' not found. A location starts at a heading with no heading "
        "above it. Available sections: "
    )


def test_location_index_counts_the_sections_of_that_location():
    """Twins share one location, so [n] picks among them, and children of twins too."""
    doc = (
        "# A\n## KI/ML\none\n### Note\nn0\n### Note\nn1\n## KI/ML\ntwo\n### Note\nn2\n"
        "# B\n## KI/ML\nb\n"
    )

    assert _read_section_ok(doc, "A > KI/ML[1]").content == "## KI/ML\ntwo\n### Note\nn2"
    # The Note under the second twin is the third of location "A > KI/ML > Note".
    assert _read_section_ok(doc, "A > KI/ML > Note[2]").content == "### Note\nn2"
    refused = locate_section(doc, _selector("A > KI/ML"))
    assert isinstance(refused, NoteSliceError)
    assert refused.message == (
        "Multiple sections found with header 'A > KI/ML'. Name one: A > KI/ML[0], A > KI/ML[1]."
    )
    assert _read_section_error(doc, "A > KI/ML[2]").message == (
        "Section 'A > KI/ML[2]' not found: 2 section(s) have the location 'A > KI/ML' "
        "(duplicate indexes 0-1)."
    )
    assert _read_section_error(doc, "> B[1]").message == (
        "Section '> B[1]' not found: 1 section(s) have the location '> B' (duplicate indexes 0-0)."
    )


def test_heading_named_like_a_location_wins():
    """Regression: a heading whose own text is "A > B" is read by that text, as before."""
    doc = "# R\n## A > B\nlit\n## A\n### B\npath\n"

    assert _read_section_ok(doc, "A > B").content == "## A > B\nlit"
    assert _read_section_ok(doc, "R/A > B").content == "## A > B\nlit"


def test_heading_named_like_a_location_keeps_its_old_refusals():
    """Regression: refusals for "A > B" read as before, with no hint at a path."""
    doc = "# R\n## A > B\nr\n# S\n## A > B\ns\n## A > B\nt\n"

    assert _read_section_error(doc, "A > B").message == (
        "Section 'A > B' is ambiguous. Qualify it: R/A > B, S/A > B."
    )
    assert _read_section_error(doc, "S/A > B[2]").message == (
        "Section 'S/A > B[2]' not found: 2 section(s) share the path 'S/A > B' "
        "(duplicate indexes 0-1)."
    )


def test_location_reaches_the_path_beside_a_heading_named_like_it():
    doc = "# A > B\nlit\n# A/B\nslash\n# A / B\nspaced\n# A\n## B\npath\n# B\nroot\n"

    # "A > B", "A/B" and "A / B" are headings' own texts; "B" ends two paths.
    assert _read_section_ok(doc, "> A > B").content == "## B\npath"
    error = _read_section_error(doc, "Nope")
    assert "> A > B" in _offered(error.message, "Available sections: ")[0]
    _assert_reads_back(doc, "> A > B", 8)


def test_old_slash_path_through_a_gt_heading_keeps_its_meaning():
    """Regression: "A > B/C" read heading "C" under "A > B" before B3 C, and still does."""
    doc = "# A > B\n## C\nold\n# A\n## B/C\nloc\n"

    assert _read_section_ok(doc, "A > B/C").content == "## C\nold"


def test_location_through_a_slash_heading_beside_an_old_path():
    doc = "# A > B\n## C\nold\n# A\n## B/C\nloc\n"

    assert _read_section_ok(doc, "> A > B/C").content == "## B/C\nloc"


@pytest.mark.parametrize("heading", ["a>b", "a >b", "a> b", "x -> y", "A >= B", ">", ">Top"])
def test_headings_with_a_bare_gt_keep_working(heading):
    """Regression: only a ">" with a space on each side separates a location."""
    doc = f"# Top\n## {heading}\nbody\n## Next\nz\n"

    sliced = _read_section_ok(doc, heading)

    assert (sliced.content, sliced.section) == (f"## {heading}\nbody", f"Top/{heading}")


def test_heading_that_starts_like_a_location_is_read_by_its_text():
    """Regression: a heading "> quote" is named by its text, ahead of the location form."""
    doc = "# Top\n## > quote\nbody\n# quote\nroot\n"

    assert _read_section_ok(doc, "> quote").content == "## > quote\nbody"


def test_names_keep_a_heading_text_that_starts_like_a_heading_prefix():
    """Selector segments drop a "## " prefix, so a name adds one for heading "# x"."""
    doc = "# A\n## # x\na\n# B\n## # x\nb\n"

    error = _read_section_error(doc, "Nope")

    assert error.message.endswith("Available sections: A, A/# # x, B, B/# # x.")
    assert _read_section_ok(doc, "B/# # x").content == "## # x\nb"
    assert _read_section_ok(doc, "B > # # x").content == "## # x\nb"


def test_each_heading_of_a_location_loses_one_heading_prefix():
    """Review B3 C: "# # x > y" lost both "# " of its first heading, unlike "> # # x > y"."""
    doc = "# # x\n## y\nhy\n# x\n## y\npy\n"

    for selector in ("# # x > y", "> # # x > y"):
        assert _selector(selector).location == ("# x", "y")
        assert _read_section_ok(doc, selector).content == "## y\nhy"
        span = locate_section(doc, _selector(selector))
        assert isinstance(span, SectionSpan) and span.heading_start == 1, (selector, span)
    # One prefix per heading, as before.
    assert _read_section_ok(doc, "# x > y").content == "## y\npy"
    assert _selector("## Top > ## Dec").location == ("Top", "Dec")
    assert _selector("## > Top").location == ("Top",)


def test_unreachable_sections_are_pointed_to_by_document_lines():
    """An empty heading has no name; the message gives the lines to read it by."""
    doc = "---\ntitle: E\n---\n# Top\n#\nempty\n"

    error = _read_section_error(doc, "Nope")

    assert error.message == (
        "Section 'Nope' not found. Available sections: Top. Not reachable by a section "
        "selector (read by lines): '' (lines 5-6)."
    )
    by_lines = slice_note_content(doc, lines=LineRange(start=5, end=6))
    assert isinstance(by_lines, NoteSlice)
    assert by_lines.content == "#\nempty"


# --- Property: every heading has a name, and every name reads back ---
#
# Generated notes mix repeated texts, "/" and ">" in heading texts, top headings that
# recur deeper, notes starting at "##", setext headings and fake headings in code
# fences. Texts hold no ", " or ".", which would blur the comma-joined offers.

_CLEAN_TEXTS = (
    "A", "B", "Top", "Log", "KI", "ML", "KI/ML", "/etc", "A/B", "Q/A",
    "a>b", "x -> y", "Log[1]", "# x",
)  # fmt: skip
# Texts a location cannot spell: empty, " > " inside, or a ">" at an edge.
_HARD_TEXTS = (*_CLEAN_TEXTS, "A > B", "> A", "B >", "")
_SETEXT_TEXTS = ("A", "B", "Top", "KI/ML", "a>b")


def _generated_note(rng: random.Random, texts: tuple[str, ...]) -> str:
    lines = ["---", "title: G", "---"] if rng.random() < 0.3 else []
    for _ in range(rng.randint(1, 12)):
        level = rng.randint(1, 4)
        if level <= 2 and rng.random() < 0.15:
            lines += ["", rng.choice(_SETEXT_TEXTS), "===" if level == 1 else "---"]
        else:
            lines.append(f"{'#' * level} {rng.choice(texts)}".rstrip())
        if rng.random() < 0.15:
            lines += ["```", f"## {rng.choice(texts)}", "```"]
        lines.append("body")
    return "\n".join(lines) + "\n"


def _start_line(doc: str, name: str) -> int:
    """Read by name and check the edit side and the returned name land on the same line."""
    sliced = _read_section_ok(doc, name)
    span = locate_section(doc, _selector(name))
    if isinstance(span, NoteSliceError):
        # Only twins refuse an unindexed edit; the offered names carry [n] for them.
        assert span.message.startswith("Multiple sections found"), span.message
    else:
        assert span.heading_start + 1 == sliced.start_line
    assert sliced.section is not None
    _assert_reads_back(doc, sliced.section, sliced.start_line)
    return sliced.start_line


def _offer_names(message: str, intro: str) -> list[str]:
    """The names after an intro such as "Qualify it: ", up to the end of that sentence."""
    if intro not in message:
        return []
    names = message.split(intro, 1)[1].split(".", 1)[0]
    return [] if names == "(none)" else names.split(", ")


def _written_as_location(text: str) -> bool:
    """A heading text a location cannot hold, or that takes a location's spelling."""
    return not text or " > " in text or text.startswith("> ") or text.endswith(" >")


def _check_generated_note(doc: str, *, every_heading_named: bool) -> None:
    body_lines = doc.split("\n")[_frontmatter_line_count(doc) :]
    sections = scan_sections("\n".join(body_lines))
    offset = len(doc.split("\n")) - len(body_lines)

    def headings_at(start: int) -> list[str]:
        return [s.heading for s in sections if offset + s.start_line == start]

    # Every heading is offered once: by a name that reads back, or by its lines.
    names, unreachable = _offered(_read_section_error(doc, "Nope").message, "Available sections: ")
    reached = [_start_line(doc, name) for name in names]
    assert len(set(reached)) == len(names)
    assert len(names) + len(unreachable) == len(sections)
    unnamed = [s for s in sections if offset + s.start_line not in reached]
    assert [
        f"'{s.heading}' (lines {offset + s.start_line}-{offset + s.end_line})" for s in unnamed
    ] == unreachable
    # Decision B3 C: without such texts in the note, every heading has a name. With
    # them, a heading may stay unnamed only when the note holds one.
    if every_heading_named:
        assert unreachable == []
    elif unnamed:
        assert any(_written_as_location(s.heading) for s in sections)

    # The names a refusal offers read back into the sections the selector matched.
    for text in {s.heading for s in sections if s.heading}:
        selector = _selector(text)
        # "Log[1]" is the second "Log", and "# x" is heading "x", as for any selector.
        matched = selector.heading
        sliced = _read_section(doc, text)
        offered = (
            [] if isinstance(sliced, NoteSlice) else _offer_names(sliced.message, "Qualify it: ")
        )
        for name in offered:
            assert matched in headings_at(_start_line(doc, name)), (text, name)
        span = locate_section(doc, selector)
        if isinstance(span, NoteSliceError):
            for name in _offer_names(span.message, "Name one: "):
                assert matched in headings_at(_start_line(doc, name)), (text, name)


@pytest.mark.parametrize("seed", range(3))
def test_every_heading_of_a_generated_note_has_a_name_that_reads_back(seed):
    rng = random.Random(seed)
    for _ in range(40):
        _check_generated_note(_generated_note(rng, _CLEAN_TEXTS), every_heading_named=True)


@pytest.mark.parametrize("seed", range(3))
def test_names_read_back_even_where_a_location_cannot_spell_the_path(seed):
    rng = random.Random(1000 + seed)
    for _ in range(40):
        _check_generated_note(_generated_note(rng, _HARD_TEXTS), every_heading_named=False)
