"""Section edits share the read selector and keep the section's layout (SPEC-35.2, #1815)."""

import pytest

from basic_memory.services.note_preparation import (
    apply_edit_operation,
    insert_relative_to_section,
    replace_section_content,
)

TABLE_NOTE = "## Table\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Notes\n"


def test_replace_section_keeps_blank_lines_around_the_body():
    """#1815: the blank line under the heading and before the next one survive."""
    result = replace_section_content(TABLE_NOTE, "## Table", "| x | y |\n|---|---|\n| 9 | 8 |")

    assert result == "## Table\n\n| x | y |\n|---|---|\n| 9 | 8 |\n\n## Notes\n"


def test_replace_section_trims_the_replacement_edges():
    result = replace_section_content(TABLE_NOTE, "Table", "\n\nnew body\n\n")

    assert result == "## Table\n\nnew body\n\n## Notes\n"


def test_replace_section_keeps_a_tight_layout_tight():
    result = replace_section_content("## A\nold\n## B\n", "A", "new")

    assert result == "## A\nnew\n## B\n"


def test_replacement_that_restates_the_heading_keeps_one_heading():
    result = replace_section_content(TABLE_NOTE, "Table", "## Table\n\nnew")

    assert result == "## Table\n\nnew\n\n## Notes\n"


def test_section_path_selects_a_nested_heading():
    note = "# Auth\n## Decisions\nauth old\n# Billing\n## Decisions\nbilling old\n"

    result = replace_section_content(note, "Billing/Decisions", "billing new")

    assert result == "# Auth\n## Decisions\nauth old\n# Billing\n## Decisions\nbilling new\n"


def test_duplicate_headings_need_an_index():
    note = "## Log\none\n## Log\ntwo\n"

    with pytest.raises(ValueError, match=r"Multiple sections found.*Log\[0\], Log\[1\]"):
        replace_section_content(note, "Log", "x")
    assert replace_section_content(note, "Log[1]", "2") == "## Log\none\n## Log\n2\n"


def test_ambiguous_paths_are_refused():
    note = "# Auth\n## Decisions\na\n# Billing\n## Decisions\nb\n"

    with pytest.raises(ValueError, match="ambiguous"):
        replace_section_content(note, "Decisions", "x")


def test_missing_section_lists_the_available_ones():
    with pytest.raises(ValueError, match="not found. Available sections: Table, Notes"):
        replace_section_content(TABLE_NOTE, "Nope", "x")


def test_setext_heading_is_a_section():
    note = "Intro\n=====\nold\n# Next\n"

    assert replace_section_content(note, "Intro", "new") == "Intro\n=====\nnew\n# Next\n"
    assert (
        insert_relative_to_section(note, "Intro", "first", "after")
        == "Intro\n=====\nfirst\n\nold\n# Next\n"
    )


def test_frontmatter_lines_are_counted():
    note = "---\ntitle: T\n---\n\n## A\nold\n"

    assert replace_section_content(note, "A", "new") == "---\ntitle: T\n---\n\n## A\nnew\n"


def test_replace_without_subsections_stops_at_the_first_child():
    note = "## A\nold\n### A1\nchild\n## B\n"

    result = replace_section_content(note, "A", "new", replace_subsections=False)

    assert result == "## A\nnew\n### A1\nchild\n## B\n"


def test_insert_before_uses_the_selector():
    note = "# Top\n## Log\nentry\n"

    assert insert_relative_to_section(note, "Top/Log", "added", "before") == (
        "# Top\n\nadded\n\n## Log\nentry\n"
    )


def test_prepend_leaves_a_blank_line_before_a_heading():
    """#1815: prepended text no longer runs into the note's opening heading."""
    assert apply_edit_operation("# Title\nbody", "prepend", "Line.") == "Line.\n\n# Title\nbody"
    assert apply_edit_operation("# Title\nbody", "prepend", "Line.\n") == "Line.\n\n# Title\nbody"
    assert (
        apply_edit_operation("---\ntitle: T\n---\n\n# Title\n", "prepend", "Line.")
        == "---\ntitle: T\n---\n\nLine.\n\n# Title"
    )
    # A body that opens with text keeps the existing single-newline join.
    assert apply_edit_operation("body", "prepend", "- item") == "- item\nbody"


# --- Line terminators ---


def test_lone_carriage_returns_address_the_same_lines_as_the_scanner():
    """A lone \\r ends a line for the scanner, so the edit must split there too."""
    note = "## A\rold\r## B\rkeep"

    assert replace_section_content(note, "A", "new") == "## A\rnew\r## B\rkeep"
    assert insert_relative_to_section(note, "B", "first", "after") == (
        "## A\rold\r## B\rfirst\r\rkeep"
    )


def test_crlf_lines_outside_the_edit_keep_their_bytes():
    note = "## A\r\nold\r\n## B\r\nkeep\r\n"

    assert replace_section_content(note, "A", "new") == "## A\r\nnew\r\n## B\r\nkeep\r\n"


def test_an_edit_at_the_end_keeps_the_final_newline_state():
    assert replace_section_content("## A\nold", "A", "new") == "## A\nnew"
    assert replace_section_content("## A\nold\n", "A", "new") == "## A\nnew\n"
    assert insert_relative_to_section("## A", "A", "first", "after") == "## A\nfirst"


def test_restated_heading_in_any_spelling_is_dropped():
    """A replacement copied from bm_read may restate the heading with closing hashes."""
    note = "## Decisions\nold\n## Next\n"

    assert replace_section_content(note, "Decisions", "## Decisions ##\nnew") == (
        "## Decisions\nnew\n## Next\n"
    )
    setext = "Decisions\n---------\nold\n## Next\n"
    assert replace_section_content(setext, "Decisions", "Decisions\n---\nnew") == (
        "Decisions\n---------\nnew\n## Next\n"
    )


# --- Headings whose own text contains "/" (decision B3 A) ---

BOTH_READINGS = "# Top\n## KI/ML\nliteral\n## KI\n### ML\nnested\n"


def test_replace_section_names_a_slash_heading_whole():
    assert replace_section_content(BOTH_READINGS, "KI/ML", "new") == (
        "# Top\n## KI/ML\nnew\n## KI\n### ML\nnested\n"
    )


def test_replace_section_still_reaches_the_path_with_spaces():
    """Regression: spaces around "/" keep the path reading (heading ML under KI)."""
    assert replace_section_content(BOTH_READINGS, "KI / ML", "new") == (
        "# Top\n## KI/ML\nliteral\n## KI\n### ML\nnew\n"
    )


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("before", "# Top\n\nadded\n\n## KI/ML\nliteral\n## KI\n### ML\nnested\n"),
        ("after", "# Top\n## KI/ML\nadded\n\nliteral\n## KI\n### ML\nnested\n"),
    ],
)
def test_insert_around_a_slash_heading(position, expected):
    assert insert_relative_to_section(BOTH_READINGS, "KI/ML", "added", position) == expected


def test_heading_with_an_empty_path_segment_can_be_edited():
    note = "## /etc\nold\n## Next\n"

    assert replace_section_content(note, "/etc", "new") == "## /etc\nnew\n## Next\n"


@pytest.mark.parametrize("command", ["replace_section", "insert_before_section"])
@pytest.mark.parametrize("section", ["A//B", "/Auth", "Auth/", "A/ /B"])
def test_malformed_path_without_such_heading_keeps_its_old_refusal(command, section):
    """Regression: the refusal is the one these selectors had before (B3 A)."""
    with pytest.raises(ValueError) as refused:
        apply_edit_operation(BOTH_READINGS, command, "x", section=section)

    assert str(refused.value) == f"Section '{section}' is not a heading name."


def test_repeated_slash_heading_is_offered_by_its_own_text():
    note = "## KI/ML\none\n## KI/ML\ntwo\n## KI\n### ML\nx\n### ML\ny\n"

    with pytest.raises(ValueError, match=r"Name one: KI/ML\[0\], KI/ML\[1\]\."):
        replace_section_content(note, "KI/ML", "new")
    # The offered name edits the second "KI/ML", not the second "ML" under "KI".
    assert replace_section_content(note, "KI/ML[1]", "TWO") == (
        "## KI/ML\none\n## KI/ML\nTWO\n## KI\n### ML\nx\n### ML\ny\n"
    )


def test_edit_refuses_paths_that_join_to_one_string():
    """Review of B3, finding 5: "X" ends two different paths that both join to "R/A/B/X".

    Before, the edit took the first "X" it met (the one under "A/B") without asking.
    """
    note = "# R\n## A/B\n### X\nx-under-slash\n## A\n### B\n#### X\nx-under-path\n"

    with pytest.raises(
        ValueError, match=r"Section 'X' is ambiguous\. Qualify it: R > A/B > X, R/A/B/X\."
    ):
        replace_section_content(note, "X", "new")
    assert replace_section_content(note, "R/A/B/X", "new") == (
        "# R\n## A/B\n### X\nx-under-slash\n## A\n### B\n#### X\nnew\n"
    )


# --- Locations: a heading's full path from the top (decision B3 C) ---

TWO_PARENTS = "# A\n## KI/ML\na\n# B\n## KI/ML\nb\n"


def test_replace_section_takes_a_location():
    assert replace_section_content(TWO_PARENTS, "B > KI/ML", "new") == (
        "# A\n## KI/ML\na\n# B\n## KI/ML\nnew\n"
    )


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("before", "added\n\n# Top\n## Dec\nx\n# Other\n## Top\n"),
        ("after", "# Top\nadded\n\n## Dec\nx\n# Other\n## Top\n"),
    ],
)
def test_insert_around_the_location_of_a_top_heading(position, expected):
    note = "# Top\n## Dec\nx\n# Other\n## Top\n"

    assert insert_relative_to_section(note, "> Top", "added", position) == expected


def test_edit_refuses_an_ambiguous_slash_heading_and_offers_its_locations():
    with pytest.raises(ValueError, match=r"Qualify it: A > KI/ML, B > KI/ML\."):
        replace_section_content(TWO_PARENTS, "KI/ML", "new")


def test_heading_written_like_a_location_is_edited_by_its_text():
    """Regression: a heading whose own text is "A > B" wins over the location reading."""
    note = "# R\n## A > B\nlit\n## A\n### B\npath\n"

    assert replace_section_content(note, "A > B", "new") == (
        "# R\n## A > B\nnew\n## A\n### B\npath\n"
    )
