"""Tests for canonical permalink utility helpers."""

import pytest

from basic_memory.utils import (
    build_permalink_candidates_after_file_paths,
    build_permalink_resolution_candidates,
    own_project_remainder,
    build_qualified_permalink_reference,
)


def test_workspace_qualified_permalink_candidates_include_legacy_without_rewrapping():
    candidates = build_permalink_resolution_candidates(
        "personal/main/notes/example",
        "main",
        workspace_permalink="personal",
    )

    assert candidates == [
        "personal/main/notes/example",
        "main/notes/example",
        "notes/example",
    ]


def test_project_prefixed_candidate_includes_short_legacy_when_project_prefix_disabled():
    candidates = build_permalink_resolution_candidates(
        "main/notes/example",
        "main",
        include_project=False,
    )

    assert candidates == [
        "main/notes/example",
        "notes/example",
    ]


def test_short_permalink_candidates_include_workspace_and_project_forms():
    candidates = build_permalink_resolution_candidates(
        "notes/example",
        "main",
        workspace_permalink="personal",
    )

    assert candidates == [
        "notes/example",
        "personal/main/notes/example",
        "main/notes/example",
    ]


def test_short_workspace_candidate_keeps_project_legacy_when_project_prefix_disabled():
    candidates = build_permalink_resolution_candidates(
        "notes/example",
        "main",
        include_project=False,
        workspace_permalink="personal",
    )

    assert candidates == [
        "notes/example",
        "personal/main/notes/example",
        "main/notes/example",
    ]


def test_qualified_permalink_reference_preserves_lookup_syntax():
    assert (
        build_qualified_permalink_reference(
            "main",
            "notes/roadmap.md",
            include_project=False,
        )
        == "notes/roadmap.md"
    )
    assert (
        build_qualified_permalink_reference(
            "main",
            "patterns/*",
            include_project=True,
        )
        == "main/patterns/*"
    )
    assert (
        build_qualified_permalink_reference(
            "main",
            "patterns/*",
            include_project=True,
            workspace_permalink="personal",
        )
        == "personal/main/patterns/*"
    )


def test_prefixed_candidates_keep_the_callers_spelling_of_the_remainder():
    """An explicit frontmatter permalink is stored verbatim, not as its slug (#1549)."""
    expected_remainders = ["s/ses_AbCdEf", "s/ses-ab-cd-ef"]

    project_prefixed = build_permalink_resolution_candidates(
        "main/s/ses_AbCdEf", "main", include_project=True
    )
    # The literal folder reading (a top-level `main/` folder, #2024) is tried before the
    # project prefix is stripped.
    assert project_prefixed == [
        "main/s/ses_AbCdEf",
        "main/s/ses-ab-cd-ef",
        "main/main/s/ses-ab-cd-ef",
        *expected_remainders,
    ]

    unprefixed_route = build_permalink_resolution_candidates(
        "main/s/ses_AbCdEf", "main", include_project=False
    )
    assert unprefixed_route == ["main/s/ses_AbCdEf", "main/s/ses-ab-cd-ef", *expected_remainders]

    workspace_qualified = build_permalink_resolution_candidates(
        "personal/main/s/ses_AbCdEf", "main", workspace_permalink="personal"
    )
    assert workspace_qualified == [
        "personal/main/s/ses_AbCdEf",
        "personal/main/s/ses-ab-cd-ef",
        "main/s/ses_AbCdEf",
        "main/s/ses-ab-cd-ef",
        *expected_remainders,
    ]


def test_non_markdown_identifiers_keep_their_extension():
    """Resource entities have no permalink, so `.txt` must not collapse to the .md stem (#1629)."""
    assert build_permalink_resolution_candidates("main/notes/foo.txt", "main") == [
        "main/notes/foo.txt"
    ]
    assert "main/notes/foo" in build_permalink_resolution_candidates("main/notes/foo.md", "main")
    # A version-like title is not a file extension.
    assert "main/release-2.0" in build_permalink_resolution_candidates("Release 2.0", "main")


@pytest.mark.parametrize(
    "identifier,include_project,workspace,expected",
    [
        # A note's permalink can end in a file suffix; routing stripped its project prefix.
        ("threads/fix.py", True, None, ["main/threads/fix.py"]),
        ("notes/foo.txt", True, None, ["main/notes/foo.txt"]),
        # The caller's spelling first, then its slug, which keeps the suffix.
        (
            "threads/Fix in tax_costs.py",
            True,
            None,
            ["main/threads/Fix in tax_costs.py", "main/threads/fix-in-tax-costs.py"],
        ),
        # Already qualified: the first candidates hold it verbatim.
        ("main/threads/fix.py", True, None, []),
        ("main::threads/fix.py", True, None, []),
        # Short permalinks without a workspace: as for any other identifier, no route.
        ("threads/fix.py", False, None, []),
        # A workspace route, then the older project form, whatever the project setting.
        (
            "threads/fix.py",
            True,
            "personal",
            ["personal/main/threads/fix.py", "main/threads/fix.py"],
        ),
        (
            "threads/fix.py",
            False,
            "personal",
            ["personal/main/threads/fix.py", "main/threads/fix.py"],
        ),
        # Already qualified on a workspace route, in either form: the reading below the
        # prefix supplies these routes after its own exact file paths (#1629).
        ("personal/main/threads/fix.py", True, "personal", []),
        ("personal/main/threads/fix.py", False, "personal", []),
        ("main/threads/fix.py", True, "personal", []),
        ("Main/threads/Fix.py", False, "personal", []),
    ],
)
def test_file_suffixed_identifiers_get_routed_candidates_after_file_paths(
    identifier, include_project, workspace, expected
):
    """The routes the first candidates leave out for a suffix (#1629), in their order."""
    assert (
        build_permalink_candidates_after_file_paths(
            identifier, "main", include_project, workspace_permalink=workspace
        )
        == expected
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "threads/fixnodot",
        "notes/plan.md",
        "Release 2.0",
        "aufgaben/ap-0.9-host-test",
        "threads/gd.py-in-den-katalog",
    ],
)
def test_identifiers_without_a_file_suffix_get_no_candidates_after_file_paths(identifier):
    """Their routed forms are already among the first candidates, so their meaning stays."""
    for workspace in (None, "personal"):
        assert (
            build_permalink_candidates_after_file_paths(
                identifier, "main", workspace_permalink=workspace
            )
            == []
        )
    assert build_permalink_candidates_after_file_paths("threads/fix.py", None) == []


def test_own_project_remainder_strips_every_routing_spelling():
    """One helper serves LinkResolver and the bulk resolver (#1626, #1629)."""
    assert own_project_remainder("main/assets/a.txt", "main") == "assets/a.txt"
    assert own_project_remainder("MAIN/Some Title", "main") == "Some Title"
    assert (
        own_project_remainder(
            "team-paul/main/assets/a.txt", "main", workspace_permalink="team-paul"
        )
        == "assets/a.txt"
    )
    # A workspace prefix only counts when the route is workspace-scoped.
    assert own_project_remainder("team-paul/main/x", "main") is None
    assert own_project_remainder("other/x", "main") is None
    assert own_project_remainder("main/", "main") is None
    assert own_project_remainder("main/x", None) is None
