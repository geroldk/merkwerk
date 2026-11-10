"""The files stage order (F2): new files first, old files after, never a lost only copy."""

import pytest

from basic_memory.index.git_file_projection_plan import (
    BLOCKED_PATH,
    EDITOR_CHANGE,
    FOREIGN_FILE,
    BaseFile,
    DiskView,
    FileProjectionPlan,
    FileStep,
    LiveNote,
    Occupant,
    RetiredFile,
    StepKind,
    identify_occupants,
    plan_file_projection,
    publication_proof_candidates,
)


def retired(path: str, *checksums: str) -> RetiredFile:
    return RetiredFile(path, frozenset(checksums))


def plan(notes, retirements=(), files=None, occupants=None, **disk):
    view = DiskView(files=files or {}, **disk)
    if occupants is None:
        occupants = identify_occupants(notes, retirements, view)
    return plan_file_projection(notes, retirements, view, occupants)


def kinds(result) -> list[tuple[StepKind, str]]:
    return [(step.kind, step.path) for step in result.steps]


def step(result, path: str) -> FileStep:
    return next(step for step in result.steps if step.path == path)


# --- Plain moves: the new file first ---


def test_a_rename_writes_the_new_path_before_the_old_one_goes():
    result = plan(
        [LiveNote("zz/n.md", "n"), LiveNote("a.md", "a", synced=True)],
        [retired("notes/n.md", "n")],
        files={"notes/n.md": "n", "a.md": "a"},
    )
    assert kinds(result) == [(StepKind.write, "zz/n.md"), (StepKind.delete, "notes/n.md")]
    assert step(result, "notes/n.md").requires == {"zz/n.md"}
    assert result.done == {"a.md"} and not result.conflicts


def test_each_old_file_goes_right_after_its_new_one():
    """C11: the duplicate window is one note long, not the whole folder."""
    count = 5
    notes = [LiveNote(f"new/{index}.md", str(index)) for index in range(count)]
    olds = [retired(f"old/{index}.md", str(index)) for index in range(count)]
    result = plan(notes, olds, files={f"old/{index}.md": str(index) for index in range(count)})
    assert kinds(result) == [
        pair
        for index in range(count)
        for pair in ((StepKind.write, f"new/{index}.md"), (StepKind.delete, f"old/{index}.md"))
    ]


def test_an_already_written_new_file_lets_the_old_one_go_first():
    """Crash after the writes: the new copy is verified by its hash, the old one goes."""
    result = plan(
        [LiveNote("moved.md", "m", synced=True)],
        [retired("old.md", "m")],
        files={"moved.md": "m", "old.md": "m"},
    )
    assert kinds(result) == [(StepKind.delete, "old.md")]
    assert result.done == {"moved.md"}


def test_a_note_deleted_in_git_and_a_vanished_old_file():
    result = plan(
        [LiveNote("keep.md", "k", synced=True)],
        [retired("gone.md", "g"), retired("vanished.md", "v")],
        files={"keep.md": "k", "gone.md": "g"},
        occupants={"gone.md": Occupant(None)},
    )
    assert kinds(result) == [(StepKind.delete, "gone.md"), (StepKind.complete, "vanished.md")]
    assert step(result, "gone.md").requires == frozenset()


# --- Reused paths: delete right before the write ---


def test_a_swap_is_broken_at_exactly_one_delete():
    notes = [LiveNote("p1.md", "b"), LiveNote("p2.md", "a")]
    rows = [retired("p1.md", "a"), retired("p2.md", "b")]
    result = plan(notes, rows, files={"p1.md": "a", "p2.md": "b"})
    assert [step.kind for step in result.steps] == [StepKind.replace, StepKind.replace]
    assert sum(step.breaks_cycle for step in result.steps) == 1
    first, second = result.steps
    assert first.breaks_cycle and first.requires == frozenset()
    # The second waits for the first: its bytes belong to the note just written.
    assert second.requires == {first.path}
    assert all(step.retire for step in result.steps)


def test_a_three_cycle_is_broken_at_exactly_one_delete():
    notes = [LiveNote("x.md", "c"), LiveNote("y.md", "a"), LiveNote("z.md", "b")]
    rows = [retired("x.md", "a"), retired("y.md", "b"), retired("z.md", "c")]
    result = plan(notes, rows, files={"x.md": "a", "y.md": "b", "z.md": "c"})
    assert len(result.steps) == 3
    assert [step.breaks_cycle for step in result.steps] == [True, False, False]
    # After the break each step's bytes belong to the note written just before it.
    for earlier, later in zip(result.steps, result.steps[1:]):
        assert later.requires == {earlier.path}


def test_a_chain_writes_the_occupants_new_path_first():
    """A folder moves up into its parent: a/b/x -> a/x, a/b/b/x -> a/b/x."""
    notes = [LiveNote("a/x.md", "upper"), LiveNote("a/b/x.md", "lower")]
    rows = [retired("a/b/x.md", "upper"), retired("a/b/b/x.md", "lower")]
    result = plan(notes, rows, files={"a/b/x.md": "upper", "a/b/b/x.md": "lower"})
    assert kinds(result) == [
        (StepKind.write, "a/x.md"),
        (StepKind.replace, "a/b/x.md"),
        (StepKind.delete, "a/b/b/x.md"),
    ]
    assert not any(step.breaks_cycle for step in result.steps)


def test_a_move_onto_a_freed_path_has_no_gap():
    """Source -> moved/Source, Target -> notes/Source in one commit."""
    notes = [LiveNote("moved/Source.md", "s"), LiveNote("notes/Source.md", "t")]
    rows = [retired("notes/Source.md", "s"), retired("notes/Target.md", "t")]
    result = plan(notes, rows, files={"notes/Source.md": "s", "notes/Target.md": "t"})
    assert kinds(result) == [
        (StepKind.write, "moved/Source.md"),
        (StepKind.replace, "notes/Source.md"),
        (StepKind.delete, "notes/Target.md"),
    ]
    assert not any(step.breaks_cycle for step in result.steps)


def test_a_move_back_with_the_same_bytes_keeps_the_file_and_settles_the_row():
    """KEEP must also SETTLE (C2); a synced keeper only completes the row."""
    rows = [retired("a.md", "a")]
    pending = plan([LiveNote("a.md", "a")], rows, files={"a.md": "a"})
    assert kinds(pending) == [(StepKind.settle, "a.md")] and step(pending, "a.md").retire
    synced = plan([LiveNote("a.md", "a", synced=True)], rows, files={"a.md": "a"})
    assert kinds(synced) == [(StepKind.complete, "a.md")]


def test_a_free_write_at_a_retired_path_completes_its_row():
    """C3: after a crash between the delete and the write, the row must not linger."""
    result = plan([LiveNote("p.md", "new")], [retired("p.md", "old")])
    assert kinds(result) == [(StepKind.write, "p.md")] and step(result, "p.md").retire


def test_a_lingering_row_on_the_notes_own_file_is_an_update_not_an_editor_change():
    """C3: disk holds the file checksum of the note now living there."""
    result = plan(
        [LiveNote("p.md", "edited", file_checksum="mine")],
        [retired("p.md", "old")],
        files={"p.md": "mine"},
    )
    assert kinds(result) == [(StepKind.write, "p.md")] and step(result, "p.md").retire
    assert not result.conflicts


# --- Editors' files: never deleted, and nobody's only copy is lost ---


def test_editor_bytes_at_an_old_path_block_only_unknown_owners():
    notes = [LiveNote("new/a.md", "a"), LiveNote("new/b.md", "b")]
    rows = [retired("old/a.md", "a"), retired("old/b.md", "b-old"), retired("old/c.md", "c")]
    files = {"old/a.md": "a", "old/b.md": "b-old", "old/c.md": "private"}
    result = plan(notes, rows, files=files)
    assert result.conflicts == (("old/c.md", EDITOR_CHANGE),)
    # old/a.md belongs to new/a.md and goes right after it (C11); old/b.md's owner is
    # unknown (its bytes changed and no base is known), so it waits for a clean pass.
    assert kinds(result) == [
        (StepKind.write, "new/a.md"),
        (StepKind.delete, "old/a.md"),
        (StepKind.write, "new/b.md"),
        (StepKind.delete, "old/b.md"),
    ]
    assert step(result, "old/b.md").final and not step(result, "old/a.md").final


def test_editor_bytes_at_a_reused_path_delete_nothing():
    result = plan(
        [LiveNote("p.md", "new")],
        [retired("p.md", "old")],
        files={"p.md": "private"},
    )
    assert result.steps == () and result.conflicts == (("p.md", EDITOR_CHANGE),)


def test_a_swap_with_editor_bytes_at_one_end_deletes_nothing():
    """C4: B's only copy (at p2) must survive when A's old path holds an editor's file."""
    notes = [LiveNote("p1.md", "b"), LiveNote("p2.md", "a")]
    rows = [retired("p1.md", "a"), retired("p2.md", "b")]
    occupants = {"p1.md": Occupant("p2.md"), "p2.md": Occupant("p1.md")}
    result = plan(notes, rows, files={"p1.md": "private", "p2.md": "b"}, occupants=occupants)
    assert result.steps == ()
    assert result.conflicts == (("p1.md", EDITOR_CHANGE),)


def test_a_three_cycle_blocked_by_an_editor_deletes_nothing():
    notes = [LiveNote("x.md", "c"), LiveNote("y.md", "a"), LiveNote("z.md", "b")]
    rows = [retired("x.md", "a"), retired("y.md", "b"), retired("z.md", "c")]
    result = plan(notes, rows, files={"x.md": "a", "y.md": "private", "z.md": "c"})
    assert result.steps == ()


def test_foreign_bytes_at_a_new_path_are_a_conflict_without_the_materializer():
    """C9: the materializer would stamp external_change_detected; it is never called."""
    result = plan([LiveNote("new.md", "n")], files={"new.md": "foreign"})
    assert result.steps == () and result.conflicts == (("new.md", FOREIGN_FILE),)


def test_the_notes_own_earlier_file_is_overwritten_by_the_materializer():
    for note in (
        LiveNote("n.md", "new", file_checksum="old"),
        LiveNote("n.md", "new", sync_checksum="old"),
    ):
        assert kinds(plan([note], files={"n.md": "old"})) == [(StepKind.write, "n.md")]


def test_a_note_sql_has_not_reached_is_skipped_and_keeps_its_old_copy():
    """C6: one stale note does not stop the others."""
    notes = [LiveNote("new/a.md", "a", ready=False), LiveNote("new/b.md", "b")]
    rows = [retired("old/a.md", "a"), retired("old/b.md", "b")]
    result = plan(notes, rows, files={"old/a.md": "a", "old/b.md": "b"})
    assert result.stale == {"new/a.md"}
    assert kinds(result) == [(StepKind.write, "new/b.md"), (StepKind.delete, "old/b.md")]


# --- Who owns old bytes (C4) ---


def test_occupants_come_from_bytes_never_from_a_stale_base():
    notes = [LiveNote("new/a.md", "a2"), LiveNote("new/b.md", "b2", file_checksum="b1")]
    rows = [retired("old/a.md"), retired("old/b.md"), retired("old/c.md")]
    disk = DiskView(files={"old/a.md": "a1", "old/b.md": "b1", "old/c.md": "c-written-later"})
    # No base: a moved-and-edited note is unknown; a local move keeps its file checksum.
    assert identify_occupants(notes, rows, disk) == {"old/b.md": Occupant("new/b.md")}
    base = {
        "old/a.md": BaseFile("a1", "new/a.md"),
        "old/b.md": BaseFile("b1", "new/b.md"),
        # A superseded pass overwrote old/c.md since: the base no longer describes disk.
        "old/c.md": BaseFile("c1", None),
    }
    assert identify_occupants(notes, rows, disk, base) == {
        "old/a.md": Occupant("new/a.md"),
        "old/b.md": Occupant("new/b.md"),
    }
    # A base equal to disk names a deleted note as gone.
    gone = {"old/c.md": BaseFile("c-written-later", None)}
    assert identify_occupants(notes, rows, disk, gone)["old/c.md"] == Occupant(None)


def test_an_unknown_owner_and_everything_behind_it_waits_for_the_end():
    notes = [LiveNote("a.md", "a"), LiveNote("z/b.md", "b")]
    rows = [retired("old.md", "x"), retired("a.md", "y")]
    files = {"old.md": "x", "a.md": "y"}
    result = plan(notes, rows, files=files, occupants={"old.md": Occupant("a.md")})
    # a.md's old bytes have an unknown owner, so a.md and old.md (whose owner is the
    # note written at a.md) both wait; z/b.md does not.
    assert kinds(result)[0] == (StepKind.write, "z/b.md")
    assert {step.path for step in result.steps if step.final} == {"a.md", "old.md"}
    assert kinds(result)[1:] == [(StepKind.replace, "a.md"), (StepKind.delete, "old.md")]


# --- Files and folders trading places (C8) ---


def test_a_file_where_a_folder_must_be_goes_first():
    notes = [LiveNote("a.md/b.md", "b"), LiveNote("moved.md", "a")]
    rows = [retired("a.md", "a")]
    result = plan(notes, rows, files={"a.md": "a"}, blocking_files={"a.md/b.md": ("a.md",)})
    assert kinds(result) == [
        (StepKind.write, "moved.md"),
        (StepKind.delete, "a.md"),
        (StepKind.write, "a.md/b.md"),
    ]
    assert step(result, "a.md/b.md").clears == {"a.md"}


def test_a_note_moving_into_a_folder_named_like_itself_is_a_cycle():
    notes = [LiveNote("a.md/b.md", "a")]
    rows = [retired("a.md", "a")]
    result = plan(notes, rows, files={"a.md": "a"}, blocking_files={"a.md/b.md": ("a.md",)})
    assert kinds(result) == [(StepKind.delete, "a.md"), (StepKind.write, "a.md/b.md")]
    assert step(result, "a.md").breaks_cycle


def test_a_folder_where_a_file_must_be_is_emptied_first():
    notes = [LiveNote("a.md", "x"), LiveNote("elsewhere.md", "y")]
    rows = [retired("a.md/b.md", "y")]
    result = plan(
        notes,
        rows,
        files={"a.md/b.md": "y"},
        directories=frozenset({"a.md"}),
        folder_files={"a.md": ("a.md/b.md",)},
    )
    assert kinds(result) == [
        (StepKind.write, "elsewhere.md"),
        (StepKind.delete, "a.md/b.md"),
        (StepKind.write, "a.md"),
    ]


def test_a_folder_or_file_basic_memory_does_not_own_blocks_the_path():
    unknown_file = plan(
        [LiveNote("a.md/b.md", "b")], files={"a.md": "?"}, blocking_files={"a.md/b.md": ("a.md",)}
    )
    assert unknown_file.conflicts == (("a.md/b.md", BLOCKED_PATH),)
    foreign = plan(
        [LiveNote("a.md", "x")], directories=frozenset({"a.md"}), folder_foreign=frozenset({"a.md"})
    )
    assert foreign.conflicts == (("a.md", BLOCKED_PATH),)
    stray = plan(
        [LiveNote("a.md", "x")],
        files={"a.md/attachment.png": "png"},
        directories=frozenset({"a.md"}),
        folder_files={"a.md": ("a.md/attachment.png",)},
    )
    assert stray.conflicts == (("a.md", BLOCKED_PATH),) and stray.steps == ()


# --- Case- and Unicode-only renames (C1) ---


def test_an_old_spelling_that_aliases_the_live_file_is_never_deleted_blindly():
    result = plan([LiveNote("A.md", "a")], [retired("a.md", "a")], files={"A.md": "a", "a.md": "a"})
    old = step(result, "a.md")
    assert old.kind is StepKind.delete and old.aliases == ("A.md",)
    assert old.requires == {"A.md"}


def test_changed_bytes_behind_an_alias_are_replaced_not_reported():
    result = plan(
        [LiveNote("A.md", "a2")],
        [retired("a.md", "a1")],
        files={"A.md": "a1", "a.md": "a1"},
        aliases={"A.md": "a.md"},
        occupants={"a.md": Occupant("A.md")},
    )
    assert kinds(result) == [(StepKind.replace, "A.md"), (StepKind.delete, "a.md")]
    assert not result.conflicts and not step(result, "A.md").retire
    # The note at the alias owns the old bytes itself: it waits for nothing.
    assert step(result, "A.md").requires == frozenset() and not step(result, "A.md").final


def test_old_bytes_behind_an_alias_with_no_known_owner_wait_for_the_end():
    result = plan(
        [LiveNote("A.md", "a2"), LiveNote("z.md", "z")],
        [retired("a.md", "a1")],
        files={"A.md": "a1", "a.md": "a1"},
        aliases={"A.md": "a.md"},
    )
    assert kinds(result) == [
        (StepKind.write, "z.md"),
        (StepKind.replace, "A.md"),
        (StepKind.delete, "a.md"),
    ]
    assert step(result, "A.md").final and step(result, "a.md").final


def alias_taken_by_another_note(archive: str) -> FileProjectionPlan:
    """U moves from notes/Source.md to archive/; a new note V takes notes/source.md.

    On macOS or Windows both spellings are one directory entry that holds U's old bytes.
    """
    notes = [
        LiveNote("archive/Source.md", "u", ready=archive != "stale", file_checksum="u"),
        LiveNote("notes/source.md", "v"),
    ]
    files = {"notes/Source.md": "u", "notes/source.md": "u"}
    if archive == "editor":
        files["archive/Source.md"] = "editor"
    return plan(
        notes,
        [retired("notes/Source.md", "u")],
        files=files,
        aliases={"notes/source.md": "notes/Source.md"},
    )


def test_a_different_note_at_an_alias_waits_for_the_moved_notes_new_copy():
    result = alias_taken_by_another_note("free")
    assert kinds(result) == [
        (StepKind.write, "archive/Source.md"),
        (StepKind.replace, "notes/source.md"),
        (StepKind.delete, "notes/Source.md"),
    ]
    assert step(result, "notes/source.md").requires == {"archive/Source.md"}
    assert step(result, "notes/Source.md").requires == {"archive/Source.md", "notes/source.md"}


@pytest.mark.parametrize("archive", ["editor", "stale"])
def test_a_different_note_at_an_alias_never_deletes_a_blocked_notes_only_copy(archive):
    result = alias_taken_by_another_note(archive)
    # The single entry holds U's only copy: neither spelling is touched.
    assert result.steps == ()
    if archive == "editor":
        assert result.conflicts == (("archive/Source.md", FOREIGN_FILE),)
    else:
        assert result.stale == {"archive/Source.md"}


def test_an_old_spelling_that_reads_the_live_file_is_renamed_not_reported():
    """The live note was written through the alias, then the pass died before the rename.

    The old spelling now reads the new bytes, which its row never expected.
    """
    notes = [LiveNote("notes/source.md", "n", synced=True)]
    rows = [retired("notes/Source.md", "o")]
    view = DiskView(
        files={"notes/source.md": "n", "notes/Source.md": "n"},
        aliases={"notes/source.md": "notes/Source.md"},
    )
    assert publication_proof_candidates(notes, rows, view) == ()
    result = plan_file_projection(notes, rows, view, identify_occupants(notes, rows, view))
    assert not result.conflicts and result.done == {"notes/source.md"}
    old = step(result, "notes/Source.md")
    assert old.kind is StepKind.delete and old.aliases == ("notes/source.md",)
    assert old.requires == {"notes/source.md"} and not old.final


# --- Bytes Basic Memory published but SQL does not explain (C7) ---


def test_published_bytes_nothing_explains_are_proven_then_replaced():
    notes = [LiveNote("x.md", "x2", file_checksum="x0")]
    rows = [retired("old.md", "o2", "o0")]
    disk = DiskView(files={"x.md": "x1", "old.md": "o1"})
    assert publication_proof_candidates(notes, rows, disk) == ("old.md", "x.md")
    proven = DiskView(files=disk.files, published=frozenset({"x.md", "old.md"}))
    result = plan_file_projection(notes, rows, proven, {"old.md": Occupant(None)})
    assert kinds(result) == [(StepKind.delete, "old.md"), (StepKind.replace, "x.md")]
    unproven = plan_file_projection(notes, rows, disk, {})
    assert unproven.conflicts == (("old.md", EDITOR_CHANGE), ("x.md", FOREIGN_FILE))


def test_explained_bytes_need_no_proof():
    notes = [
        LiveNote("a.md", "a", file_checksum="a0"),
        LiveNote("b.md", "b", ready=False),
        LiveNote("c.md", "c"),
        # Another note's old bytes at a reused path, and the old file behind an alias.
        LiveNote("d.md", "d"),
        LiveNote("E.md", "e"),
    ]
    disk = DiskView(
        files={
            "a.md": "a0",
            "b.md": "anything",
            "c.md": "c",
            "r.md": "r",
            "d.md": "old-d",
            "E.md": "e0",
            "e.md": "e0",
        },
        aliases={"E.md": "e.md"},
    )
    rows = [retired("r.md", "r"), retired("d.md", "old-d"), retired("e.md", "e0")]
    assert publication_proof_candidates(notes, rows, disk) == ()


def test_groups_are_found_across_finished_ones():
    """Tarjan's walk meets a group it already finished from a later root."""
    from basic_memory.index.git_file_projection_plan import _strongly_connected

    groups = _strongly_connected({"a": ["b"], "b": ["a"], "c": ["b", "d"], "d": ["c"]})
    assert sorted(sorted(group) for group in groups) == [["a", "b"], ["c", "d"]]


def test_a_cycle_waiting_on_another_cycle_is_broken_after_that_one():
    """Only a cycle that waits on nothing outside itself is broken first."""
    notes = [
        LiveNote("f.md", "one"),  # moved out of the folder f.md/ into a file named f.md
        LiveNote("p1.md", "b"),
        LiveNote("p2.md", "a"),
    ]
    rows = [
        retired("f.md/1.md", "one"),
        # f.md/2.md holds A's older bytes: it waits for A at p2.md, which is in the swap.
        retired("f.md/2.md", "a-old"),
        retired("p1.md", "a"),
        retired("p2.md", "b"),
    ]
    files = {"f.md/1.md": "one", "f.md/2.md": "a-old", "p1.md": "a", "p2.md": "b"}
    occupants = {
        "f.md/1.md": Occupant("f.md"),
        "f.md/2.md": Occupant("p2.md"),
        "p1.md": Occupant("p2.md"),
        "p2.md": Occupant("p1.md"),
    }
    result = plan(
        notes,
        rows,
        files=files,
        occupants=occupants,
        directories=frozenset({"f.md"}),
        folder_files={"f.md": ("f.md/1.md", "f.md/2.md")},
    )
    assert kinds(result) == [
        (StepKind.replace, "p1.md"),
        (StepKind.replace, "p2.md"),
        (StepKind.delete, "f.md/2.md"),
        (StepKind.delete, "f.md/1.md"),
        (StepKind.write, "f.md"),
    ]
    assert [step.path for step in result.steps if step.breaks_cycle] == ["p1.md", "f.md/1.md"]
