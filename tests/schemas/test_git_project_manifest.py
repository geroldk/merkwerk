"""Validate the portable Git identity file, independently of local SQL IDs."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from basic_memory.schemas.git_project import GitNoteIdentity, GitProjectManifest


@pytest.mark.parametrize("duplicate", ["path", "identity"])
def test_duplicate_notes_are_rejected(duplicate: str) -> None:
    first = GitNoteIdentity(path="a.md", external_id=uuid4())
    second = GitNoteIdentity(
        path="a.md" if duplicate == "path" else "b.md",
        external_id=first.external_id if duplicate == "identity" else uuid4(),
    )
    with pytest.raises(ValidationError, match="Duplicate note"):
        GitProjectManifest(project_external_id=uuid4(), notes=(first, second))


@pytest.mark.parametrize(
    "first,second",
    [("a.md", "A.md"), ("Notes/x.md", "notes/x.md"), ("caf\u00e9.md", "cafe\u0301.md")],
)
def test_paths_that_one_file_system_would_merge_are_rejected(first: str, second: str) -> None:
    """macOS and Windows store these as one file; one note would overwrite the other (C1)."""
    notes = (
        GitNoteIdentity(path=first, external_id=uuid4()),
        GitNoteIdentity(path=second, external_id=uuid4()),
    )
    with pytest.raises(ValidationError, match="differ only by case or Unicode form"):
        GitProjectManifest(project_external_id=uuid4(), notes=notes)


def test_manifest_has_canonical_order_and_roundtrips_uuids() -> None:
    first = GitNoteIdentity(path="überblick.md", external_id=uuid4())
    second = GitNoteIdentity(path="a.md", external_id=uuid4())
    manifest = GitProjectManifest(project_external_id=uuid4(), notes=(first, second))
    reordered = GitProjectManifest(
        project_external_id=manifest.project_external_id, notes=(second, first)
    )
    assert manifest.canonical_json() == reordered.canonical_json()
    parsed = GitProjectManifest.model_validate_json(manifest.canonical_json())
    assert parsed.note_ids == manifest.note_ids
    assert parsed.project_external_id == manifest.project_external_id


@pytest.mark.parametrize("mutation", ["version", "extra", "invalid_uuid", "reserved_path"])
def test_unknown_or_invalid_manifest_is_rejected(mutation: str) -> None:
    data = {"format_version": 1, "project_external_id": str(uuid4()), "notes": []}
    if mutation == "version":
        data["format_version"] = 2
    elif mutation == "extra":
        data["local_root"] = "/private/host/path"
    elif mutation == "invalid_uuid":
        data["project_external_id"] = "not-a-uuid"
    else:
        data["notes"] = [{"external_id": str(uuid4()), "path": ".basic-memory/private.md"}]
    with pytest.raises(ValidationError):
        GitProjectManifest.model_validate_json(json.dumps(data))
