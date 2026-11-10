"""Portable identity boundary stored with each Git project snapshot."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from basic_memory.utils import portable_path_key

GIT_PROJECT_MANIFEST_PATH = ".basic-memory/project.json"


def validate_note_path(path: str) -> None:
    parts = path.split("/")
    if (
        not path
        or any(part in {"", ".", ".."} for part in parts)
        or any(part.casefold() in {".git", ".basic-memory"} for part in parts)
        or "\\" in path
        or any(ord(character) < 32 for character in path)
        or PurePosixPath(path).suffix.casefold() != ".md"
        or ":" in path
    ):
        raise ValueError(f"Not a normalized project Markdown path: {path!r}")


class GitNoteIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    external_id: UUID
    path: str

    @field_validator("path")
    @classmethod
    def require_note_path(cls, value: str) -> str:
        validate_note_path(value)
        return value


class GitProjectManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    project_external_id: UUID
    notes: tuple[GitNoteIdentity, ...] = ()

    @model_validator(mode="after")
    def require_unique_notes(self) -> Self:
        if len({note.path for note in self.notes}) != len(self.notes):
            raise ValueError("Duplicate note path in Git project manifest")
        # Trigger: two paths differ only by case or Unicode normalization.
        # Why: macOS and Windows store them as one file, so one note would overwrite the
        #   other and the files stage could not tell an old path from a live one (C1).
        # Outcome: such a manifest is never published or projected.
        if len({portable_path_key(note.path) for note in self.notes}) != len(self.notes):
            raise ValueError(
                "Note paths in the Git project manifest differ only by case or "
                "Unicode form; they would be one file on macOS and Windows"
            )
        if len({note.external_id for note in self.notes}) != len(self.notes):
            raise ValueError("Duplicate note identity in Git project manifest")
        return self

    @property
    def note_ids(self) -> dict[str, UUID]:
        return {note.path: note.external_id for note in self.notes}

    @property
    def note_ids_by_uuid(self) -> dict[UUID, str]:
        return {note.external_id: note.path for note in self.notes}

    def canonical_json(self) -> str:
        data = self.model_dump(mode="json")
        data["notes"] = sorted(data["notes"], key=lambda note: note["path"])
        return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
