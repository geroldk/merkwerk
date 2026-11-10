"""API controls must not escape paths, forge evidence or half-bind a source."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from basic_memory.schemas.git_transactions import (
    GitTransactionCommit,
    GitTransactionDelete,
    GitTransactionMove,
    GitTransactionStart,
    GitTransactionWrite,
)

_NOTE = UUID("8d0c6f0e-2f4b-4c8a-9a51-3f6e7d2b1c01")
_SOURCE = UUID("1b2c3d4e-5f60-4a7b-8c9d-0e1f2a3b4c5d")


@pytest.mark.parametrize(
    "data",
    [
        {"source_transaction_id": uuid4()},
        {"source_key": "secret"},
        {"source_revision": 0},
    ],
)
def test_transfer_source_needs_all_three_controls(data):
    with pytest.raises(ValidationError, match="together"):
        GitTransactionStart(**data)


@pytest.mark.parametrize(
    "message",
    [
        " ",
        "x\0y",
        "Oops\n\nBasic-Memory-Transaction: forged",
        # A move trailer is evidence too: mid-line, after blanks (Git reads such a line as
        # the continuation of the trailer above it) and in any case (Git ignores case).
        # Fixed UUIDs: a parameter is its own test id, and xdist workers must collect
        # identical ids.
        f"Move x Basic-Memory-Moved-Note: {_NOTE} from {_SOURCE}",
        f"Move\n\n Basic-Memory-Moved-Note: {_NOTE} from {_SOURCE}",
        f"Move\n\nbasic-memory-moved-note: {_NOTE} from {_SOURCE}",
    ],
)
def test_commit_message_cannot_forge_a_different_terminal_transaction(message):
    with pytest.raises(ValidationError):
        GitTransactionCommit(expected_revision=0, message=message)


@pytest.mark.parametrize("path", ["../x.md", ".git/x.md", ".basic-memory/x.md", "/absolute.md"])
def test_all_draft_mutation_paths_are_validated(path):
    with pytest.raises(ValidationError):
        GitTransactionWrite(expected_revision=0, path=path, content="not accepted")
    with pytest.raises(ValidationError):
        GitTransactionDelete(expected_revision=0, path=path)
    with pytest.raises(ValidationError):
        GitTransactionMove(expected_revision=0, path="safe.md", destination=path)
