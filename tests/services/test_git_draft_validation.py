"""Draft validation parses each exact note text once; uniqueness is checked every time."""

import pytest

from basic_memory.markdown import EntityParser
from basic_memory.repository.git_transaction_repository import GitTransactionError
from basic_memory.services import git_transactions
from basic_memory.services.git_transactions import validate_git_draft


def note(title: str, permalink: str, body: str = "Body") -> str:
    return f"---\ntitle: {title}\ntype: note\npermalink: {permalink}\n---\n\n{body}\n"


@pytest.mark.asyncio
async def test_unchanged_notes_are_parsed_once_and_permalinks_stay_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(git_transactions, "_VALIDATED", {})
    parsed: list[str] = []
    parse = EntityParser.parse_markdown_content

    async def counting(self, **arguments):
        parsed.append(arguments["file_path"].as_posix())
        return await parse(self, **arguments)

    monkeypatch.setattr(EntityParser, "parse_markdown_content", counting)
    tree = {"a.md": note("A", "a"), "b.md": note("B", "b")}
    await validate_git_draft(tree, tmp_path)
    await validate_git_draft(tree, tmp_path)
    assert parsed == ["a.md", "b.md"]
    # A changed text is parsed again.
    await validate_git_draft({**tree, "b.md": note("B", "b", "Neu")}, tmp_path)
    assert parsed == ["a.md", "b.md", "b.md"]
    # Two known notes with one permalink are still refused.
    twin = {"a.md": tree["a.md"], "c.md": note("C", "a")}
    with pytest.raises(GitTransactionError, match="unique permalinks"):
        await validate_git_draft(twin, tmp_path)


@pytest.mark.asyncio
async def test_a_refused_note_is_never_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(git_transactions, "_VALIDATED", {})
    broken = {"a.md": "---\ntitle: A\n---\n\nNo type or permalink\n"}
    for _ in range(2):
        with pytest.raises(GitTransactionError, match="Canonical"):
            await validate_git_draft(broken, tmp_path)
    assert git_transactions._VALIDATED == {}


@pytest.mark.asyncio
async def test_the_memory_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(git_transactions, "_VALIDATED", {})
    monkeypatch.setattr(git_transactions, "_VALIDATED_LIMIT", 2)
    await validate_git_draft({"a.md": note("A", "a"), "b.md": note("B", "b")}, tmp_path)
    await validate_git_draft({"c.md": note("C", "c")}, tmp_path)
    assert len(git_transactions._VALIDATED) == 1
