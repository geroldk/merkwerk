"""Durable cross-project move intents next to the target's bare repository (I1)."""

import stat
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from basic_memory.repository.git_move_intent_repository import (
    GitMoveIntentError,
    GitMoveIntentRepository,
)
from basic_memory.repository.git_project_repository import GitProjectRepository
from basic_memory.schemas.git_moves import GitMoveIntent, GitMoveIntentView


@pytest.fixture
def store(tmp_path: Path) -> GitMoveIntentRepository:
    project = uuid4()
    repository = GitProjectRepository.create(tmp_path / "target.git", project_external_id=project)
    return GitMoveIntentRepository(repository, project)


def intent(store: GitMoveIntentRepository, **changes: object) -> GitMoveIntent:
    values: dict[str, object] = {
        "move_id": uuid4(),
        "created_at": datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        "source_project_id": uuid4(),
        "source_path": "notes/a.md",
        "source_checksum": "c" * 64,
        "source_base_commit": "a" * 40,
        "target_project_id": store.project_id,
        "external_id": uuid4(),
        "target_path": "moved/a.md",
        "title": "A",
        "target_content": "---\ntitle: A\ntype: note\npermalink: moved/a\n---\n",
        "target_permalink": "moved/a",
        "transaction_id": uuid4(),
        "transaction_key": "secret",
    }
    return GitMoveIntent.model_validate({**values, **changes})


def test_an_intent_is_saved_privately_and_replaced_atomically(store):
    first = intent(store)
    store.save(first)
    path = store.path / f"{first.move_id}.json"
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    later = first.model_copy(update={"state": "source_deleted", "source_commit": "b" * 40})
    store.save(later)
    assert store.load(first.move_id) == later
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # No temporary file is left next to it.
    assert [p.name for p in store.path.iterdir()] == [path.name]
    store.remove(first.move_id)
    assert store.load_all() == []
    store.remove(first.move_id)  # removing twice is not an error


def test_intents_are_listed_oldest_first(store):
    older = intent(store, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = intent(store, created_at=datetime(2026, 2, 1, tzinfo=UTC))
    store.save(newer)
    store.save(older)
    assert store.load_all() == [older, newer]


def test_symlinks_and_foreign_files_are_refused(store, tmp_path):
    saved = intent(store)
    store.save(saved)
    # A stray file is an error, not skipped: the move it might be stays visible.
    (store.path / "notes.json").write_text("{}")
    with pytest.raises(GitMoveIntentError, match="Unknown file"):
        store.load_all()
    (store.path / "notes.json").unlink()
    # An intent of another target project is refused.
    foreign = intent(store, target_project_id=uuid4())
    store.save(foreign)
    with pytest.raises(GitMoveIntentError, match="another target"):
        store.load_all()
    store.remove(foreign.move_id)
    # A symlinked intent file or directory is refused.
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(saved.model_dump_json())
    linked = intent(store)
    (store.path / f"{linked.move_id}.json").symlink_to(elsewhere)
    with pytest.raises(GitMoveIntentError, match="symlink"):
        store.load(linked.move_id)
    with pytest.raises(GitMoveIntentError, match="symlink"):
        store.save(linked)
    with pytest.raises(GitMoveIntentError, match="symlink"):
        store.remove(linked.move_id)
    (store.path / f"{linked.move_id}.json").unlink()
    moved = store.path.rename(tmp_path / "real-moves")
    store.path.symlink_to(moved)
    with pytest.raises(GitMoveIntentError, match="symlinks"):
        store.move_ids()
    with pytest.raises(GitMoveIntentError, match="symlinks"):
        store.save(saved)


def test_an_intent_whose_name_differs_from_its_move_is_refused(store):
    saved = intent(store)
    other = uuid4()
    store.path.mkdir(mode=0o700)
    (store.path / f"{other}.json").write_text(saved.model_dump_json())
    with pytest.raises(GitMoveIntentError, match="another move"):
        store.load(other)
    with pytest.raises(KeyError):
        store.load(uuid4())


def test_the_intent_schema_is_strict_and_the_view_keeps_the_secrets(store):
    saved = intent(store)
    with pytest.raises(ValidationError):
        GitMoveIntent.model_validate({**saved.model_dump(), "unexpected": 1})
    with pytest.raises(ValidationError):
        intent(store, target_path="../escape.md")
    view = GitMoveIntentView.of(saved).model_dump()
    assert "target_content" not in view and "transaction_key" not in view
    assert view["transaction_id"] == saved.transaction_id


def test_an_intent_file_named_in_another_spelling_is_refused(store):
    saved = intent(store)
    store.save(saved)
    path = store.path / f"{saved.move_id}.json"
    path.rename(store.path / f"{str(saved.move_id).upper()}.json")
    with pytest.raises(GitMoveIntentError, match="Unknown file"):
        store.move_ids()
