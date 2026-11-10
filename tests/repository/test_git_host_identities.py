"""Who owns a note UUID across one Git host (I1): real SQL rows and real Git histories."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from basic_memory.models import Entity, Project
from basic_memory.repository.git_host_identities import (
    GitHostIdentities,
    GitHostProject,
    GitNoteOwnedElsewhere,
)
from basic_memory.repository.git_project_repository import GitProjectRepository


@dataclass
class Host:
    identities: GitHostIdentities
    projects: dict[str, GitHostProject]

    def repo(self, name: str) -> GitProjectRepository:
        return self.projects[name].repository

    def uuid(self, name: str) -> UUID:
        return self.projects[name].external_id


@pytest.fixture
async def host(test_project: Project, session_maker, tmp_path: Path) -> Host:
    """Projects a (test_project), b and c, each with its own bare repository."""
    projects: dict[str, GitHostProject] = {}
    async with session_maker() as session, session.begin():
        rows = {"a": test_project}
        for name in ("b", "c"):
            rows[name] = Project(name=name, path=str(tmp_path / name), is_active=True)
            session.add(rows[name])
        await session.flush()
        for name, row in rows.items():
            projects[name] = GitHostProject(
                row.id,
                UUID(row.external_id),
                name,
                GitProjectRepository.create(
                    tmp_path / f"{name}.git", project_external_id=UUID(row.external_id)
                ),
            )
    return Host(GitHostIdentities(projects.values()), projects)


def publish(
    repository: GitProjectRepository,
    *,
    add: dict[str, UUID] | None = None,
    remove: tuple[UUID, ...] = (),
    moved: dict[UUID, UUID] | None = None,
) -> str:
    base = repository.snapshot()
    note_ids = {
        path: note for path, note in base.manifest.note_ids.items() if note not in remove
    } | (add or {})
    prepared = repository.prepare(
        transaction_id=uuid4(),
        base_commit=base.commit_id,
        files={path: f"bytes of {path}" for path in note_ids},
        note_ids=note_ids,
        moved_notes=moved,
    )
    return repository.publish(prepared).commit_id


async def own_in_sql(session_maker, project_id: int, note: UUID) -> None:
    async with session_maker() as session, session.begin():
        session.add(
            Entity(
                project_id=project_id,
                external_id=str(note),
                title=str(note),
                note_type="note",
                content_type="text/markdown",
                file_path=f"{note}.md",
            )
        )


async def classify(host: Host, session_maker, claimant: str):
    repository = host.repo(claimant)
    async with session_maker() as session:
        return await host.identities.classify_foreign(
            session,
            project_id=host.projects[claimant].project_id,
            manifest=repository.snapshot().manifest,
            history=(repository, repository.head()),
        )


@pytest.mark.asyncio
async def test_sql_lag_waits_and_an_unproven_twin_is_a_duplicate(host, session_maker):
    released, held = uuid4(), uuid4()
    a = host.projects["a"]
    await own_in_sql(session_maker, a.project_id, released)
    await own_in_sql(session_maker, a.project_id, held)
    publish(host.repo("a"), add={"held.md": held})
    publish(host.repo("b"), add={"released.md": released, "held.md": held})
    foreign = await classify(host, session_maker, "b")
    # Rule (a): a's head no longer holds `released`; only its SQL lags behind.
    assert foreign.deferred == {released: a.project_id}
    # Rule (c): a's head still holds `held`, and b holds it without any move proof.
    assert foreign.duplicates == {held: a.project_id}
    message = host.identities.duplicates_message("Already owned", foreign)
    assert message == f"Already owned: {held} (owned by project 'a' ({host.uuid('a')}))"
    deferral = host.identities.deferral(foreign)
    assert str(deferral).startswith("Deferred: waiting for project 'a'")
    assert deferral.owner_project_ids == {a.project_id}


@pytest.mark.asyncio
async def test_a_proven_move_waits_while_the_source_head_still_holds_the_note(host, session_maker):
    note = uuid4()
    await own_in_sql(session_maker, host.projects["a"].project_id, note)
    publish(host.repo("a"), add={"n.md": note})
    publish(host.repo("b"), add={"n.md": note}, moved={note: host.uuid("a")})
    foreign = await classify(host, session_maker, "b")
    assert foreign.deferred == {note: host.projects["a"].project_id}
    assert not foreign.duplicates


@pytest.mark.asyncio
async def test_an_old_trailer_does_not_cover_a_later_reintroduction(host, session_maker):
    """Round trip a->b->a, then the note is put into b again without a move: duplicate."""
    note = uuid4()
    a, b = host.repo("a"), host.repo("b")
    publish(a, add={"n.md": note})
    publish(a, remove=(note,))
    publish(b, add={"n.md": note}, moved={note: host.uuid("a")})
    publish(b, remove=(note,))
    publish(a, add={"n.md": note}, moved={note: host.uuid("b")})
    await own_in_sql(session_maker, host.projects["a"].project_id, note)
    publish(b, add={"again.md": note})
    foreign = await classify(host, session_maker, "b")
    assert foreign.duplicates == {note: host.projects["a"].project_id}


@pytest.mark.asyncio
async def test_a_chain_waits_and_a_double_move_is_a_duplicate(host, session_maker):
    note = uuid4()
    a_id = host.projects["a"].project_id
    publish(host.repo("a"), add={"n.md": note})
    await own_in_sql(session_maker, a_id, note)
    # Chain a -> b -> c, seen before b and a synced: c names b, the owner a never took
    # the note by a move, so a will release it once its own move reaches it.
    publish(host.repo("c"), add={"n.md": note}, moved={note: host.uuid("b")})
    assert (await classify(host, session_maker, "c")).deferred == {note: a_id}


@pytest.mark.asyncio
async def test_two_hosts_moving_one_note_out_of_one_project_is_a_duplicate(host, session_maker):
    note = uuid4()
    b_id = host.projects["b"].project_id
    publish(host.repo("a"), add={"n.md": note})
    publish(host.repo("a"), remove=(note,))
    # This host moved the note a -> b; the peer moved it a -> c.
    publish(host.repo("b"), add={"n.md": note}, moved={note: host.uuid("a")})
    await own_in_sql(session_maker, b_id, note)
    publish(host.repo("c"), add={"n.md": note}, moved={note: host.uuid("a")})
    assert (await classify(host, session_maker, "c")).duplicates == {note: b_id}


@pytest.mark.asyncio
async def test_a_chain_through_a_project_the_note_left_is_followed(host, session_maker):
    """The owner a took the note from b long ago; c now names b as its source.

    Before b's newer history is here the claim is ambiguous (a and c both took it from b)
    and refused; once b shows that it took the note from a again, the chain is proven.
    """
    note = uuid4()
    a, b = host.repo("a"), host.repo("b")
    a_id = host.projects["a"].project_id
    publish(b, add={"n.md": note})
    publish(b, remove=(note,))
    publish(a, add={"n.md": note}, moved={note: host.uuid("b")})
    await own_in_sql(session_maker, a_id, note)
    publish(host.repo("c"), add={"n.md": note}, moved={note: host.uuid("b")})
    assert (await classify(host, session_maker, "c")).duplicates == {note: a_id}
    publish(b, add={"n.md": note}, moved={note: host.uuid("a")})
    publish(b, remove=(note,))
    assert (await classify(host, session_maker, "c")).deferred == {note: a_id}


@pytest.mark.asyncio
async def test_an_owner_unknown_to_the_host_is_a_duplicate(host, session_maker, tmp_path):
    note = uuid4()
    async with session_maker() as session, session.begin():
        stranger = Project(name="stranger", path=str(tmp_path / "s"), is_active=True)
        session.add(stranger)
        await session.flush()
        stranger_id = stranger.id
    await own_in_sql(session_maker, stranger_id, note)
    publish(host.repo("b"), add={"n.md": note}, moved={note: uuid4()})
    assert (await classify(host, session_maker, "b")).duplicates == {note: stranger_id}


@pytest.mark.asyncio
async def test_new_uuids_must_be_free_of_rows_heads_and_reservations(host, session_maker):
    in_sql, in_head, reserved, free = uuid4(), uuid4(), uuid4(), uuid4()
    b_id = host.projects["b"].project_id
    await own_in_sql(session_maker, host.projects["a"].project_id, in_sql)
    publish(host.repo("c"), add={"n.md": in_head})
    move = uuid4()
    host.identities.reservations[reserved] = move
    async with session_maker() as session:
        for note, owner in ((in_sql, "project 'a'"), (in_head, "project 'c'")):
            with pytest.raises(GitNoteOwnedElsewhere, match=owner):
                await host.identities.require_unowned(session, project_id=b_id, note_ids={note})
        with pytest.raises(GitNoteOwnedElsewhere, match=f"open cross-project move {move}"):
            await host.identities.require_unowned(session, project_id=b_id, note_ids={reserved})
        await host.identities.require_unowned(
            session, project_id=b_id, note_ids={reserved, free}, holder=move
        )
        await host.identities.require_unowned(session, project_id=b_id, note_ids=set())


@pytest.mark.asyncio
async def test_an_owner_wakes_only_the_projects_waiting_for_it(host):
    woken: list[str] = []
    ids = {name: project.project_id for name, project in host.projects.items()}
    for name, project_id in ids.items():
        host.identities.register(project_id, lambda name=name: woken.append(name))
    host.identities.wait_for(ids["b"], {ids["a"]})
    host.identities.wait_for(ids["c"], {ids["b"]})
    host.identities.settled(ids["a"])
    assert woken == ["b"]
    # b completed too: it waits no more, and c, which waits for b, is woken.
    host.identities.settled(ids["b"])
    assert woken == ["b", "c"]
    host.identities.nudge_waiting()
    assert woken == ["b", "c", "c"]
    assert host.identities.describe(-1) == "another project"


@pytest.mark.asyncio
async def test_custody_skips_unknown_and_repeated_sources(host, session_maker):
    """A chain that names a project this host does not know, or loops, ends cleanly."""
    looping, unknown, stranger = uuid4(), uuid4(), uuid4()
    a_id = host.projects["a"].project_id
    publish(host.repo("a"), add={"l.md": looping, "u.md": unknown})
    await own_in_sql(session_maker, a_id, looping)
    await own_in_sql(session_maker, a_id, unknown)
    # b names c for `looping` and an unknown project for `unknown`; c names b for both.
    publish(
        host.repo("b"),
        add={"l.md": looping, "u.md": unknown},
        moved={looping: host.uuid("c"), unknown: stranger},
    )
    publish(host.repo("b"), remove=(looping, unknown))
    publish(
        host.repo("c"),
        add={"l.md": looping, "u.md": unknown},
        moved={looping: host.uuid("b"), unknown: host.uuid("b")},
    )
    # No chain reaches a, and a took neither by a move: both chains are still travelling.
    assert (await classify(host, session_maker, "c")).deferred == {
        looping: a_id,
        unknown: a_id,
    }


@pytest.mark.asyncio
async def test_an_empty_manifest_has_no_foreign_notes(host, session_maker):
    foreign = await classify(host, session_maker, "b")
    assert foreign.deferred == {} and foreign.duplicates == {}
    async with session_maker() as session:
        assert await host.identities.sql_owners(session, 1, []) == {}


def test_full_caches_start_over():
    from basic_memory.repository import git_host_identities as module

    cache = {str(index): index for index in range(module._INTRODUCTION_CACHE_LIMIT)}
    GitHostIdentities._remember(cache, {"new": -1})
    assert cache == {"new": -1}
