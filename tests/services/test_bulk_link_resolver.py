"""Tests for bulk exact link resolution used by relation repair."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from basic_memory import db
from basic_memory.config import BasicMemoryConfig
from basic_memory.indexing.models import RelationTargetRequest
from basic_memory.models import Entity, Project
from basic_memory.repository import EntityRepository, ProjectRepository
from basic_memory.services.bulk_link_resolver import (
    BulkLinkResolver,
    ProjectEntityIdentityIndex,
    RelationTargetReference,
)
from basic_memory.services.exceptions import AmbiguousIdentifierError
from basic_memory.services.link_resolver import LinkResolver


def note_permalink(project: Project, path: str, app_config: BasicMemoryConfig) -> str:
    """Return the configured canonical permalink for a test note."""
    if app_config.permalinks_include_project:
        return f"{project.permalink}/{path}"
    return path


def test_relation_target_reference_keeps_incomplete_project_paths_local() -> None:
    target = RelationTargetReference.parse("/")

    assert target.project_path() == (None, "/")


@pytest_asyncio.fixture
async def bulk_entities(
    entity_repository: EntityRepository,
    test_project: Project,
    session_maker,
    app_config: BasicMemoryConfig,
) -> list[Entity]:
    """Create exact, ambiguous, and regular-file targets for bulk resolution."""
    now = datetime.now(timezone.utc)
    entities = [
        Entity(
            title="Auth Service",
            note_type="component",
            content_type="text/markdown",
            file_path="components/Auth Service.md",
            permalink=note_permalink(test_project, "components/auth-service", app_config),
            created_at=now,
            updated_at=now,
            project_id=test_project.id,
        ),
        Entity(
            title="Core Service",
            note_type="component",
            content_type="text/markdown",
            file_path="components/Core Service.md",
            permalink=note_permalink(test_project, "components/core-service", app_config),
            created_at=now,
            updated_at=now,
            project_id=test_project.id,
        ),
        Entity(
            title="Core Service",
            note_type="component",
            content_type="text/markdown",
            file_path="archive/Core Service.md",
            permalink=note_permalink(test_project, "archive/core-service", app_config),
            created_at=now,
            updated_at=now,
            project_id=test_project.id,
        ),
        Entity(
            title="image.png",
            note_type="file",
            content_type="image/png",
            file_path="assets/image.png",
            permalink=None,
            created_at=now,
            updated_at=now,
            project_id=test_project.id,
        ),
    ]
    async with db.scoped_session(session_maker) as session:
        for entity in entities:
            await entity_repository.add(session, entity)
    return entities


@pytest.mark.asyncio
async def test_bulk_resolution_matches_regular_strict_resolution(
    entity_repository: EntityRepository,
    search_service,
    session_maker,
    app_config: BasicMemoryConfig,
    test_project: Project,
    bulk_entities: list[Entity],
) -> None:
    """The bulk flavor preserves the regular resolver's exact-match contract."""
    regular_resolver = LinkResolver(
        entity_repository,
        search_service,
        session_maker,
        app_config,
    )
    bulk_resolver = BulkLinkResolver(entity_repository, app_config)
    auth_service, _, _, image = bulk_entities
    assert auth_service.permalink is not None
    # A note whose permalink ends in a file suffix, filed in another folder; its title
    # differs from the link text, so only the routed permalink can name it.
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        suffixed = await entity_repository.add(
            session,
            Entity(
                title="Fix in tax_costs",
                note_type="note",
                content_type="text/markdown",
                file_path="archive/threads/Fix in tax_costs.py.md",
                permalink=note_permalink(test_project, "threads/fix-in-tax-costs.py", app_config),
                created_at=now,
                updated_at=now,
                project_id=test_project.id,
            ),
        )
    link_texts = [
        auth_service.external_id.upper(),
        auth_service.title,
        auth_service.permalink,
        auth_service.file_path,
        "components/Auth Service",
        image.title,
        "threads/fix-in-tax-costs.py",
        "Does Not Exist",
        "Core Service",
    ]

    async with db.scoped_session(session_maker) as session:
        bulk_results = await bulk_resolver.resolve_relation_targets(
            [RelationTargetRequest(link_text) for link_text in link_texts],
            session=session,
        )
        for link_text in link_texts[:-1]:
            regular_result = await regular_resolver.resolve_link(
                link_text,
                strict=True,
                load_relations=False,
                session=session,
            )
            bulk_result = bulk_results[RelationTargetRequest(link_text)]
            assert (bulk_result.id if bulk_result else None) == (
                regular_result.id if regular_result else None
            )
        # Agreement alone would also hold if both missed it.
        suffixed_result = bulk_results[RelationTargetRequest("threads/fix-in-tax-costs.py")]
        assert suffixed_result is not None and suffixed_result.id == suffixed.id

        with pytest.raises(AmbiguousIdentifierError):
            await regular_resolver.resolve_link(
                "Core Service",
                strict=True,
                load_relations=False,
                session=session,
            )

    assert bulk_results[RelationTargetRequest("Core Service")] is None


@pytest.mark.asyncio
async def test_bulk_and_regular_resolution_agree_on_file_suffixed_identifiers(
    entity_repository: EntityRepository,
    search_service,
    session_maker,
    app_config: BasicMemoryConfig,
    test_project: Project,
) -> None:
    """Routed permalinks after file paths, a resource first (#1629), titles guarded (#1148)."""
    now = datetime.now(timezone.utc)
    rows = {
        "resource": ("foo.txt", "notes/foo.txt", None),
        "spelled like the resource": ("Foo note", "notes/foo.txt.md", "notes/foo.txt"),
        "moved": ("Fix in tax_costs", "archive/threads/Fix.py.md", "threads/fix-in-tax-costs.py"),
        "mirror": ("Mirror", "mirror/korpus-zeilen.py.md", "mirror/korpus-zeilen.py"),
        "duplicate owning the permalink": ("threads/dup.py", "archive/one.md", "threads/dup.py"),
        "duplicate": ("threads/dup.py", "archive/two.md", "archive/two"),
    }
    entities: dict[str, Entity] = {}
    async with db.scoped_session(session_maker) as session:
        for name, (title, file_path, permalink) in rows.items():
            entities[name] = await entity_repository.add(
                session,
                Entity(
                    title=title,
                    note_type="note" if permalink else "file",
                    content_type="text/markdown" if permalink else "text/plain",
                    file_path=file_path,
                    permalink=(
                        note_permalink(test_project, permalink, app_config) if permalink else None
                    ),
                    created_at=now,
                    updated_at=now,
                    project_id=test_project.id,
                ),
            )
    expected = {
        "notes/foo.txt": "resource",
        "threads/fix-in-tax-costs.py": "moved",
        f"{test_project.permalink}/threads/fix-in-tax-costs.py": "moved",
        "mirror/korpus-zeilen.py": "mirror",
        "threads/dup.py": None,
    }
    regular_resolver = LinkResolver(entity_repository, search_service, session_maker, app_config)

    async with db.scoped_session(session_maker) as session:
        bulk_results = await BulkLinkResolver(
            entity_repository, app_config
        ).resolve_relation_targets(
            [RelationTargetRequest(link_text) for link_text in expected], session=session
        )
        for link_text, name in expected.items():
            bulk = bulk_results[RelationTargetRequest(link_text)]
            if name is None:
                assert bulk is None, link_text
                with pytest.raises(AmbiguousIdentifierError):
                    await regular_resolver.resolve_link(link_text, strict=True, session=session)
                continue
            regular = await regular_resolver.resolve_link(
                link_text, strict=True, load_relations=False, session=session
            )
            assert regular is not None and regular.id == entities[name].id, link_text
            assert bulk is not None and bulk.id == entities[name].id, link_text


@pytest.mark.parametrize("workspace", [None, "team-paul"])
@pytest.mark.parametrize("include_project", [True, False])
def test_project_entity_index_routes_file_suffixed_permalinks_after_file_paths(
    test_project: Project, workspace: str | None, include_project: bool
) -> None:
    """Each configuration's stored form; workspace permalinks are always complete."""
    now = datetime.now(timezone.utc)
    stored = (
        f"{workspace}/{test_project.permalink}/"
        if workspace
        else f"{test_project.permalink}/"
        if include_project
        else ""
    )
    note = Entity(
        title="Fix",
        note_type="note",
        content_type="text/markdown",
        file_path="archive/Fix.py.md",
        permalink=f"{stored}threads/fix.py",
        external_id="00000000-0000-4000-8000-000000000001",
        created_at=now,
        updated_at=now,
        project_id=test_project.id,
    )
    index = ProjectEntityIdentityIndex.from_entities(test_project, [note])

    match = index.resolve_strict(
        "threads/fix.py", include_project_permalinks=include_project, workspace_permalink=workspace
    )

    assert match.entity is note


def test_project_entity_index_keeps_a_spelled_resource_ahead_of_the_routed_permalink(
    test_project: Project,
) -> None:
    """`notes/Foo.txt` names the colliding resources in another spelling, not the note."""
    now = datetime.now(timezone.utc)
    rows = [
        ("foo.txt", "notes/foo.txt", None),
        ("FOO.txt", "notes/FOO.txt", None),
        ("Foo note", "archive/Foo note.txt.md", f"{test_project.permalink}/notes/foo.txt"),
    ]
    entities = [
        Entity(
            title=title,
            note_type="note" if permalink else "file",
            content_type="text/markdown" if permalink else "text/plain",
            file_path=file_path,
            permalink=permalink,
            external_id=f"00000000-0000-4000-8000-00000000000{number}",
            created_at=now,
            updated_at=now,
            project_id=test_project.id,
        )
        for number, (title, file_path, permalink) in enumerate(rows)
    ]
    index = ProjectEntityIdentityIndex.from_entities(test_project, entities)

    match = index.resolve_strict(
        "notes/Foo.txt", include_project_permalinks=True, workspace_permalink=None
    )

    assert match.entity is None and not match.ambiguous


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", [None, "team-paul"])
@pytest.mark.parametrize("include_project", [True, False])
async def test_bulk_and_regular_resolution_keep_resources_ahead_of_routed_permalinks(
    entity_repository: EntityRepository,
    search_service,
    session_maker,
    app_config: BasicMemoryConfig,
    test_project: Project,
    workspace: str | None,
    include_project: bool,
) -> None:
    """Another spelling, or the other qualified form, of a resource path stays the resource."""
    from basic_memory.workspace_context import workspace_permalink_context

    project = test_project.permalink
    now = datetime.now(timezone.utc)
    rows = {
        "foo resource": ("foo.txt", "notes/foo.txt", None),
        "foo note": ("Foo note", "archive/Foo note.txt.md", f"{project}/notes/foo.txt"),
        "bar resource": ("bar.txt", "docs/bar.txt", None),
        "bar note": ("Bar note", "archive/Bar note.txt.md", f"team-paul/{project}/docs/bar.txt"),
    }
    entities: dict[str, Entity] = {}
    async with db.scoped_session(session_maker) as session:
        for name, (title, file_path, permalink) in rows.items():
            entities[name] = await entity_repository.add(
                session,
                Entity(
                    title=title,
                    note_type="note" if permalink else "file",
                    content_type="text/markdown" if permalink else "text/plain",
                    file_path=file_path,
                    permalink=permalink,
                    created_at=now,
                    updated_at=now,
                    project_id=test_project.id,
                ),
            )
    expected: dict[str, str | None] = {
        "notes/Foo.txt": None,
        f"{project}/docs/bar.txt": "bar resource",
    }
    if workspace:
        expected[f"{workspace}/{project}/notes/foo.txt"] = "foo resource"
    config = app_config.model_copy(update={"permalinks_include_project": include_project})
    regular_resolver = LinkResolver(entity_repository, search_service, session_maker, config)

    with workspace_permalink_context(workspace, "organization" if workspace else None):
        async with db.scoped_session(session_maker) as session:
            bulk_results = await BulkLinkResolver(
                entity_repository, config
            ).resolve_relation_targets(
                [RelationTargetRequest(link_text) for link_text in expected], session=session
            )
            for link_text, name in expected.items():
                regular = await regular_resolver.resolve_link(
                    link_text, strict=True, load_relations=False, session=session
                )
                bulk = bulk_results[RelationTargetRequest(link_text)]
                want = entities[name].id if name else None
                assert (regular.id if regular else None) == want, link_text
                assert (bulk.id if bulk else None) == want, link_text


@pytest.mark.asyncio
async def test_bulk_resolution_normalizes_file_paths(
    entity_repository: EntityRepository,
    session_maker,
    app_config: BasicMemoryConfig,
    bulk_entities: list[Entity],
) -> None:
    """Bulk file lookup uses the same Path normalization as EntityRepository."""
    image = bulk_entities[-1]
    now = datetime.now(timezone.utc)
    project_id = entity_repository.project_id
    assert project_id is not None
    custom_permalink_note = Entity(
        title="Custom Guide",
        note_type="note",
        content_type="text/markdown",
        file_path="docs/Guide.md",
        permalink="hand-picked-guide",
        created_at=now,
        updated_at=now,
        project_id=project_id,
    )
    root_filename_note = Entity(
        title="Alpha — a descriptive human title",
        note_type="note",
        content_type="text/markdown",
        file_path="alpha_note.md",
        permalink="descriptive-alpha",
        created_at=now,
        updated_at=now,
        project_id=project_id,
    )
    unicode_filename_note = Entity(
        title="A descriptive French school title",
        note_type="note",
        content_type="text/markdown",
        file_path="École_note.md",
        permalink="descriptive-school",
        created_at=now,
        updated_at=now,
        project_id=project_id,
    )
    async with db.scoped_session(session_maker) as session:
        await entity_repository.add(session, custom_permalink_note)
        await entity_repository.add(session, root_filename_note)
        await entity_repository.add(session, unicode_filename_note)

    resolver = BulkLinkResolver(entity_repository, app_config)

    async with db.scoped_session(session_maker) as session:
        results = {
            request.link_text: entity
            for request, entity in (
                await resolver.resolve_relation_targets(
                    [
                        RelationTargetRequest(link_text)
                        for link_text in (
                            "./assets//image.png",
                            "docs/Guide",
                            "alpha_note",
                            "alpha-note",
                            "ALPHA-NOTE.MD",
                            "école-note",
                        )
                    ],
                    session=session,
                )
            ).items()
        }

    resolved_image = results["./assets//image.png"]
    assert resolved_image is not None
    assert resolved_image.id == image.id
    resolved_guide = results["docs/Guide"]
    assert resolved_guide is not None
    assert resolved_guide.id == custom_permalink_note.id
    for identifier in ("alpha_note", "alpha-note", "ALPHA-NOTE.MD"):
        resolved_root_note = results[identifier]
        assert resolved_root_note is not None
        assert resolved_root_note.id == root_filename_note.id
    resolved_unicode_note = results["école-note"]
    assert resolved_unicode_note is not None
    assert resolved_unicode_note.id == unicode_filename_note.id


def test_project_entity_index_declines_ambiguous_file_path_aliases(
    test_project: Project,
    bulk_entities: list[Entity],
) -> None:
    """Bulk resolution never chooses between colliding underscore/hyphen aliases."""
    now = datetime.now(timezone.utc)
    project_id = test_project.id
    aliases = [
        Entity(
            title="Hyphenated target",
            note_type="note",
            content_type="text/markdown",
            file_path="alpha-note.md",
            permalink="hyphenated-target",
            created_at=now,
            updated_at=now,
            project_id=project_id,
        ),
        Entity(
            title="Underscored target",
            note_type="note",
            content_type="text/markdown",
            file_path="alpha_note.md",
            permalink="underscored-target",
            created_at=now,
            updated_at=now,
            project_id=project_id,
        ),
    ]
    index = ProjectEntityIdentityIndex.from_entities(test_project, [*bulk_entities, *aliases])

    match = index.resolve_strict(
        "ALPHA-NOTE",
        include_project_permalinks=False,
        workspace_permalink=None,
    )

    assert match.entity is None


def test_project_entity_index_reports_title_derived_permalink_ambiguity(
    test_project: Project,
    bulk_entities: list[Entity],
) -> None:
    bulk_entities[1].permalink = "core-service"
    index = ProjectEntityIdentityIndex.from_entities(test_project, bulk_entities)

    match = index.resolve_strict(
        "Core Service",
        include_project_permalinks=False,
        workspace_permalink=None,
    )

    assert match.entity is None
    assert match.ambiguous


@pytest.mark.asyncio
async def test_bulk_resolution_routes_cross_project_targets(
    entity_repository: EntityRepository,
    session_maker,
    tmp_path,
    app_config: BasicMemoryConfig,
    bulk_entities: list[Entity],
) -> None:
    """Explicit and project/path references resolve from one multi-project snapshot."""
    project_repository = ProjectRepository()
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        other_project = await project_repository.create(
            session,
            {
                "name": "Other Project",
                "description": "Secondary project",
                "path": str(tmp_path / "other-project"),
                "is_active": True,
                "is_default": False,
            },
        )
        other_entity_repository = EntityRepository(project_id=other_project.id)
        target = await other_entity_repository.add(
            session,
            Entity(
                title="Cross Project Note",
                note_type="note",
                content_type="text/markdown",
                file_path="docs/Cross Project Note.md",
                permalink=note_permalink(
                    other_project,
                    "docs/cross-project-note",
                    app_config,
                ),
                created_at=now,
                updated_at=now,
                project_id=other_project.id,
            ),
        )

    resolver = BulkLinkResolver(
        entity_repository,
        app_config,
        project_repository=project_repository,
    )
    async with db.scoped_session(session_maker) as session:
        results = {
            request.link_text: entity
            for request, entity in (
                await resolver.resolve_relation_targets(
                    [
                        RelationTargetRequest(link_text)
                        for link_text in (
                            "other project::Cross Project Note",
                            "Other Project/docs/cross-project-note",
                            "missing::Cross Project Note",
                        )
                    ],
                    session=session,
                )
            ).items()
        }

    explicit_result = results["other project::Cross Project Note"]
    path_result = results["Other Project/docs/cross-project-note"]
    assert explicit_result is not None
    assert path_result is not None
    assert explicit_result.id == target.id
    assert path_result.id == target.id
    assert results["missing::Cross Project Note"] is None


@pytest.mark.asyncio
async def test_bulk_resolution_skips_database_work_for_an_empty_batch(
    entity_repository: EntityRepository,
    session_maker,
    app_config: BasicMemoryConfig,
) -> None:
    resolver = BulkLinkResolver(entity_repository, app_config)

    async with db.scoped_session(session_maker) as session:
        assert await resolver.resolve_relation_targets([], session=session) == {}


@pytest.mark.asyncio
async def test_bulk_resolution_requires_the_current_project_to_exist(
    session_maker,
    app_config: BasicMemoryConfig,
) -> None:
    resolver = BulkLinkResolver(EntityRepository(project_id=999_999), app_config)

    async with db.scoped_session(session_maker) as session:
        with pytest.raises(RuntimeError, match="Current project 999999 does not exist"):
            await resolver.resolve_relation_targets(
                [RelationTargetRequest("Target")], session=session
            )


@pytest.mark.asyncio
async def test_path_targets_resolve_from_the_note_that_carries_them(
    entity_repository: EntityRepository,
    test_project: Project,
    session_maker,
    app_config: BasicMemoryConfig,
) -> None:
    """The same authored path names a different file from each source note."""
    now = datetime.now(timezone.utc)
    entities = [
        Entity(
            title=title,
            note_type="note",
            content_type="text/markdown",
            file_path=file_path,
            permalink=permalink,
            created_at=now,
            updated_at=now,
            project_id=test_project.id,
        )
        for title, file_path, permalink in (
            ("Alpha Guide", "alpha/Guide.md", "alpha/guide"),
            ("Beta Guide", "beta/Guide.md", "beta/guide"),
            ("Shared", "Shared.md", "shared"),
        )
    ]
    async with db.scoped_session(session_maker) as session:
        for entity in entities:
            await entity_repository.add(session, entity)
    alpha_guide, beta_guide, shared = entities
    from_alpha = RelationTargetRequest("./Guide.md", source_path="alpha/notes.md")
    from_beta = RelationTargetRequest("./Guide.md", source_path="beta/notes.md")
    up_from_alpha = RelationTargetRequest("../Shared.md", source_path="alpha/notes.md")
    wikilink_from_beta = RelationTargetRequest("[[../Shared.md|the shared note]]", "beta/notes.md")
    rooted = RelationTargetRequest("/beta/Guide.md", source_path="alpha/notes.md")
    escaping = RelationTargetRequest("../../Shared.md", source_path="alpha/notes.md")
    no_source = RelationTargetRequest("../Shared.md")
    by_title = RelationTargetRequest("./guide", source_path="alpha/notes.md")

    async with db.scoped_session(session_maker) as session:
        results = await BulkLinkResolver(entity_repository, app_config).resolve_relation_targets(
            [
                from_alpha,
                from_beta,
                up_from_alpha,
                wikilink_from_beta,
                rooted,
                escaping,
                no_source,
                by_title,
            ],
            session=session,
        )

    assert {request: entity.id if entity else None for request, entity in results.items()} == {
        from_alpha: alpha_guide.id,
        from_beta: beta_guide.id,
        up_from_alpha: shared.id,
        wikilink_from_beta: shared.id,
        rooted: beta_guide.id,
        escaping: None,
        no_source: None,
        # A path names a file exactly; it never falls back to a permalink or title.
        by_title: None,
    }


@pytest.mark.asyncio
async def test_bulk_resolution_retries_own_project_qualified_targets(
    entity_repository: EntityRepository,
    search_service,
    session_maker,
    app_config: BasicMemoryConfig,
    test_project: Project,
    bulk_entities: list[Entity],
) -> None:
    """`[[<project>/assets/image.png]]` and `[[<project>/Title]]` resolve like LinkResolver."""
    auth_service, _, _, image = bulk_entities
    regular_resolver = LinkResolver(entity_repository, search_service, session_maker, app_config)
    bulk_resolver = BulkLinkResolver(entity_repository, app_config)
    link_texts = [
        f"{test_project.permalink}/{image.file_path}",
        f"{test_project.permalink}/{auth_service.title}",
    ]

    async with db.scoped_session(session_maker) as session:
        bulk_results = await bulk_resolver.resolve_relation_targets(
            [RelationTargetRequest(link_text) for link_text in link_texts],
            session=session,
        )
        for link_text, expected in zip(link_texts, (image, auth_service)):
            regular = await regular_resolver.resolve_link(
                link_text, strict=True, load_relations=False, session=session
            )
            bulk = bulk_results[RelationTargetRequest(link_text)]
            assert regular is not None and regular.id == expected.id, link_text
            assert bulk is not None and bulk.id == expected.id, link_text


@pytest.mark.asyncio
async def test_bulk_resolution_strips_workspace_and_project_prefixes(
    entity_repository: EntityRepository,
    session_maker,
    app_config: BasicMemoryConfig,
    test_project: Project,
    bulk_entities: list[Entity],
) -> None:
    """`[[<workspace>/<project>/assets/image.png]]` resolves on a workspace route."""
    from basic_memory.workspace_context import workspace_permalink_context

    image = bulk_entities[-1]
    link_text = f"team-paul/{test_project.permalink}/{image.file_path}"
    bulk_resolver = BulkLinkResolver(entity_repository, app_config)

    with workspace_permalink_context("team-paul", "organization"):
        async with db.scoped_session(session_maker) as session:
            results = await bulk_resolver.resolve_relation_targets(
                [RelationTargetRequest(link_text)], session=session
            )

    resolved = results[RelationTargetRequest(link_text)]
    assert resolved is not None and resolved.id == image.id
