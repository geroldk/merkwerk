"""Creation and modification times as durable frontmatter (BM-010).

In a Git-managed project the Markdown file is the only durable state; SQL rows are rebuilt
from it. Without `created` and `modified` in the frontmatter, every rebuild (restore, new
host, projection of a peer's history) dates each note with the projection time. These
helpers keep both keys in the accepted bytes:

* `created` is set once, from the first write (or from the entity row for notes that
  predate this rule) and never changed afterwards.
* `modified` is renewed by every write that changes the note, and left alone by a plain
  move or by a write that changes nothing but `created`/`modified` themselves.

Values are ISO 8601 strings with an explicit offset, which `entity_parser` reads canonically.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from typing import Any

from basic_memory.file_utils import (
    ParseError,
    has_frontmatter,
    parse_frontmatter,
    remove_frontmatter,
)
from basic_memory.services.note_preparation import (
    PreparedEntityMove,
    PreparedEntityWrite,
    rewrite_frontmatter_fields,
)

CREATED_FIELD = "created"
MODIFIED_FIELD = "modified"
TIME_FIELDS = frozenset({CREATED_FIELD, MODIFIED_FIELD})


def format_note_time(value: datetime) -> str:
    """Render an aware (or local-naive) datetime as ISO 8601 with offset, to the second."""
    return (value if value.tzinfo is not None else value.astimezone()).isoformat(timespec="seconds")


def _frontmatter(markdown: str) -> dict[str, Any] | None:
    if not has_frontmatter(markdown):
        return {}
    try:
        return parse_frontmatter(markdown)
    except ParseError:
        return None


def _time_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, datetime):
        return format_note_time(value)
    return value if isinstance(value, str) and value else None


def same_apart_from_times(first: str, second: str) -> bool:
    """True when two notes differ at most in their `created`/`modified` keys."""
    first_meta, second_meta = _frontmatter(first), _frontmatter(second)
    if first_meta is None or second_meta is None:
        return False
    strip = {k: v for k, v in first_meta.items() if k not in TIME_FIELDS}
    other = {k: v for k, v in second_meta.items() if k not in TIME_FIELDS}
    return strip == other and remove_frontmatter(first, strip=False) == remove_frontmatter(
        second, strip=False
    )


def wanted_note_times(
    new_markdown: str,
    *,
    now: datetime,
    current_markdown: str | None,
    fallback_created: datetime | None = None,
    fallback_modified: datetime | None = None,
) -> dict[str, str] | None:
    """The `created`/`modified` values `new_markdown` must carry, or None if unreadable.

    `current_markdown` is the accepted bytes before this write (None for a create).
    `fallback_*` are the entity row's times, used only for a note that has no such key yet.
    """
    new_meta = _frontmatter(new_markdown)
    current_meta = _frontmatter(current_markdown) if current_markdown is not None else {}
    if new_meta is None or current_meta is None:
        return None
    created = (
        _time_value(current_meta, CREATED_FIELD)
        or _time_value(new_meta, CREATED_FIELD)
        or format_note_time(fallback_created or now)
    )
    if current_markdown is None:
        # A new note keeps a modified time the writer supplied (an import), else now.
        modified = _time_value(new_meta, MODIFIED_FIELD) or format_note_time(now)
    elif same_apart_from_times(new_markdown, current_markdown):
        # Nothing but the time keys differ: an explicit value stands (the one-off backfill
        # relies on this); a write that changes nothing keeps the old time.
        modified = (
            _time_value(new_meta, MODIFIED_FIELD)
            or _time_value(current_meta, MODIFIED_FIELD)
            or format_note_time(fallback_modified or now)
        )
        created = _time_value(new_meta, CREATED_FIELD) or created
    else:
        modified = format_note_time(now)
    return {CREATED_FIELD: created, MODIFIED_FIELD: modified}


_OPENING_FENCE = re.compile(r"\A---[ \t]*(\r?\n)")
_CLOSING_FENCE = re.compile(r"^---[ \t]*\r?$", re.MULTILINE)


def _set_time_lines(markdown: str, wanted: dict[str, str]) -> str | None:
    """Set the time keys by touching only their own lines, or None if that is not possible.

    A YAML round trip can re-quote or reflow unrelated keys. Adding or replacing the two
    lines textually leaves every other byte of the note as it was.
    """
    opening = _OPENING_FENCE.match(markdown)
    if opening is None:
        return None
    newline = opening.group(1)
    closing = _CLOSING_FENCE.search(markdown, opening.end())
    if closing is None:
        return None
    block = markdown[opening.end() : closing.start()]
    for key, value in wanted.items():
        line = f"{key}: '{value}'"
        found = re.search(rf"^{key}:[^\n]*$", block, re.MULTILINE)
        if found is not None:
            following = block[found.end() :].lstrip("\r").lstrip("\n")
            if following[:1] in {" ", "\t"}:  # a multi-line value: leave it to the YAML path
                return None
            block = (
                block[: found.start()]
                + line
                + ("\r" if found.group().endswith("\r") else "")
                + block[found.end() :]
            )
        else:
            block += line + newline
    return markdown[: opening.end()] + block + markdown[closing.start() :]


def stamp_markdown_times(markdown: str, wanted: dict[str, str]) -> str:
    """Return `markdown` carrying exactly the wanted times (unchanged if it already does)."""
    meta = _frontmatter(markdown)
    if meta is None or all(_time_value(meta, key) == value for key, value in wanted.items()):
        return markdown
    textual = _set_time_lines(markdown, wanted)
    if textual is not None:
        check = _frontmatter(textual)
        if check is not None and all(
            _time_value(check, key) == value for key, value in wanted.items()
        ):
            return textual
    return rewrite_frontmatter_fields(markdown, updates=wanted)


def stamp_note_times(
    prepared: PreparedEntityWrite,
    *,
    now: datetime,
    current_markdown: str | None,
    fallback_created: datetime | None = None,
    fallback_modified: datetime | None = None,
) -> PreparedEntityWrite:
    """Return `prepared` with `created`/`modified` set per the rules in the module doc."""
    if prepared.entity_markdown.frontmatter_state == "malformed":
        return prepared
    wanted = wanted_note_times(
        prepared.markdown_content,
        now=now,
        current_markdown=current_markdown,
        fallback_created=fallback_created,
        fallback_modified=fallback_modified,
    )
    if wanted is None:
        return prepared
    markdown = stamp_markdown_times(prepared.markdown_content, wanted)
    if markdown == prepared.markdown_content:
        return _with_entity_times(prepared, wanted)
    frontmatter = prepared.entity_markdown.frontmatter
    entity_markdown = prepared.entity_markdown.model_copy(
        update={
            "frontmatter": frontmatter.model_copy(
                update={"metadata": {**frontmatter.metadata, **wanted}}
            )
        }
    )
    entity_metadata = {**(prepared.entity_fields.entity_metadata or {}), **wanted}
    return _with_entity_times(
        replace(
            prepared,
            markdown_content=markdown,
            entity_fields=replace(prepared.entity_fields, entity_metadata=entity_metadata),
            entity_markdown=entity_markdown,
        ),
        wanted,
    )


def _with_entity_times(
    prepared: PreparedEntityWrite, wanted: dict[str, str]
) -> PreparedEntityWrite:
    """Make the entity row's times agree with the frontmatter the note now carries."""
    created = datetime.fromisoformat(wanted[CREATED_FIELD])
    modified = datetime.fromisoformat(wanted[MODIFIED_FIELD])
    return replace(
        prepared,
        entity_fields=replace(prepared.entity_fields, created_at=created, updated_at=modified),
    )


def keep_move_times(
    prepared: PreparedEntityMove,
    *,
    current_markdown: str,
    entity_created_at: datetime,
    entity_updated_at: datetime,
) -> PreparedEntityMove:
    """A plain move changes neither time; a note without the keys gets the row's times."""
    current_meta = _frontmatter(current_markdown)
    new_meta = _frontmatter(prepared.markdown_content)
    if current_meta is None or new_meta is None:
        return prepared
    wanted = {
        CREATED_FIELD: _time_value(current_meta, CREATED_FIELD)
        or format_note_time(entity_created_at),
        MODIFIED_FIELD: _time_value(current_meta, MODIFIED_FIELD)
        or format_note_time(entity_updated_at),
    }
    if all(_time_value(new_meta, key) == value for key, value in wanted.items()):
        return prepared
    return replace(
        prepared,
        markdown_content=rewrite_frontmatter_fields(prepared.markdown_content, updates=wanted),
    )
