"""Structural section scanning for markdown note bodies.

A section is one heading-bounded span of a note body: the heading line through the
line before the next heading of the same or higher level (subsections stay inside
their parent, so spans nest). Sections are coordinates into canonical content —
body-relative line numbers and utf-8 byte offsets — never a copy of it (SPEC-47 /
issue #1403).

The scan is structural, not semantic: it runs on every markdown parse regardless of
the ``bm_parse_semantics`` opt-out, so graph-silent notes (e.g. PDF extraction text)
are still section-addressable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Literal

from markdown_it import MarkdownIt

# Path segments join on "/" for storage and selector matching. A heading whose own
# text contains "/" ("Ein/Aus") is addressed by that whole text: a selector equal to
# a heading's text wins over its reading as a path (see _selector_lookup).
HEADING_PATH_SEPARATOR = "/"

# A location names a heading by its full path from the top, joined with " > "
# ("A > KI/ML", "Top > Dec"); inside it "/" is plain heading text. A location of
# one heading starts with "> " ("> Top"), which tells it from the heading name
# "Top" that any heading of that text ends. Only a ">" with a space on each side
# separates, so headings such as "a>b" or "x -> y" stay single heading names.
_LOCATION_SEPARATOR = " > "
_LOCATION_ROOT = "> "

# Plain parser: heading boundaries need no observation/relation plugins, and the
# token stream already skips fenced code blocks and handles setext headings.
_md = MarkdownIt()

# markdown-it normalizes "\r\n?|\n" to "\n" before assigning token line maps, so
# every coordinate in this module must count lines by that same terminator rule.
# str.split("\n") would fold a lone "\r" into its neighbouring line and shift
# every later heading out of sync with the token stream.
_LINE_TERMINATOR = re.compile(r"\r\n|\r|\n")


def _split_lines(text: str) -> list[str]:
    r"""Split text on markdown-it's line-terminator rule (``\r\n``, lone ``\r``, or ``\n``).

    Mirrors ``str.split("\n")`` shape: a trailing terminator leaves one empty
    artifact element. Returned lines carry no terminator characters, so joining
    with ``"\n"`` and re-splitting round-trips the same list.
    """
    return _LINE_TERMINATOR.split(text)


def _ends_with_terminator(text: str) -> bool:
    """True when the text ends on a line terminator (so split() grew an artifact)."""
    return text.endswith(("\n", "\r"))


def document_lines(text: str) -> list[str]:
    """Count physical Markdown lines, without a phantom line after a final newline.

    Unlike str.splitlines(), Unicode separators inside prose do not move the
    coordinates away from those used by the section and line-range reader.
    """
    if not text:
        return []
    lines = _split_lines(text)
    return lines[:-1] if _ends_with_terminator(text) else lines


def setext_heading_underlined_at(text: str, underline_line: int) -> bool:
    """Return whether a setext heading in ``text`` has its underline on ``underline_line``.

    ``underline_line`` is 0-indexed and counted by markdown-it's line rule. A setext
    heading token's map ends on the line after its underline.
    """
    return any(
        token.type == "heading_open"
        and token.markup in ("=", "-")
        and token.map is not None
        and token.map[1] == underline_line + 1
        for token in _md.parse(text)
    )


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """One heading-bounded span of a note body, addressed by its heading path.

    Lines are 1-indexed and body-relative (frontmatter excluded); offsets are
    utf-8 byte positions into the encoded body, with ``end_offset`` exclusive so
    ``body.encode()[start_offset:end_offset]`` round-trips the section bytes.
    """

    heading: str
    level: int
    path: tuple[str, ...]
    duplicate_index: int
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    # The heading's last line: start_line for an ATX heading, the underline for setext.
    heading_end_line: int

    @property
    def heading_path(self) -> str:
        """The joined ancestor path stored in the section index."""
        return HEADING_PATH_SEPARATOR.join(self.path)


@dataclass(frozen=True, slots=True)
class _ScannedHeading:
    """One heading token flattened out of the markdown-it stream."""

    level: int
    text: str
    line_index: int
    # Exclusive 0-indexed end: a setext heading spans its text and its underline.
    end_line_index: int


def _scan_headings(body: str) -> list[_ScannedHeading]:
    """Collect heading levels, inline text, and 0-indexed start lines in order."""
    tokens = _md.parse(body)
    headings: list[_ScannedHeading] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        # heading_open tags are always h1-h6; the following inline token carries the
        # authored heading text (trailing ATX hashes already stripped).
        headings.append(
            _ScannedHeading(
                level=int(token.tag[1:]),
                text=tokens[index + 1].content.strip(),
                line_index=token.map[0],
                end_line_index=token.map[1],
            )
        )
    return headings


def scan_sections(body: str) -> tuple[MarkdownSection, ...]:
    """Scan a frontmatter-stripped note body into its heading-bounded sections.

    The scan is pure and deterministic over the exact string given: read paths
    re-run it on the content bytes being served instead of trusting stored rows,
    so a section slice can never disagree with the text it slices.
    """
    if not body:
        return ()

    headings = _scan_headings(body)
    if not headings:
        return ()

    lines = _split_lines(body)
    # A trailing terminator yields one empty artifact element in the split; it is
    # not a document line, but its start offset is exactly the body's byte length,
    # which makes it the natural exclusive end for the final section.
    line_count = len(lines) - 1 if _ends_with_terminator(body) else len(lines)

    # Offsets index the ORIGINAL bytes, so each line advances by its own
    # terminator's width ("\r\n" is two bytes; terminators are ASCII, so their
    # str length is their utf-8 byte length). findall yields one terminator per
    # split boundary, i.e. one for every line but the last.
    terminators = _LINE_TERMINATOR.findall(body)
    line_start_offsets: list[int] = []
    offset = 0
    for index, line in enumerate(lines):
        line_start_offsets.append(offset)
        offset += len(line.encode("utf-8"))
        if index < len(terminators):
            offset += len(terminators[index])
    total_bytes = offset

    def line_offset(line_index: int) -> int:
        return line_start_offsets[line_index] if line_index < len(lines) else total_bytes

    sections: list[MarkdownSection] = []
    # Stack of (level, heading) ancestors: a new heading pops everything at its own
    # level or deeper, so the stack always spells the open path down to it.
    path_stack: list[tuple[int, str]] = []
    duplicate_counts: dict[tuple[str, ...], int] = {}
    for position, heading in enumerate(headings):
        while path_stack and path_stack[-1][0] >= heading.level:
            path_stack.pop()
        path_stack.append((heading.level, heading.text))
        path = tuple(text for _, text in path_stack)
        duplicate_index = duplicate_counts.get(path, 0)
        duplicate_counts[path] = duplicate_index + 1

        # A section ends at the next heading of the same or higher level; the
        # heading-level cap of 6 bounds this look-ahead to linear total work.
        # Index the original list: copying every suffix is quadratic even when
        # the next heading immediately closes the section.
        end_line_index = line_count
        for later_position in range(position + 1, len(headings)):
            later = headings[later_position]
            if later.level <= heading.level:
                end_line_index = later.line_index
                break

        sections.append(
            MarkdownSection(
                heading=heading.text,
                level=heading.level,
                path=path,
                duplicate_index=duplicate_index,
                start_line=heading.line_index + 1,
                end_line=end_line_index,
                start_offset=line_start_offsets[heading.line_index],
                end_offset=line_offset(end_line_index),
                heading_end_line=heading.end_line_index,
            )
        )
    return tuple(sections)


# --- Read-time slicing (section=, lines=, max_tokens=) ---
#
# The note-read API re-runs scan_sections on the exact content it serves instead
# of consulting stored note_section rows: derived rows are eventually consistent,
# while these helpers are pure over the string in hand, so a slice can never
# disagree with the text it slices.

# Heuristic token budget: the repo carries no tokenizer dependency, so max_tokens
# is charged in characters at ~4 chars/token (the common English-plus-markdown
# approximation).
_CHARS_PER_TOKEN = 4

# The explicit truncation marker: appended as a final line, excluded from the
# budget, and naming the exact document-absolute line to resume reading from.
TRUNCATION_MARKER_TEMPLATE = (
    "… [truncated at max_tokens={max_tokens}; continue with lines={continue_line}-]"
)

_DUPLICATE_INDEX_SUFFIX = re.compile(r"\[(\d+)\]\s*$")
# ATX headings require the space after the hashes, so only "# Heading" forms are
# stripped from selector segments; "#tag"-like text stays literal.
_ATX_HEADING_PREFIX = re.compile(r"^#{1,6} ")


@dataclass(frozen=True, slots=True)
class SectionSelector:
    """A parsed ``section=`` selector: its three readings plus a duplicate index.

    ``heading`` reads the whole selector as one heading's text, "/" and " > "
    included; ``segments`` reads it as a path split on "/", and is None when that
    split leaves an empty segment ("/etc", "Ende/", "A//B"); ``location`` reads it
    as a full path from the top split on " > " ("A > KI/ML", "> Top"), and is None
    when it is not written as one or leaves an empty segment. Which reading applies
    depends on the note's headings (``_selector_lookup``). ``raw`` keeps the
    caller's original spelling for error messages, ``stem`` the same text trimmed
    and without its ``[n]``. Bare selectors address duplicate index 0 (the first
    section with that path).
    """

    raw: str
    stem: str
    heading: str
    segments: tuple[str, ...] | None
    location: tuple[str, ...] | None
    duplicate_index: int
    # True when the caller wrote the [n] suffix; edits need it to pick among duplicates.
    indexed: bool = False

    @property
    def malformed(self) -> bool:
        """True when only a heading bearing the whole text could give it a meaning."""
        return self.segments is None and self.location is None


def _selector_segment(text: str) -> str:
    """Clean one heading name: outer whitespace and a "## "-style prefix go."""
    return _ATX_HEADING_PREFIX.sub("", text.strip()).strip()


def _is_written_as_location(heading: str) -> bool:
    return _LOCATION_SEPARATOR in heading or heading.startswith(_LOCATION_ROOT)


def _location_segments(heading: str) -> tuple[str, ...] | None:
    """Split a selector written as a location; None when it is not one or has a gap.

    ``heading`` is the whole selector with one "## " prefix already dropped. Each
    heading of the location loses exactly one such prefix, so "# # x > y" is
    heading "y" under "# x", as "> # # x > y" is.
    """
    if heading.startswith(_LOCATION_ROOT):
        first, *rest = heading.removeprefix(_LOCATION_ROOT).split(_LOCATION_SEPARATOR)
        first = _selector_segment(first)
    elif _LOCATION_SEPARATOR in heading:
        # Trigger: no "> " in front, so the dropped prefix belonged to the first heading.
        # Why: dropping one more read "# # x > y" as heading "y" under "x" (review B3 C).
        # Outcome: the first heading is only trimmed; the others lose their own prefix.
        first, *rest = heading.split(_LOCATION_SEPARATOR)
        first = first.strip()
    else:
        return None
    segments = (first, *(_selector_segment(part) for part in rest))
    return segments if all(segments) else None


def parse_section_selector(selector: str) -> SectionSelector | None:
    """Parse a section selector, or return None when it names no heading at all.

    Accepted forms: ``"Decisions"``, ``"Auth/Decisions"`` (path form),
    ``"Heading[1]"`` (duplicate index on the last segment), heading-styled
    segments like ``"## Auth"`` (leading hashes plus one space are stripped), a
    heading whose own text contains "/" (``"Ein/Aus"``), named whole, and a
    location from the top (``"A > KI/ML"``, ``"> Top"``). A trailing ``[n]`` is
    always the duplicate index, so a heading whose text itself ends in ``[1]`` is
    named ``"Heading[1][0]"``.

    Only an empty selector (``""``, ``"[2]"``) is refused here. One whose path
    reading has an empty segment (``"/etc"``) may still be a heading's whole text,
    which only the note's headings can tell.
    """
    text = selector.strip()
    duplicate_index = 0
    suffix = _DUPLICATE_INDEX_SUFFIX.search(text)
    indexed = suffix is not None
    if suffix:
        duplicate_index = int(suffix.group(1))
        text = text[: suffix.start()].rstrip()

    heading = _selector_segment(text)
    if not heading:
        return None
    segments = tuple(_selector_segment(part) for part in text.split(HEADING_PATH_SEPARATOR))
    return SectionSelector(
        raw=selector,
        stem=text,
        heading=heading,
        segments=segments if all(segments) else None,
        location=_location_segments(heading),
        duplicate_index=duplicate_index,
        indexed=indexed,
    )


@dataclass(frozen=True, slots=True)
class LineRange:
    """A 1-indexed inclusive document-absolute line range; ``end=None`` reads to EOF."""

    start: int
    end: int | None


def line_range_past_end(start_line: int, total_lines: int) -> str | None:
    """Return the error for a range that starts after the document's last line.

    Line 1 of an empty document stays readable: reading from the top is always valid,
    and it returns empty content rather than an error.
    """
    if start_line <= max(total_lines, 1):
        return None
    return f"start_line {start_line} is past the end of the document ({total_lines} lines)"


def parse_line_range(value: str) -> LineRange | None:
    """Parse ``"N-M"``, ``"N-"`` (to end), or ``"N"``; return None when malformed."""
    first, dash, last = (part.strip() for part in value.strip().partition("-"))
    if not first.isdigit():
        return None
    start = int(first)
    if start < 1:
        return None
    if not dash:
        return LineRange(start=start, end=start)
    if not last:
        return LineRange(start=start, end=None)
    if not last.isdigit():
        return None
    end = int(last)
    if end < start:
        return None
    return LineRange(start=start, end=end)


@dataclass(frozen=True, slots=True)
class NoteSlice:
    """One served slice of a note, with document-absolute coordinates.

    ``content`` never carries a frontmatter block unless an explicit line range
    covered it; line numbers always count over the full stored document
    (frontmatter included), matching line-range follow-up reads. An empty
    document has no line 1, so any slice of it is the empty range 0-0 of 0.
    """

    content: str
    section: str | None
    start_line: int
    end_line: int
    total_lines: int
    truncated: bool
    continue_line: int | None


@dataclass(frozen=True, slots=True)
class NoteSliceError:
    """A slice failure: an unknown, ambiguous, or out-of-range section, or a past-end range."""

    message: str
    # True when the selector is malformed as a path ("A//B"), is no location, and no
    # heading bears its whole text either: callers report it as the malformed
    # selector it was before such a selector could name a heading.
    invalid_selector: bool = False


def _frontmatter_line_count(full_text: str) -> int:
    """Count the leading lines occupied by a well-formed opening frontmatter block.

    Structural, delimiter-only rule (top-of-file ``---`` fence pairs, matching
    ``mcp.note_reads.parse_opening_frontmatter``); YAML validity is deliberately
    not checked here — the markdown layer stays free of YAML parsing. Returns 0
    when no terminated block opens the document.
    """
    lines = _split_lines(full_text)
    if not lines or lines[0].strip() != "---":
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return index + 1
    return 0


# --- Selector lookup ---
#
# Looking a selector up and wording the outcome are kept apart: the names that
# messages offer and that reads return are checked by looking them up again, and
# that check must not word messages (which would offer names, and so on).

type _Reading = Literal["heading", "path", "location"]


@dataclass(frozen=True, slots=True)
class _Headings:
    """A note's sections, indexed for repeated lookups."""

    sections: tuple[MarkdownSection, ...]
    # Own heading text -> its sections in document order. A path's last segment is a
    # heading's own text, so a lookup scans only the sections ending in it; checking
    # every offered name stays cheap on notes with thousands of headings.
    by_text: dict[str, tuple[MarkdownSection, ...]]
    # Full path -> its sections in document order, so position equals duplicate_index.
    by_path: dict[tuple[str, ...], tuple[MarkdownSection, ...]]
    # Frontmatter lines before the body, to word document-absolute line numbers.
    line_offset: int


def _index_headings(sections: tuple[MarkdownSection, ...], line_offset: int) -> _Headings:
    by_text: dict[str, list[MarkdownSection]] = {}
    by_path: dict[tuple[str, ...], list[MarkdownSection]] = {}
    for section in sections:
        by_text.setdefault(section.heading, []).append(section)
        by_path.setdefault(section.path, []).append(section)
    return _Headings(
        sections=sections,
        by_text={text: tuple(group) for text, group in by_text.items()},
        by_path={path: tuple(group) for path, group in by_path.items()},
        line_offset=line_offset,
    )


@dataclass(frozen=True, slots=True)
class _Lookup:
    """The sections one reading of a selector reaches, before any message is worded."""

    reading: _Reading
    matches: tuple[MarkdownSection, ...]
    # Distinct full paths among the matches, told apart as tuples: ("A/B", "X") and
    # ("A", "B", "X") join to the same string but are different headings.
    paths: tuple[tuple[str, ...], ...]
    # The addressed section: set when all matches share one path and the index exists.
    section: MarkdownSection | None


def _lookup(
    headings: _Headings,
    segments: tuple[str, ...],
    duplicate_index: int,
    *,
    reading: _Reading,
) -> _Lookup:
    """Find the sections whose full path ends with these segments (suffix match)."""
    matches = tuple(
        section
        for section in headings.by_text.get(segments[-1], ())
        if section.path[-len(segments) :] == segments
    )
    paths = tuple(dict.fromkeys(section.path for section in matches))
    # All matches share one full path; document order equals duplicate_index order.
    addressed = (
        next((s for s in matches if s.duplicate_index == duplicate_index), None)
        if len(paths) == 1
        else None
    )
    return _Lookup(reading=reading, matches=matches, paths=paths, section=addressed)


def _location_lookup(
    headings: _Headings, location: tuple[str, ...], duplicate_index: int
) -> _Lookup:
    """Find the sections whose full path is exactly this location (read from the top)."""
    matches = headings.by_path.get(location, ())
    return _Lookup(
        reading="location",
        matches=matches,
        paths=(location,) if matches else (),
        section=matches[duplicate_index] if duplicate_index < len(matches) else None,
    )


def _selector_lookup(headings: _Headings, selector: SectionSelector) -> _Lookup | None:
    """Look a selector up as one heading, a location or a path; None when malformed."""
    index = selector.duplicate_index
    # Trigger: the selector contains "/" or is written as a location, and some
    #   heading's own text is exactly it ("Ein/Aus", "/etc", "A > B"); a "## Ein/Aus"
    #   line inside a code fence is no heading.
    # Why: "/" otherwise always splits the path, so such a heading had no selector at
    #   all; a heading named exactly as written takes precedence (decisions B3 A, C).
    # Outcome: the selector names that heading, and that reading's errors (ambiguous,
    #   index out of range, unindexed duplicates) are final, with no retry as a path
    #   or location. A selector without "/" and " > " reads the same either way.
    heading = selector.heading
    if (
        HEADING_PATH_SEPARATOR in heading or _is_written_as_location(heading)
    ) and heading in headings.by_text:
        # Without "/" ("A > B") this is the one-segment path reading it always was,
        # so it keeps that reading's messages too.
        reading: _Reading = "heading" if HEADING_PATH_SEPARATOR in heading else "path"
        return _lookup(headings, (heading,), index, reading=reading)
    path = (
        None
        if selector.segments is None
        else _lookup(headings, selector.segments, index, reading="path")
    )
    if selector.location is None:
        return path
    # Trigger: the selector is written as a location ("A > KI/ML", "> Top").
    # Why: before decision B3 C " > " was plain heading text, so a "/" path through
    #   a heading named like "A > B" ("A > B/C": heading "C" under "A > B") reached a
    #   section; such an old selector keeps that meaning.
    # Outcome: the old path reading when it reaches any section, its errors
    #   included; otherwise the location, read from the top heading down.
    if path is not None and path.matches:
        return path
    return _location_lookup(headings, selector.location, index)


# --- Names that read back ---
#
# A read returns the section's name, and the messages offer names to pick from. A
# caller hands such a name on to an edit, so each one is looked up again and kept
# only if it reaches the same section. The joined heading path is not enough: a
# heading whose text contains "/" splits differently when read back ("Plan/KI/ML"
# is heading "ML" under "KI"), a heading may be named like a path ("Top/Dec"), and
# a heading ending in "[1]" loses that text to the duplicate index. The location
# ("A > KI/ML", "> Top") comes last: it reads every path exactly, unless a heading
# is named like it or its text holds " > " itself.


def _reaches(headings: _Headings, name: str, target: MarkdownSection) -> bool:
    """True when the selector ``name`` addresses exactly ``target``."""
    selector = parse_section_selector(name)
    found = None if selector is None else _selector_lookup(headings, selector)
    return found is not None and found.section == target


def _spelled(path: tuple[str, ...]) -> tuple[str, ...]:
    """Heading texts as a selector spells them, so that reading them back keeps them."""
    # A heading text that itself starts like "# x" would lose that prefix when read
    # back (selector segments drop a "## " prefix), so it gets one more "# " to drop.
    return tuple(f"# {text}" if _ATX_HEADING_PREFIX.match(text) else text for text in path)


def _location_spellings(path: tuple[str, ...]) -> list[str]:
    """Spell a full path as a location: "A > KI/ML" (then "> A > KI/ML"), or "> Top"."""
    joined = _LOCATION_SEPARATOR.join(_spelled(path))
    if len(path) == 1:
        return [_LOCATION_ROOT + joined]
    # The "> " form is the way out when a heading is named exactly "A > KI/ML".
    return [joined, _LOCATION_ROOT + joined]


def _selector_name(headings: _Headings, target: MarkdownSection, *, indexed: bool) -> str | None:
    """The first spelling that reads back to ``target``, or None when none does.

    Tried in order: the full heading path, then shorter path suffixes, then the same
    with spaces around "/" (which a heading named "KI/ML" does not take over), then
    the location from the top. ``indexed`` names the duplicate index ("Log[1]"), as
    twins need for an edit; otherwise the bare name comes first and "[0]" is the
    fallback for a heading whose own text ends in "[n]".
    """
    path = _spelled(target.path)
    suffixes = [path[-length:] for length in range(len(path), 0, -1)]
    stems = (
        [HEADING_PATH_SEPARATOR.join(suffix) for suffix in suffixes]
        + [f" {HEADING_PATH_SEPARATOR} ".join(suffix) for suffix in suffixes if len(suffix) > 1]
        + _location_spellings(target.path)
    )
    index = f"[{target.duplicate_index}]"
    for stem in stems:
        for name in (stem + index,) if indexed else (stem, stem + index):
            if _reaches(headings, name, target):
                return name
    return None


def _section_label(
    headings: _Headings,
    target: MarkdownSection,
    selector: SectionSelector,
    *,
    indexed: bool,
) -> str:
    """Name a section the caller's selector reached, so the name reaches it again."""
    name = _selector_name(headings, target, indexed=indexed)
    # Trigger: no canonical spelling reads back (all are taken by other headings or
    #   ambiguous), yet the caller's own spelling found this section.
    # Why: the caller's stem with this section's index reads the same way again.
    # Outcome: the caller's spelling with an explicit index.
    return name if name is not None else f"{selector.stem}[{target.duplicate_index}]"


def _unreachable(headings: _Headings, section: MarkdownSection) -> str:
    """Point to a section no selector names: its heading text and document lines."""
    # A location string would itself be a selector now, and the one left over for
    # such a section reaches another one, so the message gives lines instead.
    start = headings.line_offset + section.start_line
    end = headings.line_offset + section.end_line
    return f"'{section.heading}' (lines {start}-{end})"


def _offer(
    headings: _Headings, targets: list[tuple[MarkdownSection, bool]]
) -> tuple[list[str], list[str]]:
    """Name each (section, indexed) target; those no selector names go by lines."""
    names: list[str] = []
    unreachable: list[str] = []
    for target, indexed in targets:
        name = _selector_name(headings, target, indexed=indexed)
        if name is None:
            unreachable.append(_unreachable(headings, target))
        else:
            names.append(name)
    return names, unreachable


def _offer_text(headings: _Headings, targets: list[tuple[MarkdownSection, bool]]) -> str:
    """Word an offer: the names, then the headings no selector reaches."""
    names, unreachable = _offer(headings, targets)
    text = ", ".join(names) or "(none)"
    if unreachable:
        text += f". Not reachable by a section selector (read by lines): {', '.join(unreachable)}"
    return text


def _with_twin_flags(
    sections: tuple[MarkdownSection, ...],
) -> list[tuple[MarkdownSection, bool]]:
    """Pair each section with whether it repeats its path, so it needs its [n]."""
    counts = Counter(section.path for section in sections)
    return [(section, counts[section.path] > 1) for section in sections]


def _first_of_paths(found: _Lookup) -> list[tuple[MarkdownSection, bool]]:
    """One target per distinct path, named without an index ("Qualify it")."""
    return [
        (next(section for section in found.matches if section.path == path), False)
        for path in found.paths
    ]


def _path_hint(headings: _Headings, selector: SectionSelector) -> str:
    """Name the path a refused one-heading selector also spells, when it reaches one."""
    # Trigger: the selector named a heading whose text contains "/", and that reading
    #   failed (two parents, or no such duplicate index).
    # Why: the selector also spells a path ("KI/ML": heading "ML" under "KI"); the
    #   caller may have meant it, and the refusal is the only place to learn how to
    #   write it now that the heading reading wins.
    # Outcome: names that read back to the path's sections, or nothing.
    if selector.segments is None:
        return ""
    found = _lookup(headings, selector.segments, selector.duplicate_index, reading="path")
    names, _ = _offer(headings, _with_twin_flags(found.matches))
    if not names:
        return ""
    *parents, last = selector.segments
    return (
        f" For the path (heading '{last}' under '{_LOCATION_SEPARATOR.join(parents)}') "
        f"write {', '.join(names)}."
    )


@dataclass(frozen=True, slots=True)
class _SectionMatch:
    """The section a selector found and the name that reads back to it."""

    section: MarkdownSection
    label: str


def _find_section(
    headings: _Headings,
    selector: SectionSelector,
    note_title: str | None,
) -> _SectionMatch | NoteSliceError:
    """Match a selector against scanned sections, returning the section and its name.

    A selector matches when a section's full path ends with the selector segments
    (suffix match), so ``"Decisions"`` finds ``("Auth", "Decisions")`` and
    ``"Auth/Decisions"`` disambiguates. A selector equal to a heading's own text,
    "/" or " > " included, matches that heading instead, and a location
    (``"A > KI/ML"``) matches the full path from the top (``_selector_lookup``).
    Comparison is exact and case-sensitive. Every name returned or offered reads
    back to its section (``_selector_name``).
    """
    note_label = f" in note '{note_title}'" if note_title else ""
    found = _selector_lookup(headings, selector)
    if found is None:
        return NoteSliceError(f"Invalid section selector: '{selector.raw}'", invalid_selector=True)
    # The one-heading reading applies only when a heading bears its text, so only
    # the path and the location reading can miss.
    if not found.matches:
        available = (
            _offer_text(headings, _with_twin_flags(headings.sections))
            if headings.sections
            else "(note has no headings)"
        )
        location_hint = (
            " A location starts at a heading with no heading above it."
            if found.reading == "location"
            else ""
        )
        return NoteSliceError(
            f"Section '{selector.raw}' not found{note_label}.{location_hint} "
            f"Available sections: {available}."
        )

    # A location is one full path, so only the other two readings can be ambiguous.
    if len(found.paths) > 1:
        if found.reading == "heading":
            return NoteSliceError(
                f"Section '{selector.raw}' is ambiguous{note_label}: {len(found.paths)} paths "
                f"end in a heading named '{selector.heading}'. "
                f"Qualify it: {_offer_text(headings, _first_of_paths(found))}."
                + _path_hint(headings, selector)
            )
        return NoteSliceError(
            f"Section '{selector.raw}' is ambiguous{note_label}. "
            f"Qualify it: {_offer_text(headings, _first_of_paths(found))}."
        )

    if found.section is None:
        count = f"{len(found.matches)} "
        indexes = f"(duplicate indexes 0-{len(found.matches) - 1})"
        if found.reading == "heading":
            return NoteSliceError(
                f"Section '{selector.raw}' not found{note_label}: {count}heading(s) named "
                f"'{selector.heading}' {indexes}." + _path_hint(headings, selector)
            )
        if found.reading == "location":
            location = _location_spellings(found.paths[0])[0]
            return NoteSliceError(
                f"Section '{selector.raw}' not found{note_label}: {count}section(s) have "
                f"the location '{location}' {indexes}."
            )
        return NoteSliceError(
            f"Section '{selector.raw}' not found{note_label}: {count}section(s) share the "
            f"path '{HEADING_PATH_SEPARATOR.join(found.paths[0])}' {indexes}."
        )
    label = _section_label(headings, found.section, selector, indexed=len(found.matches) > 1)
    return _SectionMatch(section=found.section, label=label)


@dataclass(frozen=True, slots=True)
class SectionSpan:
    """Where a selected section sits in a full document, for edits.

    Positions are 0-indexed into ``full_text.split("\\n")`` and count the whole
    document, frontmatter included. ``end`` and ``subsections_start`` are exclusive.
    """

    heading: str
    level: int
    label: str
    heading_start: int
    # First line after the heading; a setext heading spans two lines.
    body_start: int
    # The next heading of the same or higher level, or the end of the document.
    end: int
    # The first subsection heading, or `end` when the section has none.
    subsections_start: int


def locate_section(
    full_text: str,
    selector: SectionSelector,
    *,
    note_title: str | None = None,
) -> SectionSpan | NoteSliceError:
    """Find the section an edit addresses, with the same selector reads use.

    Reads take the first of several same-path sections; an edit refuses instead,
    unless the selector names one with [n], because changing the wrong section
    loses text.
    """
    doc_lines = _split_lines(full_text)
    frontmatter_lines = _frontmatter_line_count(full_text)
    sections = scan_sections("\n".join(doc_lines[frontmatter_lines:]))
    headings = _index_headings(sections, frontmatter_lines)
    found = _find_section(headings, selector, note_title)
    if isinstance(found, NoteSliceError):
        return found
    matched, label = found.section, found.label
    twins = headings.by_path[matched.path]
    if len(twins) > 1 and not selector.indexed:
        note_label = f" in note '{note_title}'" if note_title else ""
        # Each choice reads back to its twin: for a heading named whole that is its
        # own text ("Ein/Aus[1]"), since "Top/Ein/Aus[1]" reads as heading "Aus".
        choices = ", ".join(
            _section_label(headings, twin, selector, indexed=True) for twin in twins
        )
        return NoteSliceError(
            f"Multiple sections found with header '{selector.raw}'{note_label}. "
            f"Name one: {choices}."
        )
    position = sections.index(matched)
    following = sections[position + 1] if position + 1 < len(sections) else None
    subsections_start = (
        following.start_line - 1
        if following is not None and following.start_line <= matched.end_line
        else matched.end_line
    )
    return SectionSpan(
        heading=matched.heading,
        level=matched.level,
        label=label,
        heading_start=frontmatter_lines + matched.start_line - 1,
        body_start=frontmatter_lines + matched.heading_end_line,
        end=frontmatter_lines + matched.end_line,
        subsections_start=frontmatter_lines + subsections_start,
    )


def _token_budget_line_count(text: str, text_lines: list[str], budget_chars: int) -> int | None:
    """Return how many leading lines of ``text`` to keep, or None when no cut applies.

    Cut preference: section boundary first, then paragraph (blank-line) boundary,
    then the last full line within budget. Never cuts mid-line, and always keeps
    the first nonblank line so a small budget cannot return only whitespace. An
    over-budget first content line is served whole — there is no boundary to cut at.
    """
    if len(text) <= budget_chars:
        return None
    line_count = len(text_lines)
    if line_count <= 1:
        return None
    minimum = next((index + 1 for index, line in enumerate(text_lines) if line.strip()), line_count)
    if minimum == line_count:
        return None

    # prefix_lengths[k] == len("\n".join(text_lines[:k]))
    prefix_lengths = [0]
    for index, line in enumerate(text_lines):
        separator = 1 if index > 0 else 0
        prefix_lengths.append(prefix_lengths[-1] + separator + len(line))

    def fits(kept: int) -> bool:
        return prefix_lengths[kept] <= budget_chars

    # Section boundaries strictly after the slice start (its own heading, when the
    # slice begins with one, is never a cut point).
    section_cuts = [
        section.start_line - 1 for section in scan_sections(text) if section.start_line > 1
    ]
    fitting = [kept for kept in section_cuts if kept >= minimum and fits(kept)]
    if fitting:
        return max(fitting)

    paragraph_cuts = [
        index for index, line in enumerate(text_lines) if index >= 1 and not line.strip()
    ]
    fitting = [kept for kept in paragraph_cuts if kept >= minimum and fits(kept)]
    if fitting:
        return max(fitting)

    # Line backstop: the largest whole-line prefix within budget; when even the
    # first content line exceeds the budget, keep it and its leading whitespace
    # anyway, preserving coordinates and serving actual content.
    for kept in range(line_count - 1, minimum - 1, -1):
        if fits(kept):
            return kept
    return minimum


def slice_note_content(
    full_text: str,
    *,
    section: SectionSelector | None = None,
    lines: LineRange | None = None,
    max_tokens: int | None = None,
    note_title: str | None = None,
) -> NoteSlice | NoteSliceError:
    """Slice a full stored note document by section, line range, and/or token budget.

    ``full_text`` is the exact content being served (frontmatter included). Section
    and token-budget slices exclude the frontmatter block; an explicit line range
    addresses the full document so continue-reads and cat coordinates agree. All
    returned coordinates are document-absolute 1-indexed lines; served content
    joins lines with "\\n", so CR/CRLF terminators come back normalized.
    """
    if section is not None and lines is not None:
        raise ValueError("section and lines are mutually exclusive")
    if section is None and lines is None and max_tokens is None:
        raise ValueError("slice_note_content requires section, lines, or max_tokens")

    doc_lines = _split_lines(full_text)
    total_lines = 0
    if full_text:
        total_lines = len(doc_lines) - 1 if _ends_with_terminator(full_text) else len(doc_lines)

    section_label: str | None = None
    if section is not None:
        frontmatter_lines = _frontmatter_line_count(full_text)
        # _split_lines strips terminators, so this join is the terminator-
        # normalized document suffix: re-splitting it yields these exact lines,
        # and scanned coordinates translate by the frontmatter offset.
        body = "\n".join(doc_lines[frontmatter_lines:])
        headings = _index_headings(scan_sections(body), frontmatter_lines)
        found = _find_section(headings, section, note_title)
        if isinstance(found, NoteSliceError):
            return found
        matched, section_label = found.section, found.label
        start_line = frontmatter_lines + matched.start_line
        end_line = frontmatter_lines + matched.end_line
    elif lines is not None:
        # Trigger: the range starts after the last line.
        # Why: clamping only the end produced an inverted "Lines 500-17" range.
        # Outcome: the read fails with a message naming the document length.
        past_end = line_range_past_end(lines.start, total_lines)
        if past_end is not None:
            return NoteSliceError(past_end)
        start_line = lines.start
        end_line = min(lines.end, total_lines) if lines.end is not None else total_lines
    else:
        # max_tokens alone budgets the whole body: slices never carry frontmatter.
        frontmatter_lines = _frontmatter_line_count(full_text)
        start_line = frontmatter_lines + 1
        end_line = total_lines

    # Trigger: the document has no lines at all (an empty text file).
    # Why: a range from line 1 to the last line came back as 1-0, a range that
    #      ends before it starts, which no follow-up read can use.
    # Outcome: the empty range 0-0 of 0, with empty content.
    if total_lines == 0:
        start_line = end_line = 0

    selected = doc_lines[max(start_line - 1, 0) : end_line]
    content = "\n".join(selected)
    truncated = False
    continue_line: int | None = None
    if max_tokens is not None:
        kept = _token_budget_line_count(content, selected, max_tokens * _CHARS_PER_TOKEN)
        if kept is not None:
            continue_line = start_line + kept
            end_line = start_line + kept - 1
            marker = TRUNCATION_MARKER_TEMPLATE.format(
                max_tokens=max_tokens, continue_line=continue_line
            )
            content = "\n".join(selected[:kept]) + "\n" + marker
            truncated = True

    return NoteSlice(
        content=content,
        section=section_label,
        start_line=start_line,
        end_line=end_line,
        total_lines=total_lines,
        truncated=truncated,
        continue_line=continue_line,
    )
