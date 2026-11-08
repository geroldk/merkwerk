"""Turn the report of the scheduled Git sync into one note per message (BM-009, V5).

``git_sync_watch.py --notify-cmd bm-notify`` hands the report to this program on standard
input, one message per line. For every message it

1. writes the line to standard error (the journal), always, first;
2. puts it into the spool, a directory with one small file per message, so a message that
   cannot be delivered now is not lost, and the sync does not report it a second time;
3. writes a note into the notify project named in the host settings (``bm_write`` on the
   running local service) and deletes the spool file once the note exists.

If the service does not answer, the messages wait in the spool and are delivered by the next
run, in order. A message the service refuses (project missing, not writable) is tried again on
each run up to ``notify.max_attempts`` times, then moved to ``rejected/`` inside the spool and
reported on standard error; it never blocks the ones behind it. Delivery is idempotent: the note
name carries the message id, and "already exists" counts as delivered, so a run that was cut off
between writing the note and deleting the spool file does not create a second note.

Without a notify project in the settings (see ``bm_host_settings.py``) the messages go to the
journal only. Exit status: 0 whenever every message is delivered or safely spooled, 1 if the
settings are unusable or a message could not even be spooled (it is on standard error then).

    bm-notify [--settings FILE] [--url URL]     read the report from standard input
    bm-notify --status                          show the spool; exit 2 if something is stale
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm_host_settings import (  # noqa: E402
    HostSettings,
    NotifySettings,
    SettingsError,
    load_host_settings,
)

MAX_LINE = 2000
REJECTED = "rejected"
WRITE_TIMEOUT = 30.0
REPORT_LINE = re.compile(r"\[BM sync\]\s+(?P<project>[^:]+):\s+(?P<what>.*)")


class Unreachable(Exception):
    """The service did not answer (or is busy): keep the message and try again next run."""


class Rejected(Exception):
    """The service answered and refused this note: counts as a failed attempt."""


@dataclass(frozen=True)
class Message:
    id: str
    host: str
    at: str  # ISO 8601 with offset, to the second
    text: str

    @property
    def project(self) -> str | None:
        found = REPORT_LINE.fullmatch(self.text)
        return found["project"].strip() if found else None

    @property
    def problem(self) -> str | None:
        found = REPORT_LINE.fullmatch(self.text)
        return found["what"].split(" ", 1)[0] if found and found["what"] else None


@dataclass(frozen=True)
class NoteRequest:
    title: str
    directory: str
    content: str
    note_type: str
    tags: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DeliveryReport:
    delivered: int = 0
    rejected: int = 0
    waiting: int = 0
    unreachable: bool = False


Writer = Callable[[NoteRequest], Awaitable[None]]


# --- Messages ---


def split_report(text: str) -> list[str]:
    """One message per non-empty line; control characters out, overlong lines cut."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", raw).strip()
        if line:
            lines.append(line[:MAX_LINE])
    return lines


def make_messages(lines: list[str], host: str, now: datetime) -> list[Message]:
    at = now.astimezone().isoformat(timespec="seconds")
    return [
        Message(
            id=hashlib.sha256(f"{host}\0{at}\0{index}\0{line}".encode()).hexdigest()[:12],
            host=host,
            at=at,
            text=line,
        )
        for index, line in enumerate(lines)
    ]


def _defuse(text: str) -> str:
    """Break up every "[[" and "]]", also nested ones ("[[[x]]]"), so nothing becomes a link."""
    while "[[" in text or "]]" in text:
        text = text.replace("[[", "[ [").replace("]]", "] ]")
    return text


def note_for(message: Message, notify: NotifySettings) -> NoteRequest:
    """The note for one message. The title carries the id, so a replay finds its own note."""
    stamp = datetime.fromisoformat(message.at).strftime("%Y%m%d-%H%M%S")
    scope = re.sub(r"[^\w.-]+", "-", message.project or "meldung").strip("-") or "meldung"
    title = f"Abgleich {message.host} {stamp} {scope} {message.id[:6]}"
    # "Meldung:" first, so a report line can never open frontmatter ("---") or a link ("[[").
    body = [f"Meldung: {_defuse(message.text)}", "", f"- [host] {message.host}"]
    body.append(f"- [zeit] {message.at}")
    if message.project:
        body.append(f"- [projekt] {_defuse(message.project)}")
    if message.problem:
        body.append(f"- [problem] {_defuse(message.problem)}")
    body.append("- [status] offen")
    return NoteRequest(
        title=title,
        directory=notify.directory,
        content="\n".join(body) + "\n",
        note_type="meldung",
        tags=["abgleich", message.host],
        metadata={"host": message.host},
    )


# --- The spool ---


def _write_private(path: Path, data: bytes) -> None:
    """Create `path` atomically with owner-only rights."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)


def enqueue(spool: Path, messages: list[Message]) -> None:
    """Make every message durable before anything else is tried."""
    _ensure_directory(spool)
    for index, message in enumerate(messages):
        # UTC, so that file-name order is time order across the change of summer time.
        stamp = (
            datetime.fromisoformat(message.at).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        entry = {
            "id": message.id,
            "host": message.host,
            "at": message.at,
            "text": message.text,
            "attempts": 0,
        }
        _write_private(
            # The running number keeps the order of one report within the same second.
            spool / f"{stamp}-{index:03d}-{message.id}.json",
            json.dumps(entry, ensure_ascii=False, sort_keys=True).encode(),
        )


def pending(spool: Path, *, reject: bool = True) -> list[tuple[Path, dict[str, Any]]]:
    """Waiting messages, oldest first. A file that cannot be read goes to ``rejected/``
    (not with ``reject=False``, which only looks: ``--status`` changes nothing)."""
    if not spool.is_dir():
        return []
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(spool.glob("*.json")):
        try:
            entry = json.loads(path.read_text())
            if not all(isinstance(entry.get(key), str) for key in ("id", "host", "at", "text")):
                raise ValueError("incomplete spool entry")
            if datetime.fromisoformat(entry["at"]).tzinfo is None:
                raise ValueError("'at' has no UTC offset")
        except (OSError, ValueError, AttributeError) as error:
            if reject:
                _reject(spool, path, f"unreadable spool file: {error}")
            continue
        found.append((path, entry))
    return found


def _reject(spool: Path, path: Path, reason: str) -> None:
    _ensure_directory(spool / REJECTED)
    try:
        os.replace(path, spool / REJECTED / path.name)
    except FileNotFoundError:  # another run already delivered or set it aside
        return
    print(f"bm-notify: REJECTED {path.name}: {reason}", file=sys.stderr)


def stale(spool: Path, now: datetime, hours: int, *, reject: bool = True) -> list[Path]:
    limit = now.astimezone() - timedelta(hours=hours)
    return [
        path
        for path, entry in pending(spool, reject=reject)
        if datetime.fromisoformat(entry["at"]) < limit
    ]


# --- Delivery ---


async def deliver(
    spool: Path, notify: NotifySettings, write: Writer, now: datetime
) -> DeliveryReport:
    """Write every waiting message as a note; stop at the first sign the service is away."""
    delivered = rejected = 0
    for path, entry in pending(spool):
        message = Message(entry["id"], entry["host"], entry["at"], entry["text"])
        try:
            await write(note_for(message, notify))
        except Unreachable:
            waiting = len(pending(spool))
            return DeliveryReport(delivered, rejected, waiting, unreachable=True)
        except Rejected as error:
            attempts = int(entry.get("attempts", 0)) + 1
            if attempts >= notify.max_attempts:
                _reject(spool, path, f"refused {attempts} times: {error}")
                rejected += 1
            else:
                entry["attempts"] = attempts
                _write_private(path, json.dumps(entry, ensure_ascii=False, sort_keys=True).encode())
                print(
                    f"bm-notify: refused ({attempts}/{notify.max_attempts}): {error}",
                    file=sys.stderr,
                )
            continue
        path.unlink(missing_ok=True)
        delivered += 1
    return DeliveryReport(delivered, rejected, len(pending(spool)))


# Codes of `bm_write` that say "not now" rather than "no": the service is away, busy or locked.
TRANSIENT_CODES = {"OUTCOME_UNKNOWN", "UNAVAILABLE", "LOCKED"}
LEADING_CODE = re.compile(r"([A-Z][A-Z_]+):")


def classify_error(text: str) -> type[Exception] | None:
    """Sort a tool error: None = the note exists (delivered), else Unreachable or Rejected.

    The code of a refusal decides, never words inside the message: its ``try_instead`` names
    the note's own path, and a time stamp or id in a title can contain "409".
    """
    code: str | None = None
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        code = payload["error"]
    elif found := LEADING_CODE.match(text):
        code = found[1]
    if code == "ALREADY_EXISTS":
        return None
    if code is not None:
        return Unreachable if code in TRANSIENT_CODES else Rejected
    # No code at all: an error from below the tool (transport, HTTP).
    lowered = text.lower()
    if "409" in lowered or "busy" in lowered or "already running" in lowered:
        return Unreachable
    return Rejected


class McpWriter:  # pragma: no cover
    """The real writer: one ``bm_write`` per note on the running local service."""

    def __init__(self, url: str, project: str) -> None:
        from fastmcp import Client

        self._client = Client(url)
        self._project = project

    async def __aenter__(self) -> McpWriter:
        try:
            await asyncio.wait_for(self._client.__aenter__(), WRITE_TIMEOUT)
        except Exception as error:  # the service is down, starting, or not a BM service
            raise Unreachable(str(error)) from error
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.__aexit__(None, None, None)

    async def __call__(self, note: NoteRequest) -> None:
        from fastmcp.exceptions import ToolError

        arguments: dict[str, Any] = {
            "project": self._project,
            "title": note.title,
            "directory": note.directory,
            "content": note.content,
            "note_type": note.note_type,
            "tags": note.tags,
            "metadata": note.metadata,
            "overwrite": False,
            "output_format": "json",
        }
        try:
            await asyncio.wait_for(self._client.call_tool("bm_write", arguments), WRITE_TIMEOUT)
        except ToolError as error:
            verdict = classify_error(str(error))
            if verdict is None:
                return
            raise verdict(str(error)) from error
        except Exception as error:  # timeout, closed connection, protocol error
            raise Unreachable(str(error)) from error


# --- Command line ---


def summary(report: DeliveryReport) -> str:
    parts = [f"{report.delivered} delivered", f"{report.waiting} waiting"]
    if report.rejected:
        parts.append(f"{report.rejected} REJECTED")
    if report.unreachable:
        parts.append("service not reachable")
    return "bm-notify: " + ", ".join(parts)


def status(settings: HostSettings, now: datetime) -> int:
    spool = settings.notify.spool
    waiting = pending(spool, reject=False)
    old = stale(spool, now, settings.notify.stale_hours, reject=False)
    refused = sorted((spool / REJECTED).glob("*.json")) if (spool / REJECTED).is_dir() else []
    print(
        f"host {settings.host_name}: {len(waiting)} waiting "
        f"({len(old)} older than {settings.notify.stale_hours} h), {len(refused)} rejected; "
        f"spool {spool}"
    )
    return 2 if old or refused else 0


async def run(
    text: str,
    settings: HostSettings,
    writer_factory: Callable[[HostSettings], Any],
    now: datetime,
) -> int:
    lines = split_report(text)
    for line in lines:  # the journal gets the message before anything can go wrong
        print(f"[bm-notify {settings.host_name}] {line}", file=sys.stderr)
    notify = settings.notify
    if notify.project is None:
        return 0
    try:
        enqueue(notify.spool, make_messages(lines, settings.host_name, now))
    except OSError as error:
        print(f"bm-notify: cannot spool the message ({error}); it is above", file=sys.stderr)
        return 1
    old = stale(notify.spool, now, notify.stale_hours)
    if old:
        print(
            f"bm-notify: WARNING {len(old)} message(s) waiting for more than "
            f"{notify.stale_hours} h in {notify.spool}",
            file=sys.stderr,
        )
    if not pending(notify.spool):
        return 0
    try:
        async with writer_factory(settings) as write:
            report = await deliver(notify.spool, notify, write, now)
    except Unreachable as error:
        report = DeliveryReport(waiting=len(pending(notify.spool)), unreachable=True)
        print(f"bm-notify: service not reachable: {error}", file=sys.stderr)
    print(summary(report), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--settings", help="host settings file (default: see bm_host_settings)")
    parser.add_argument("--url", help="service URL; default from the settings")
    parser.add_argument("--status", action="store_true", help="show the spool and exit")
    args = parser.parse_args(argv)
    text = "" if args.status or sys.stdin.isatty() else sys.stdin.read()
    try:
        settings = load_host_settings(args.settings)
    except SettingsError as error:
        for line in split_report(text):
            print(f"[bm-notify] {line}", file=sys.stderr)
        print(f"bm-notify: Refused: {error}", file=sys.stderr)
        return 1
    now = datetime.now().astimezone()
    if args.status:
        return status(settings, now)

    def writer(current: HostSettings) -> McpWriter:
        assert current.notify.project is not None
        return McpWriter(args.url or current.bm_url, current.notify.project)

    return asyncio.run(run(text, settings, writer, now))


if __name__ == "__main__":
    sys.exit(main())
