"""Run the explicit Git sync for each project on a schedule and report what needs a person.

Basic Memory has no built-in timer for the peer sync. This script is meant to be started by
a systemd timer (see docs/BM006_GIT_SYNC_SCHEDULE.md). It calls ``bm_project(command="sync")``
on the running local MCP service, so it never opens the repositories itself.

What counts as needing attention:

* ``diverged`` and ``remote_rewritten``: both sides published different history. Nothing is
  merged automatically; a person reviews and publishes a merge. Reported at once.
* ``needs_review``: automatic merging is on, but notes conflict and a person decides (the
  report names kinds and paths, never contents). A successful ``merged`` is routine and silent.
* ``invalid_incoming`` and ``local_changed``: the peer sent something that failed validation,
  or the local side moved during the attempt. Reported at once.
* ``transport`` and ``interrupted``: normal while offline. Reported only after
  ``--tolerate`` consecutive failures, so a closed laptop does not raise an alarm.
* The service itself not answering counts as a transport failure for every project.

A problem is reported once when it appears and once when it clears, not on every run.
The report goes to stderr and, if given, to ``--notify-cmd`` on its standard input.
Exit status: 0 nothing to do, 3 at least one project needs attention.

What differs from machine to machine (service URL, projects to visit, tolerance, state file) is
read from the host settings file (``bm_host_settings.py``); a flag on the command line wins.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

IMMEDIATE = {
    "diverged",
    "remote_rewritten",
    "needs_review",
    "invalid_incoming",
    "local_changed",
}
TOLERATED = {"transport", "interrupted"}

SyncCall = Callable[[str], Awaitable[dict[str, Any]]]


class NotConfigured(Exception):
    """The project has no peer, so there is nothing to synchronize."""


class Busy(Exception):
    """Another sync of the same project is running; try again next time."""


@dataclass(frozen=True)
class Verdict:
    project: str
    problem: str | None  # None = fine
    detail: str = ""
    # A tolerated failure says nothing about the earlier problem: keep it as it was.
    keep_previous: bool = False


def classify(project: str, result: dict[str, Any], failures: int, tolerate: int) -> Verdict:
    """Turn one sync answer into a verdict. ``failures`` counts earlier consecutive failures."""
    attempt = result.get("last_attempt") or {}
    kind = attempt.get("kind")
    if kind == "completed":
        outcome = attempt.get("outcome")
        if outcome in IMMEDIATE:
            review = result.get("open_review") or {}
            counts = ", ".join(f"{k}={v}" for k, v in sorted(review.get("conflicts", {}).items()))
            paths = ", ".join(review.get("paths", [])[:5])
            extra = f" review={review['id']} {counts} [{paths}]" if review else ""
            return Verdict(project, outcome, f"outcome={outcome}{extra}")
        return Verdict(project, None, f"outcome={outcome}")
    if kind == "failed":
        reason = attempt.get("reason")
        if reason in IMMEDIATE:
            return Verdict(project, reason, f"reason={reason}")
        if failures + 1 >= tolerate:
            return Verdict(project, f"unreachable:{reason}", f"{failures + 1} failures in a row")
        return Verdict(
            project, None, f"tolerated {reason} ({failures + 1}/{tolerate})", keep_previous=True
        )
    if result.get("relationship") == "diverged":
        return Verdict(project, "diverged", "relationship=diverged")
    return Verdict(project, None, f"kind={kind}")


def is_busy(error_text: str) -> bool:
    """Another sync of the same project is running (HTTP 409 or the service's own wording)."""
    lowered = error_text.lower()
    return "409" in lowered or "busy" in lowered or "already running" in lowered


def load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


async def run_once(
    projects: list[str],
    call: SyncCall,
    state: dict[str, Any],
    *,
    tolerate: int = 3,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[list[str], int]:
    """Sync each project, update ``state`` in place, return (messages, attention count)."""
    messages: list[str] = []
    for project in projects:
        entry = state.setdefault(project, {"problem": None, "failures": 0})
        try:
            result = await call(project)
        except NotConfigured:
            state.pop(project, None)
            continue
        except Busy:
            continue
        except Exception:  # service down, timeout, protocol error
            result = {"last_attempt": {"kind": "failed", "reason": "transport"}}
        verdict = classify(project, result, entry["failures"], tolerate)
        failed = (result.get("last_attempt") or {}).get("kind") == "failed"
        entry["failures"] = entry["failures"] + 1 if failed else 0
        previous = entry["problem"]
        if verdict.keep_previous:
            # Trigger: an offline hiccup while a person-needed problem (needs_review,
            #   diverged) is open.
            # Why: reporting "wieder in Ordnung" would be false, and the next run would
            #   report the same problem again.
            # Outcome: the open problem stays open and nothing is reported.
            continue
        if verdict.problem != previous:
            if verdict.problem is not None:
                messages.append(f"[BM sync] {project}: {verdict.problem} ({verdict.detail})")
            else:
                messages.append(f"[BM sync] {project}: wieder in Ordnung ({verdict.detail})")
            entry["problem"] = verdict.problem
        if verdict.problem is None:
            entry["last_ok"] = now().isoformat(timespec="seconds")
    attention = sum(1 for e in state.values() if isinstance(e, dict) and e.get("problem"))
    return messages, attention


def notify(command: str | None, messages: list[str], *, always: bool = False) -> None:
    """Print the report and hand it to the notify command.

    With ``always`` the command also runs when there is nothing new to say (empty input), so a
    command that keeps a queue can deliver what is waiting as soon as its target is back.
    """
    text = "\n".join(messages)
    if messages:
        print(text, file=sys.stderr)
    if command and (messages or always):
        subprocess.run(command, shell=True, input=text.encode(), check=False, timeout=120)


class McpTransport:  # pragma: no cover
    """The real transport: the running local service, opened once per run."""

    def __init__(self, url: str) -> None:
        from fastmcp import Client

        self._client = Client(url)

    async def __aenter__(self) -> McpTransport:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.__aexit__(None, None, None)

    async def _json(self, arguments: dict[str, Any]) -> Any:
        from fastmcp.exceptions import ToolError
        from mcp.types import TextContent

        try:
            out = await self._client.call_tool("bm_project", {**arguments, "output_format": "json"})
        except ToolError as error:
            text = str(error)
            if "No Git peer" in text or "not owned by this Git host" in text:
                raise NotConfigured from error
            if is_busy(text):
                raise Busy from error
            raise
        if out.data is not None:
            return out.data
        block = out.content[0]
        if not isinstance(block, TextContent):
            raise ValueError("unexpected non-text answer from bm_project")
        return json.loads(block.text)

    async def sync(self, project: str) -> dict[str, Any]:
        return await self._json({"command": "sync", "project": project})

    async def names(self) -> list[str]:
        # bm_project(list) answers {"projects": [...]}, not a bare list.
        answer = await self._json({"command": "list"})
        return [entry["name"] for entry in answer["projects"]]


SERVICE = "dienst"  # reported in place of a project when nothing else is known


async def service_down(project: str) -> dict[str, Any]:
    """Stands in for the sync call while the service does not answer at all."""
    raise ConnectionError("the Basic Memory service does not answer")


def projects_to_visit(
    configured: tuple[str, ...] | None, listed: list[str] | None, state: dict[str, Any]
) -> list[str]:
    """The projects to sync: the configured ones, else what the service lists.

    With the service down and nothing configured, the projects seen on earlier passes (the
    state file) stand in, so their failures are counted; with no history at all the service
    itself is the one thing to report.
    """
    if configured:
        return list(configured)
    if listed is not None:
        return listed
    return [name for name, entry in state.items() if isinstance(entry, dict)] or [SERVICE]


async def main_async(args: argparse.Namespace) -> int:  # pragma: no cover
    from contextlib import AsyncExitStack

    from bm_host_settings import load_host_settings, resolve_sync_options

    settings = load_host_settings(args.settings)
    options = resolve_sync_options(
        settings,
        url=args.url,
        state=args.state,
        tolerate=args.tolerate,
        projects=args.project,
    )
    state = load_state(options.state)
    async with AsyncExitStack() as stack:
        call: SyncCall = service_down
        listed: list[str] | None = None
        try:
            transport = await stack.enter_async_context(McpTransport(options.url))
            listed = None if options.projects else await transport.names()
            call = transport.sync
        except Exception as error:  # the service itself does not answer: a failure per project
            print(f"git_sync_watch: service not reachable: {error}", file=sys.stderr)
        messages, attention = await run_once(
            projects_to_visit(options.projects, listed, state),
            call,
            state,
            tolerate=options.tolerate,
        )
    save_state(options.state, state)
    notify(args.notify_cmd, messages, always=args.notify_always)
    return 3 if attention else 0


def main() -> int:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bm_host_settings import SettingsError

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--settings", help="host settings file (see bm_host_settings.py)")
    parser.add_argument("--url", help="service URL; default from the settings")
    parser.add_argument("--state", help="state file; default from the settings")
    parser.add_argument("--project", action="append", help="repeatable; default: from the settings")
    parser.add_argument("--tolerate", type=int, help="failures before 'unreachable'")
    parser.add_argument("--notify-cmd", help="shell command that reads the report on stdin")
    parser.add_argument(
        "--notify-always",
        action="store_true",
        help="run the notify command on every pass, with an empty report if nothing is new",
    )
    try:
        return asyncio.run(main_async(parser.parse_args()))
    except SettingsError as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
