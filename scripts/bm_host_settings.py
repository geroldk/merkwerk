"""Settings of one host for the scheduled Git sync and its notifications (BM-009, V3/V5).

Everything that differs from one machine to the next lives in one small JSON file, not in
scripts and not in `config.json`: the host name, which projects the scheduled sync visits,
and where a message about a waiting sync goes. `bm git backup` saves `config.json` only, so
these settings stay with the machine, and a restore on another host does not inherit them.

    {
      "host_name": "ryzen",
      "bm_url": "http://127.0.0.1:8765/mcp",
      "notify": {
        "project": "meldungen-ryzen",
        "directory": "abgleich",
        "spool": "~/.local/state/bm-git-sync/spool",
        "stale_hours": 24,
        "max_attempts": 5
      },
      "sync": {
        "projects": ["main", "pkm"],
        "tolerate": 3,
        "state": "~/.local/state/bm-git-sync/state.json"
      }
    }

Every key is optional. Without the file a host behaves as before: all projects are visited,
messages go to the journal only. ``notify.project`` null (or absent) means "journal only";
which project it names, and whether that project has a peer, is the operator's choice (one
shared project for all hosts, or one per host). ``sync.projects`` null means every project
the service knows; a project without a peer is skipped by the sync itself.

The file is looked for at ``--settings``, else ``$MERKWERK_HOST_SETTINGS``, else
``$XDG_CONFIG_HOME/merkwerk/host.json`` (``~/.config/merkwerk/host.json``). Unknown keys are
refused, so a misspelt key cannot silently fall back to a default.

This module uses only the standard library: it is copied next to the scripts that import it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8765/mcp"
HOST_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,30}")
STATE_DIRECTORY = Path("~/.local/state/bm-git-sync")
ENVIRONMENT_VARIABLE = "MERKWERK_HOST_SETTINGS"


class SettingsError(ValueError):
    """The settings file is unusable; nothing falls back silently."""


@dataclass(frozen=True)
class NotifySettings:
    project: str | None = None
    directory: str = "abgleich"
    spool: Path = STATE_DIRECTORY / "spool"
    stale_hours: int = 24
    max_attempts: int = 5


@dataclass(frozen=True)
class SyncSettings:
    projects: tuple[str, ...] | None = None
    tolerate: int = 3
    state: Path = STATE_DIRECTORY / "state.json"


@dataclass(frozen=True)
class HostSettings:
    host_name: str
    bm_url: str = DEFAULT_URL
    notify: NotifySettings = field(default_factory=NotifySettings)
    sync: SyncSettings = field(default_factory=SyncSettings)
    source: Path | None = None  # the file these came from; None = defaults


def default_host_name() -> str:
    """The machine's short name, made fit for `HOST_NAME`."""
    short = socket.gethostname().split(".")[0].lower()
    cleaned = re.sub(r"[^a-z0-9-]+", "-", short).strip("-")[:31]
    return cleaned if HOST_NAME.fullmatch(cleaned) else "host"


def settings_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(ENVIRONMENT_VARIABLE):
        return Path(os.environ[ENVIRONMENT_VARIABLE]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser()
    return base / "merkwerk" / "host.json"


def _section(data: dict[str, Any], key: str, allowed: set[str], where: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SettingsError(f"{where}{key} must be an object")
    unknown = sorted(str(key) for key in set(value) - allowed)
    if unknown:
        raise SettingsError(f"unknown key(s) in {where}{key}: {', '.join(unknown)}")
    return value


def _text(value: Any, name: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"{name} must be a non-empty string")
    if pattern is not None and not pattern.fullmatch(value):
        raise SettingsError(f"{name} {value!r} does not match {pattern.pattern}")
    return value


def _whole(value: Any, name: str, *, minimum: int) -> int:
    # bool is an int in Python; "true" is not a count.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SettingsError(f"{name} must be a whole number of at least {minimum}")
    return value


def _project_name(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) > 64 or text.startswith(".") or re.search(r"[/\\\x00-\x1f]", text):
        raise SettingsError(f"{name} {text!r} is not a project name")
    return text


FOLDER_PART = re.compile(r"[\w][\w .-]*")


def _folder(value: Any, name: str) -> str:
    """A folder inside the notify project, one `bm_write` accepts: relative, plain names."""
    text = _text(value, name).strip("/")
    # Letters, digits, space, '.', '-', '_' (what bm_write keeps unchanged); no dot-first part,
    # no '..': a name that gets sanitized would make every note be refused.
    if not text or not all(FOLDER_PART.fullmatch(part) for part in text.split("/")):
        raise SettingsError(
            f"{name} {value!r} must be a relative folder of letters, digits, spaces, '.', '-', '_' "
            "that does not start with a dot"
        )
    return text


def _absolute(value: Any, name: str) -> Path:
    path = Path(_text(value, name)).expanduser()
    if not path.is_absolute():
        raise SettingsError(f"{name} {value!r} must be an absolute path (or start with ~)")
    return path


def parse_settings(data: Any, *, source: Path | None = None) -> HostSettings:
    """Validate the decoded JSON and fill the defaults."""
    if not isinstance(data, dict):
        raise SettingsError("the settings file must hold a JSON object")
    unknown = sorted(str(key) for key in set(data) - {"host_name", "bm_url", "notify", "sync"})
    if unknown:
        raise SettingsError(f"unknown key(s): {', '.join(unknown)}")
    notify = _section(
        data, "notify", {"project", "directory", "spool", "stale_hours", "max_attempts"}, ""
    )
    sync = _section(data, "sync", {"projects", "tolerate", "state"}, "")
    defaults_notify, defaults_sync = NotifySettings(), SyncSettings()
    projects = sync.get("projects")
    if projects is not None:
        if not isinstance(projects, list) or not projects:
            raise SettingsError("sync.projects must be a non-empty list, or null for all")
        projects = tuple(_project_name(p, "sync.projects entry") for p in projects)
        if len(set(projects)) != len(projects):
            raise SettingsError("sync.projects names a project twice")
    project = notify.get("project")
    return HostSettings(
        host_name=_text(data["host_name"], "host_name", pattern=HOST_NAME)
        if data.get("host_name") is not None
        else default_host_name(),
        bm_url=_text(data.get("bm_url", DEFAULT_URL), "bm_url"),
        notify=NotifySettings(
            project=None if project is None else _project_name(project, "notify.project"),
            directory=_folder(
                notify.get("directory", defaults_notify.directory), "notify.directory"
            ),
            spool=_absolute(notify.get("spool", str(defaults_notify.spool)), "notify.spool"),
            stale_hours=_whole(
                notify.get("stale_hours", defaults_notify.stale_hours),
                "notify.stale_hours",
                minimum=1,
            ),
            max_attempts=_whole(
                notify.get("max_attempts", defaults_notify.max_attempts),
                "notify.max_attempts",
                minimum=1,
            ),
        ),
        sync=SyncSettings(
            projects=projects,
            tolerate=_whole(
                sync.get("tolerate", defaults_sync.tolerate), "sync.tolerate", minimum=1
            ),
            state=_absolute(sync.get("state", str(defaults_sync.state)), "sync.state"),
        ),
        source=source,
    )


def load_host_settings(explicit: str | os.PathLike[str] | None = None) -> HostSettings:
    """Read the settings file; a missing default file means defaults, a missing named one is an error."""
    path = settings_path(explicit)
    named = explicit is not None or bool(os.environ.get(ENVIRONMENT_VARIABLE))
    try:
        text = path.read_text()
    except FileNotFoundError:
        if named:
            raise SettingsError(f"settings file {path} does not exist") from None
        return parse_settings({})
    except OSError as error:
        raise SettingsError(f"cannot read {path}: {error}") from error
    try:
        data = json.loads(text)
    except ValueError as error:
        raise SettingsError(f"{path} is not valid JSON: {error}") from error
    try:
        return parse_settings(data, source=path)
    except SettingsError as error:
        raise SettingsError(f"{path}: {error}") from error


@dataclass(frozen=True)
class SyncOptions:
    url: str
    state: Path
    tolerate: int
    projects: tuple[str, ...] | None  # None = ask the service for every project


def resolve_sync_options(
    settings: HostSettings,
    *,
    url: str | None = None,
    state: str | None = None,
    tolerate: int | None = None,
    projects: list[str] | None = None,
) -> SyncOptions:
    """Command-line flags win over the settings file, which wins over the defaults."""
    return SyncOptions(
        url=url or settings.bm_url,
        state=Path(state).expanduser() if state else settings.sync.state,
        tolerate=tolerate if tolerate is not None else settings.sync.tolerate,
        projects=tuple(projects) if projects else settings.sync.projects,
    )


def describe(settings: HostSettings) -> str:
    notify = settings.notify
    where = (
        f"project {notify.project!r}, folder {notify.directory!r}"
        if notify.project
        else "journal only (no notify project)"
    )
    visited = ", ".join(settings.sync.projects) if settings.sync.projects else "all projects"
    return "\n".join(
        [
            f"host:    {settings.host_name}",
            f"source:  {settings.source or 'defaults (no settings file)'}",
            f"service: {settings.bm_url}",
            f"notify:  {where}; spool {notify.spool}; stale after {notify.stale_hours} h; "
            f"gives up after {notify.max_attempts} refusals",
            f"sync:    {visited}; tolerate {settings.sync.tolerate}; state {settings.sync.state}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the effective settings of this host.")
    parser.add_argument("--settings", help="settings file (default: see module documentation)")
    args = parser.parse_args()
    try:
        print(describe(load_host_settings(args.settings)))
    except SettingsError as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
