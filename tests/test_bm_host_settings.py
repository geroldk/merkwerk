"""BM-009 (V3): what differs per machine is read from one file, strictly."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bm_host_settings as hs  # noqa: E402  # ty: ignore[unresolved-import]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(hs.ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_without_a_file_the_defaults_apply_and_nothing_is_notified():
    settings = hs.load_host_settings()
    assert hs.HOST_NAME.fullmatch(settings.host_name)
    assert settings.source is None
    assert settings.bm_url == hs.DEFAULT_URL
    assert settings.notify.project is None  # journal only
    assert settings.sync.projects is None  # every project the service knows
    assert settings.sync.tolerate == 3
    assert "journal only" in hs.describe(settings)


def test_the_default_file_location_follows_xdg(tmp_path):
    write(
        tmp_path / "xdg" / "merkwerk" / "host.json",
        '{"host_name": "ryzen", "notify": {"project": "meldungen-ryzen"}}',
    )
    settings = hs.load_host_settings()
    assert (settings.host_name, settings.notify.project) == ("ryzen", "meldungen-ryzen")
    assert settings.source == tmp_path / "xdg" / "merkwerk" / "host.json"


def test_every_value_can_be_set_and_home_is_expanded(tmp_path):
    path = write(
        tmp_path / "host.json",
        """{
          "host_name": "nuc1",
          "bm_url": "http://127.0.0.1:9999/mcp",
          "notify": {"project": "meldungen", "directory": "abgleich/nuc1",
                     "spool": "~/spool", "stale_hours": 6, "max_attempts": 2},
          "sync": {"projects": ["main", "PKM"], "tolerate": 5, "state": "~/state.json"}
        }""",
    )
    settings = hs.load_host_settings(path)
    assert settings.host_name == "nuc1" and settings.bm_url.endswith(":9999/mcp")
    assert settings.notify.directory == "abgleich/nuc1"
    assert settings.notify.spool == tmp_path / "home" / "spool"
    assert (settings.notify.stale_hours, settings.notify.max_attempts) == (6, 2)
    assert settings.sync.projects == ("main", "PKM")
    assert settings.sync.tolerate == 5
    assert settings.sync.state == tmp_path / "home" / "state.json"
    assert "main, PKM" in hs.describe(settings)


def test_the_path_comes_from_the_flag_before_the_environment_before_xdg(monkeypatch, tmp_path):
    flag = write(tmp_path / "flag.json", '{"host_name": "flag"}')
    env = write(tmp_path / "env.json", '{"host_name": "env"}')
    write(tmp_path / "xdg" / "merkwerk" / "host.json", '{"host_name": "xdg"}')
    assert hs.load_host_settings().host_name == "xdg"
    monkeypatch.setenv(hs.ENVIRONMENT_VARIABLE, str(env))
    assert hs.load_host_settings().host_name == "env"
    assert hs.load_host_settings(flag).host_name == "flag"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"hostname": "x"}', "unknown key.*hostname"),
        ('{"notify": {"projekt": "x"}}', "unknown key.*projekt"),
        ('{"sync": {"project": ["x"]}}', "unknown key.*project"),
        ('{"notify": []}', "notify must be an object"),
        ("[]", "must hold a JSON object"),
        ('{"host_name": "Ryzen"}', "host_name"),
        ('{"host_name": ""}', "host_name"),
        ('{"notify": {"project": "a/b"}}', "not a project name"),
        ('{"notify": {"project": ".hidden"}}', "not a project name"),
        ('{"notify": {"directory": "../up"}}', "relative folder"),
        ('{"notify": {"directory": "a//b"}}', "relative folder"),
        ('{"notify": {"directory": ".git"}}', "does not start with a dot"),
        ('{"notify": {"directory": "meldungen@nuc1"}}', "relative folder"),
        ('{"notify": {"directory": "C++"}}', "relative folder"),
        ('{"notify": {"spool": "spool"}}', "absolute path"),
        ('{"sync": {"state": "state.json"}}', "absolute path"),
        ('{"notify": {"stale_hours": 0}}', "at least 1"),
        ('{"notify": {"max_attempts": true}}', "whole number"),
        ('{"sync": {"tolerate": "3"}}', "whole number"),
        ('{"sync": {"projects": []}}', "non-empty list"),
        ('{"sync": {"projects": "main"}}', "non-empty list"),
        ('{"sync": {"projects": ["main", "main"]}}', "twice"),
        ('{"bm_url": 5}', "bm_url"),
    ],
)
def test_a_misspelt_or_malformed_file_is_refused_not_defaulted(tmp_path, text, message):
    path = write(tmp_path / "host.json", text)
    with pytest.raises(hs.SettingsError, match=message) as error:
        hs.load_host_settings(path)
    assert str(path) in str(error.value)


def test_a_named_file_must_exist_and_be_json(tmp_path, monkeypatch):
    with pytest.raises(hs.SettingsError, match="does not exist"):
        hs.load_host_settings(tmp_path / "missing.json")
    monkeypatch.setenv(hs.ENVIRONMENT_VARIABLE, str(tmp_path / "also-missing.json"))
    with pytest.raises(hs.SettingsError, match="does not exist"):
        hs.load_host_settings()
    broken = write(tmp_path / "broken.json", "{not json")
    with pytest.raises(hs.SettingsError, match="not valid JSON"):
        hs.load_host_settings(broken)
    unreadable = tmp_path / "dir.json"
    unreadable.mkdir()
    with pytest.raises(hs.SettingsError, match="cannot read"):
        hs.load_host_settings(unreadable)


@pytest.mark.parametrize(
    ("machine", "expected"), [("Ryzen.fritz.box", "ryzen"), ("NUC_1", "nuc-1"), ("!!!", "host")]
)
def test_the_default_host_name_is_cleaned(monkeypatch, machine, expected):
    monkeypatch.setattr(hs.socket, "gethostname", lambda: machine)
    assert hs.default_host_name() == expected


def test_flags_win_over_the_file_which_wins_over_the_defaults(tmp_path):
    settings = hs.parse_settings(
        {"bm_url": "http://file/mcp", "sync": {"projects": ["a"], "tolerate": 7, "state": "/s"}}
    )
    from_file = hs.resolve_sync_options(settings)
    assert (from_file.url, from_file.projects, from_file.tolerate) == ("http://file/mcp", ("a",), 7)
    assert from_file.state == Path("/s")
    flags = hs.resolve_sync_options(
        settings, url="http://flag/mcp", state="/t", tolerate=1, projects=["b", "c"]
    )
    assert (flags.url, flags.projects, flags.tolerate) == ("http://flag/mcp", ("b", "c"), 1)
    assert flags.state == Path("/t")
    assert hs.resolve_sync_options(hs.parse_settings({})).projects is None


def test_main_shows_the_effective_settings_or_refuses(tmp_path, monkeypatch, capsys):
    good = write(tmp_path / "good.json", '{"host_name": "ryzen"}')
    monkeypatch.setattr(sys, "argv", ["bm_host_settings.py", "--settings", str(good)])
    assert hs.main() == 0
    assert "host:    ryzen" in capsys.readouterr().out
    bad = write(tmp_path / "bad.json", '{"typo": 1}')
    monkeypatch.setattr(sys, "argv", ["bm_host_settings.py", "--settings", str(bad)])
    assert hs.main() == 1
    assert "Refused" in capsys.readouterr().err
