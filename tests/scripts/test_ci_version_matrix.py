"""scripts/ci_version_matrix.py picks the floor, the newest of the floor's series and the latest release, once each, and writes the badge."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest
from packaging.version import Version

_SPEC = importlib.util.spec_from_file_location("ci_version_matrix", Path(__file__).resolve().parents[2] / "scripts" / "ci_version_matrix.py")
assert _SPEC is not None and _SPEC.loader is not None
cvm = importlib.util.module_from_spec(_SPEC)
sys.modules["ci_version_matrix"] = cvm
_SPEC.loader.exec_module(cvm)


def _versions(*texts: str) -> list[Version]:
    """Versions from strings."""
    return [Version(t) for t in texts]


def test_the_floor_is_the_lowest_version_the_lock_resolves() -> None:
    """A package locked at several versions (one per marker environment) reports the lowest; an absent package raises."""
    lock = '[[package]]\nname = "polars"\nversion = "1.44.1"\n\n[[package]]\nname = "polars"\nversion = "1.36.1"\n\n[[package]]\nname = "polars-ds"\nversion = "0.13.0"\n'
    assert cvm.lock_floor(lock, "polars") == "1.36.1"
    assert cvm.lock_floor(lock, "polars-ds") == "0.13.0"
    with pytest.raises(LookupError):
        cvm.lock_floor(lock, "pandas")


def test_floor_series_newest_and_latest_are_three_legs_with_latest_labelled() -> None:
    """polars-like case: a floor in the 1.x series, a newer 1.x, and a 2.x latest give three legs."""
    legs = cvm.plan("1.36.1", _versions("1.36.1", "1.40.0", "1.44.2", "2.0.0"))
    assert [(leg.version, leg.label) for leg in legs] == [("1.36.1", "1.36.1 (min)"), ("1.44.2", "1.44.2"), ("2.0.0", "2.0.0 (latest)")]


def test_a_latest_that_is_the_series_newest_is_not_run_twice() -> None:
    """When no new major exists the series' newest release is the latest: two legs."""
    legs = cvm.plan("1.36.1", _versions("1.36.1", "1.44.2"))
    assert [leg.label for leg in legs] == ["1.36.1 (min)", "1.44.2 (latest)"]


def test_a_series_whose_newest_release_is_the_floor_collapses_into_it() -> None:
    """pandas-like case: the locked 2.3.3 is the last 2.x, so only the floor and the 3.x latest remain."""
    legs = cvm.plan("2.3.3", _versions("2.2.3", "2.3.3", "3.0.0", "3.0.6"))
    assert [leg.label for leg in legs] == ["2.3.3 (min)", "3.0.6 (latest)"]


def test_a_floor_that_is_also_the_latest_is_one_leg() -> None:
    """Nothing newer than the lock: a single leg that says so."""
    assert [leg.label for leg in cvm.plan("2.0.0", _versions("1.9.0", "2.0.0"))] == ["2.0.0 (min, latest)"]


def test_prereleases_and_fully_yanked_releases_are_ignored() -> None:
    """Only final releases with at least one live file are candidates."""
    releases = {
        "1.0.0": [{"yanked": False}],
        "1.1.0rc1": [{"yanked": False}],
        "1.2.0": [{"yanked": True}],
        "1.3.0.dev1": [{"yanked": False}],
        "1.4.0": [{"yanked": True}, {"yanked": False}],
        "not-a-version": [{"yanked": False}],
        "1.5.0": [],
    }
    assert [str(v) for v in cvm.usable_releases(releases)] == ["1.0.0", "1.4.0"]


def test_matrix_and_badge_documents() -> None:
    """The include list carries version, label and Python; the badge message lists the labels without the min marker and is coloured by the result."""
    legs = cvm.plan("1.36.1", _versions("1.36.1", "1.44.2", "2.0.1"))
    include = cvm.matrix(legs, "3.12")["include"]
    assert include[0] == {"version": "1.36.1", "label": "1.36.1 (min)", "python-version": "3.12"}
    ok = cvm.badge("polars", include, "success")
    assert ok == {"schemaVersion": 1, "label": "polars", "message": "1.36.1 | 1.44.2 | 2.0.1 (latest)", "color": "brightgreen"}
    assert cvm.badge("polars", include, "failure")["color"] == "red"
    assert cvm.badge("polars", include, "something-else")["color"] == "lightgrey"


def test_fetch_retries_network_errors_then_returns_the_release_map() -> None:
    """Two failures then a good answer: the map comes back and the sleeps were taken."""
    calls = {"n": 0}
    sleeps: list[float] = []

    def opener(url, timeout):
        """Fail twice, then serve a minimal PyPI document."""
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("boom")
        return io.BytesIO(json.dumps({"releases": {"1.0.0": [{"yanked": False}]}}).encode())

    assert cvm.fetch_releases("x", opener=opener, sleep=sleeps.append) == {"1.0.0": [{"yanked": False}]}
    assert calls["n"] == 3 and len(sleeps) == 2


def test_fetch_gives_up_with_a_clear_error() -> None:
    """Persistent failure raises RuntimeError naming the package."""

    def opener(url, timeout):
        """Always fail."""
        raise urllib.error.URLError("down")

    with pytest.raises(RuntimeError, match="polars"):
        cvm.fetch_releases("polars", opener=opener, sleep=lambda _s: None)
