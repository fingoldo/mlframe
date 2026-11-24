"""Cluster 10 regression: the version-check policy in the benchmark aggregator (audit 2026-11-20, findings 10-F5 and 10-F7).

10-F5: a capture missing from the manifest is reported as a skipped comparison, not silently passed.
10-F7: captures older than MAX_VERSION_AGE_DAYS are reported; the constant is no longer dead.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mlframe.feature_selection._benchmarks import _aggregate as agg


def _capture(tag: str, sha: str, ts: datetime) -> dict:
    """Build one manifest capture with the fields _check_versions reads."""
    return {
        "tag": tag,
        "git_sha": sha,
        "capture_ts": ts.isoformat(),
        "cpu_model": "cpu",
        "versions": {"numpy": "2.2.6", "numba": "0.62.1", "cupy": "14", "sklearn": "1.8.0"},
    }


@pytest.fixture
def fake_manifest(monkeypatch):
    """Serve an in-memory manifest in place of MANIFEST.json."""

    def _install(captures: list[dict]) -> None:
        """Replace the manifest loader with one returning these captures."""
        monkeypatch.setattr(agg, "_load", lambda _path: {"captures": captures})

    return _install


def test_missing_capture_is_reported_as_skipped(fake_manifest, caplog):
    """A before key absent from the manifest logs that the version check was skipped (10-F5)."""
    now = datetime(2026, 11, 24, tzinfo=timezone.utc)
    fake_manifest([_capture("after", "a1", now)])
    before = {"tag": "before", "git_sha": "b0"}
    after = {"tag": "after", "git_sha": "a1"}
    with caplog.at_level("WARNING"):
        out = agg._check_versions(before, after, now=now)
    assert out == []
    assert any("version check skipped" in r.getMessage() and "no before capture" in r.getMessage() for r in caplog.records)


def test_stale_capture_is_reported(fake_manifest):
    """A capture older than MAX_VERSION_AGE_DAYS produces a warning (10-F7)."""
    now = datetime(2026, 11, 24, tzinfo=timezone.utc)
    old = now - timedelta(days=agg.MAX_VERSION_AGE_DAYS + 30)
    fake_manifest([_capture("before", "b0", old), _capture("after", "a1", now)])
    out = agg._check_versions({"tag": "before", "git_sha": "b0"}, {"tag": "after", "git_sha": "a1"}, now=now)
    assert any("before capture" in w and "days old" in w for w in out)
    assert not any("after capture" in w for w in out)


def test_fresh_captures_produce_no_age_warning(fake_manifest):
    """Control: recent captures with equal versions produce no warning (10-F7)."""
    now = datetime(2026, 11, 24, tzinfo=timezone.utc)
    recent = now - timedelta(days=1)
    fake_manifest([_capture("before", "b0", recent), _capture("after", "a1", recent)])
    out = agg._check_versions({"tag": "before", "git_sha": "b0"}, {"tag": "after", "git_sha": "a1"}, now=now)
    assert out == []
