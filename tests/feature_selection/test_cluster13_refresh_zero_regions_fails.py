"""Cluster 13 regression: a new-registry refresh that tunes zero regions exits non-zero (audit 2026-11-20, finding 13-F27).

The function used to return 0 whatever tune_spec produced. It now returns 1 for zero regions and 0 when regions were tuned.
pyutilz's registry functions are replaced by stubs, so no real sweep runs.
"""

import pyutilz.performance.kernel_tuning as kt

from mlframe.feature_selection._benchmarks.kernel_tuning_cache import cli


def _patch(monkeypatch, regions):
    """Stub the registry so tune_spec reports the given region count for a known kernel."""
    monkeypatch.setattr(kt, "discover_tuners", lambda package=None: None)
    monkeypatch.setattr(kt, "get_registry", lambda: {"k": object()})
    monkeypatch.setattr(kt, "tune_spec", lambda spec, force=False, skip_existing=True: regions)


def test_zero_regions_is_a_failure(monkeypatch, capsys):
    """tune_spec returning zero regions makes the refresh return 1 (13-F27)."""
    _patch(monkeypatch, regions=0)
    assert cli._refresh_via_new_registry("k") == 1
    assert "0 regions" in capsys.readouterr().err


def test_tuned_regions_still_succeed(monkeypatch):
    """Control: a successful tune with regions returns 0 (13-F27)."""
    _patch(monkeypatch, regions=3)
    assert cli._refresh_via_new_registry("k") == 0
