"""Sensor: ``MLFRAME_PREWARM_HEAVY_LIBS`` env-var gate skips the heavy
import block in ``_prewarm_numba_cache_body``.

iter604 added the gate so short-lived processes that won't use neural /
shap can opt out of the 6-10s heavy-lib prewarm (lightning + shap +
mlframe.training.neural). The numba kernel prewarm is unaffected --
only the lightning/shap/neural import block is gated.
"""

from __future__ import annotations

from unittest import mock

import pytest

pytestmark = [pytest.mark.fast]


def _reload_warmup():
    """Reimport _core_numba_warmup so the module-level imports re-run."""
    import mlframe.metrics._core_numba_warmup as _w

    return _w


def test_gate_skips_lightning_import_when_set(monkeypatch):
    """With the gate ON ("0"/"false"/"skip"), the heavy-lib import block
    must not run -- verified by patching importlib.util.find_spec to a
    fatal sentinel and confirming the prewarm doesn't trip on it.

    Without the gate, find_spec would be called for "lightning" /
    "pytorch_lightning" / "shap"; with the gate the entire ``if not
    _skip_heavy:`` branch is skipped.
    """
    _w = _reload_warmup()

    monkeypatch.setenv("MLFRAME_PREWARM_HEAVY_LIBS", "0")
    calls = []

    def _fake_find_spec(name, *a, **kw):
        calls.append(name)
        return None  # CPU-only host

    with mock.patch("importlib.util.find_spec", side_effect=_fake_find_spec):
        _w._prewarm_numba_cache_body()

    # Verify find_spec was NOT called for the heavy libs (lightning /
    # pytorch_lightning / shap). The gate short-circuits before the
    # importlib lookup.
    heavy_lookups = [c for c in calls if c in ("lightning", "pytorch_lightning", "shap")]
    assert heavy_lookups == [], f"gate=0 should skip heavy-lib import block; observed lookups: {heavy_lookups}"


def test_gate_unset_runs_lightning_import(monkeypatch):
    """Default behavior: gate unset -> heavy-lib import block runs ->
    find_spec is consulted at least once for lightning/shap."""
    _w = _reload_warmup()
    monkeypatch.delenv("MLFRAME_PREWARM_HEAVY_LIBS", raising=False)
    calls = []

    def _fake_find_spec(name, *a, **kw):
        calls.append(name)
        return None

    with mock.patch("importlib.util.find_spec", side_effect=_fake_find_spec):
        _w._prewarm_numba_cache_body()

    heavy_lookups = [c for c in calls if c in ("lightning", "pytorch_lightning", "shap")]
    assert len(heavy_lookups) >= 1, f"gate unset (default) should consult find_spec for at least one of lightning/pytorch_lightning/shap; observed: {calls}"


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "skip", "Skip"])
def test_gate_accepts_multiple_skip_values(monkeypatch, value):
    """The gate parser is case-insensitive and accepts the documented
    skip aliases."""
    _w = _reload_warmup()
    monkeypatch.setenv("MLFRAME_PREWARM_HEAVY_LIBS", value)
    calls = []

    def _fake_find_spec(name, *a, **kw):
        calls.append(name)
        return None

    with mock.patch("importlib.util.find_spec", side_effect=_fake_find_spec):
        _w._prewarm_numba_cache_body()

    heavy_lookups = [c for c in calls if c in ("lightning", "pytorch_lightning", "shap")]
    assert heavy_lookups == [], f"gate={value!r} should skip; observed: {heavy_lookups}"


def test_gate_runs_when_value_is_truthy(monkeypatch):
    """Anything outside the documented skip set keeps the legacy
    behavior (consult find_spec)."""
    _w = _reload_warmup()
    monkeypatch.setenv("MLFRAME_PREWARM_HEAVY_LIBS", "1")
    calls = []

    def _fake_find_spec(name, *a, **kw):
        calls.append(name)
        return None

    with mock.patch("importlib.util.find_spec", side_effect=_fake_find_spec):
        _w._prewarm_numba_cache_body()

    heavy_lookups = [c for c in calls if c in ("lightning", "pytorch_lightning", "shap")]
    assert len(heavy_lookups) >= 1, f"gate=1 (truthy / non-skip) should NOT skip; observed: {calls}"
