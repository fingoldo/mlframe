"""A probe exception of the accuracy gate keeps the candidate (fail-open) but is counted, and ``fit_meta_`` reports the count; ``MLFRAME_FE_PROBE_STRICT`` makes the exception propagate.

Before this, a probe that failed on every candidate looked like a gate that had kept everything on merit: the only trace was a warning line."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_accuracy_gate as gate


def _frame(n: int = 600):
    """A binary target that depends on the source column, and one engineered column."""
    rng = np.random.default_rng(0)
    src = rng.standard_normal(n)
    y = (src + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    return src[:, None], (src**2)[:, None], y


def _boom(*args, **kwargs):
    """A probe that cannot fit."""
    raise ValueError("the probe cannot fit")


def test_a_probe_exception_keeps_the_candidate_and_is_counted(monkeypatch):
    """Fail-open: ``None`` (cannot assess) comes back, the candidate is kept by ``keep_engineered_over_source``, and the process-wide count grows by one."""
    monkeypatch.delenv("MLFRAME_FE_PROBE_STRICT", raising=False)
    monkeypatch.setattr(gate, "_held_out_probe_score", _boom)
    src, eng, y = _frame()
    before = gate.probe_failure_count()
    assert gate.measure_feature_uplift(src, eng, y, classification=True) is None
    assert gate.probe_failure_count() == before + 1
    assert gate.keep_engineered_over_source(src, eng, y, threshold=0.015) is True
    assert gate.probe_failure_count() >= before + 2


def test_a_measured_zero_uplift_is_not_a_failure():
    """A probe that ran and found nothing is a measurement, not a failure: the count does not move."""
    src, eng, y = _frame()
    before = gate.probe_failure_count()
    assert gate.measure_feature_uplift(src, eng, y, classification=True) is not None
    assert gate.probe_failure_count() == before


@pytest.mark.parametrize("value", ["1", "true", "on"])
def test_strict_mode_raises_instead_of_keeping(monkeypatch, value):
    """With ``MLFRAME_FE_PROBE_STRICT`` on, the probe exception propagates and the count does not move (the failure is not absorbed)."""
    monkeypatch.setenv("MLFRAME_FE_PROBE_STRICT", value)
    monkeypatch.setattr(gate, "_held_out_probe_score", _boom)
    src, eng, y = _frame()
    before = gate.probe_failure_count()
    with pytest.raises(ValueError, match="cannot fit"):
        gate.measure_feature_uplift(src, eng, y, classification=True)
    assert gate.probe_failure_count() == before


@pytest.mark.parametrize("value", ["0", "off", ""])
def test_strict_mode_is_off_for_the_false_spellings(monkeypatch, value):
    """The usual false spellings keep the fail-open behaviour."""
    monkeypatch.setenv("MLFRAME_FE_PROBE_STRICT", value)
    monkeypatch.setattr(gate, "_held_out_probe_score", _boom)
    src, eng, y = _frame()
    assert gate.measure_feature_uplift(src, eng, y, classification=True) is None


def test_fit_meta_reports_the_probe_failures_of_the_fit():
    """``fit_meta_['fe_probe_failures']`` is the number of fail-open probes inside this fit: zero on a healthy fit, and ``-1`` when the fit start was not marked (an object restored from an old pickle)."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._column_stats import _probe_failures_this_fit

    now = gate.probe_failure_count()
    assert _probe_failures_this_fit(SimpleNamespace(_fe_probe_failures_at_start=now)) == 0
    assert _probe_failures_this_fit(SimpleNamespace(_fe_probe_failures_at_start=now - 3)) == 3
    assert _probe_failures_this_fit(SimpleNamespace()) == -1
