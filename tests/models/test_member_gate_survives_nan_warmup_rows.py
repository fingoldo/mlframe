"""NaN warm-up rows (time-ordered OOF) must not switch the member outlier gate off."""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling import compute_member_quality_gate
from mlframe.models.ensembling.predict import _compute_outlier_gate


def _members(n=400):
    """Four members around a signal, the last one a gross outlier; the first 50 rows NaN in every member."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    preds = [signal + rng.normal(scale=0.05, size=n) for _ in range(3)] + [signal + 5.0 + rng.normal(size=n)]
    for p in preds:
        p[:50] = np.nan
    return preds


def test_quality_gate_still_drops_the_outlier_member_with_nan_warmup_rows():
    """The outlier (index 3) is excluded and the gate is not reported as too restrictive."""
    kept, excluded, stats = compute_member_quality_gate(_members())
    assert 3 not in kept
    assert {i for i, _ in excluded} == {3}
    assert not stats.get("filter_too_restrictive", False)


def test_predict_gate_drops_the_outlier_when_one_member_has_a_nan_cell():
    """One NaN cell in member 0 used to make every statistic NaN and keep the 5-sigma outlier."""
    preds = _members()
    for p in preds:
        p[:50] = p[50:100]
    preds[0][60] = np.nan
    skipped, *_ = _compute_outlier_gate(preds, np.stack(preds), 0.0, 0.0, 2.5, 2.5)
    assert 3 in skipped
