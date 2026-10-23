"""The additive fusion rejects a compound that loses one half's linear signal and accepts one that keeps both."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_additive_fusion import _evaluate_fusion_pair


def _halves(scale_b: float, n: int = 6000, seed: int = 0):
    """Two informative halves ``A`` and ``B * scale_b`` of a target ``A + B``, the shared self, and the target codes."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = a + b + 0.05 * rng.standard_normal(n)
    y_dense = np.searchsorted(np.quantile(y, np.linspace(0, 1, 11)[1:-1]), y, side="right").astype(np.int64)
    ha = {"name": "ha", "vals": a, "mi": 0.0, "floor": 0.0}
    hb = {"name": "hb", "vals": b * scale_b, "mi": 0.0, "floor": 0.0}
    return SimpleNamespace(fe_additive_fusion_ols_r_margin_sd=2.0, _fe_rejection_records_=[]), ha, hb, y_dense


def test_halves_on_similar_scales_are_fused():
    """With both halves on one scale the unweighted sum keeps both terms, so the fusion is admitted."""
    me, ha, hb, y_dense = _halves(scale_b=1.0)
    assert _evaluate_fusion_pair(me, ha, hb, y_dense, 10, len(y_dense), 1.0) is not None


def test_a_half_on_a_much_smaller_scale_is_not_fused_and_the_rejection_is_recorded():
    """The sum is then the large half alone: the compound loses the second term the scale-free tests credited, so the fragments stay."""
    me, ha, hb, y_dense = _halves(scale_b=1e-3)
    assert _evaluate_fusion_pair(me, ha, hb, y_dense, 10, len(y_dense), 1.0) is None
    rows = [r for r in me._fe_rejection_records_ if r["gate"] == "additive_fusion"]
    assert len(rows) == 1 and "scale imbalance" in rows[0]["reason"]


def test_gpu_twin_agrees_with_the_host_on_both_cases():
    """The device gate takes the same decision as the host one (it holds the same condition)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._fe_additive_fusion_gpu_resident import _gpu_evaluate_pair
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin  # noqa: F401  (import check for the binner the twin uses)

    for scale_b, expect_fused in ((1.0, True), (1e-3, False)):
        me, ha, hb, y_dense = _halves(scale_b=scale_b)
        vals = cp.asarray(np.column_stack([ha["vals"], hb["vals"]]))
        ha_g, hb_g = {**ha, "col": 0}, {**hb, "col": 1}
        yc = cp.asarray(y_dense.astype(np.float64))
        qs = cp.linspace(0.0, 1.0, 11)[1:-1]
        got = _gpu_evaluate_pair(me, cp, ha_g, hb_g, vals, y_dense, qs, yc, float(np.std(y_dense)), len(y_dense), 1.0)
        assert (got is not None) == expect_fused, f"scale_b={scale_b}: device gate returned {got is not None}"
