"""The additive fusion rejects a compound that loses one half's linear signal and accepts one that keeps both."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_additive_fusion import _evaluate_fusion_pair, _evaluate_fusion_pair_weighted, _half_scale


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


def test_a_half_on_a_much_smaller_scale_is_fused_on_the_normalised_sum():
    """The plain sum is the large half alone and loses the small half's term, so it is refused; the fusion then takes the sum of the
    halves divided by their scales, which keeps both terms, and records those scales with the compound."""
    me, ha, hb, y_dense = _halves(scale_b=1e-3)
    assert _evaluate_fusion_pair_weighted(me, ha, hb, y_dense, 10, len(y_dense), 1.0, (1.0, 1.0)) is None
    got = _evaluate_fusion_pair(me, ha, hb, y_dense, 10, len(y_dense), 1.0)
    assert got is not None
    binary, values, _binned, _mi, weights = got
    assert weights == (_half_scale(ha), _half_scale(hb)) and weights[1] < 1e-2
    expected = ha["vals"] / weights[0] + hb["vals"] / weights[1] if binary == "add" else ha["vals"] / weights[0] - hb["vals"] / weights[1]
    assert np.allclose(values, expected)


def test_gpu_twin_agrees_with_the_host_on_both_cases():
    """The device gate takes the same decision as the host one (it holds the same condition)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._fe_additive_fusion_gpu_resident import _gpu_evaluate_pair
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin  # noqa: F401  (import check for the binner the twin uses)

    for scale_b in (1.0, 1e-3):
        me, ha, hb, y_dense = _halves(scale_b=scale_b)
        vals = cp.asarray(np.column_stack([ha["vals"], hb["vals"]]))
        ha_g, hb_g = {**ha, "col": 0, "scale": _half_scale(ha)}, {**hb, "col": 1, "scale": _half_scale(hb)}
        yc = cp.asarray(y_dense.astype(np.float64))
        qs = cp.linspace(0.0, 1.0, 11)[1:-1]
        got = _gpu_evaluate_pair(me, cp, ha_g, hb_g, vals, y_dense, qs, yc, float(np.std(y_dense)), len(y_dense), 1.0)
        assert got is not None, f"scale_b={scale_b}: device gate refused a fusion the host admits"
