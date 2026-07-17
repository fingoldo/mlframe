"""Parity tests for the column-major brute-force kernels.

``_topn_fixed_r_colmajor`` and ``_topn_fixed_r_parallel_colmajor`` operate on a ``(f, n)`` phi
(transpose of the row-major ``(n, f)`` form) so each feature column is a contiguous row, turning
the inner ``phi[t, fcol]`` strided read into a unit-stride read. These tests pin bit-identical
output against the row-major kernels across every metric and a sweep of cardinalities so the
optimisation never silently changes the ranking.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("metric_code", [0, 1, 2, 3])
@pytest.mark.parametrize("r", [1, 2, 3, 4, 5])
def test_colmajor_matches_rowmajor_per_cardinality(metric_code, r):
    """Colmajor matches rowmajor per cardinality."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import (
        _topn_fixed_r,
        _topn_fixed_r_colmajor,
        generate_combinations,
    )

    rng = np.random.default_rng(metric_code * 13 + r)
    n, f = 250, 8
    phi = np.ascontiguousarray(rng.standard_normal((n, f)))
    phi_T = np.ascontiguousarray(phi.T)
    base = rng.standard_normal(n)
    y = (rng.uniform(size=n) > 0.5).astype(np.float64)

    seq = np.arange(f, dtype=np.int32)
    combos = generate_combinations(seq, r)
    tc_row, tl_row = _topn_fixed_r(phi, base, y, combos, metric_code, 10)
    tc_col, tl_col = _topn_fixed_r_colmajor(phi_T, base, y, combos, metric_code, 10)

    # Same arithmetic in the same row-order; under fastmath, numba is free to reorder the FMA in
    # ``margin[t] = base[t] + main_sum[t] + phi[..., t]`` between layouts, so per-element loss
    # results can differ by ~1 ULP (~2e-16). Top-N membership (as a set of feature tuples) must
    # be identical and per-loss values must agree to 1e-12 relative.
    row_set = {tuple(c.tolist()) for c in tc_row}
    col_set = {tuple(c.tolist()) for c in tc_col}
    assert row_set == col_set, f"top-N sets differ: row={row_set} col={col_set}"
    np.testing.assert_allclose(np.sort(tl_row), np.sort(tl_col), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("metric_code", [0, 1, 2, 3])
def test_colmajor_parallel_matches_rowmajor(metric_code):
    """Colmajor parallel matches rowmajor."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import (
        _topn_fixed_r_parallel,
        _topn_fixed_r_parallel_colmajor,
        generate_combinations,
    )

    rng = np.random.default_rng(metric_code + 100)
    n, f = 200, 9
    phi = np.ascontiguousarray(rng.standard_normal((n, f)))
    phi_T = np.ascontiguousarray(phi.T)
    base = rng.standard_normal(n)
    y = (rng.uniform(size=n) > 0.5).astype(np.float64)

    seq = np.arange(f, dtype=np.int32)
    combos = generate_combinations(seq, 4)
    n_chunks = 4
    cc_row, cl_row = _topn_fixed_r_parallel(phi, base, y, combos, metric_code, 8, n_chunks)
    cc_col, cl_col = _topn_fixed_r_parallel_colmajor(phi_T, base, y, combos, metric_code, 8, n_chunks)

    # prange chunk boundaries are deterministic. Per-chunk top-N sets must match exactly; per-loss
    # values may diverge by ~1 ULP under fastmath FMA reorder between layouts (sorted compare).
    for ch in range(cc_row.shape[0]):
        row_set = {tuple(c.tolist()) for c in cc_row[ch] if c[0] != -1}
        col_set = {tuple(c.tolist()) for c in cc_col[ch] if c[0] != -1}
        assert row_set == col_set, f"chunk {ch}: row={row_set} col={col_set}"
    np.testing.assert_allclose(np.sort(cl_row.ravel()), np.sort(cl_col.ravel()), rtol=1e-12, atol=1e-15)


def test_brute_force_top_n_unchanged_after_colmajor_routing():
    """End-to-end: the public brute_force_top_n result is stable after the dispatcher transpose."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    rng = np.random.default_rng(7)
    phi = rng.standard_normal((300, 9))
    base = rng.standard_normal(300) * 0.1
    y = base + phi[:, [1, 4, 7]].sum(axis=1) + 0.02 * rng.standard_normal(300)

    res_rmse = brute_force_top_n(phi, base, y, classification=False, metric="rmse", max_card=5, top_n=15)
    res_mae = brute_force_top_n(phi, base, y, classification=False, metric="mae", max_card=5, top_n=15)
    # winner subset must be the planted relevant features
    assert set(res_rmse[0][1]) == {1, 4, 7}
    assert set(res_mae[0][1]) == {1, 4, 7}


def test_score_margin_inline_sigmoid_parity():
    """``_sigmoid`` got ``inline='always'``; verify per-metric score_margin is unchanged."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import score_margin

    rng = np.random.default_rng(42)
    margin = rng.standard_normal(500) * 3.0
    y = (rng.uniform(size=500) > 0.5).astype(np.float64)

    # Independent reference (numpy, no numba): exact same arithmetic in the same order
    def ref(code):
        """Helper that ref."""
        if code == 0:
            return float(np.mean(np.abs(y - margin)))
        if code == 1:
            return float(np.mean((y - margin) ** 2))
        p = np.where(margin >= 0, 1.0 / (1.0 + np.exp(-margin)), np.exp(margin) / (1.0 + np.exp(margin)))
        if code == 2:
            return float(np.mean((p - y) ** 2))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))

    for code in (0, 1, 2, 3):
        got = float(score_margin(margin, y, code))
        exp = ref(code)
        # fastmath rounding tolerance: rel diff well under 1e-12 in practice
        assert abs(got - exp) <= 1e-12 * max(abs(exp), 1.0), f"code={code} got={got} exp={exp}"
