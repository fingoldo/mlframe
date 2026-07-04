"""Regression: the noise-floor permutation null in optimise_hermite_pair bins the (fixed) engineered column ONCE and
reuses it across the 50 shuffles via _plugin_mi_from_binned_njit, instead of re-binning it every permutation. This is a
~10x speedup on the null loop (the argsort is ~3/4 of a plug-in MI call) and MUST stay BIT-IDENTICAL: the guard's reject
decision compares the real MI to the null p95, so any drift in the null distribution would flip which engineered pairs
survive. Pins that _plugin_mi_from_binned_njit(_quantile_bin_njit(comb), y) == _plugin_mi_classif_njit(comb, y)."""
import numpy as np

from mlframe.feature_selection.filters.hermite_fe import (
    _plugin_mi_classif_njit, _plugin_mi_from_binned_njit, _quantile_bin_njit,
)


def test_perm_null_bin_once_is_bit_identical_to_rebinning():
    rng = np.random.default_rng(0)
    n, bins = 20000, 20
    mismatches = 0
    max_abs_delta = 0.0
    for _ in range(15):
        # include an outlier-inflated column (the tail-concentrated regime the guard exists for)
        comb = np.ascontiguousarray(rng.standard_normal(n) * (1.0 + 6.0 * (rng.random() < 0.15)), dtype=np.float64)
        yc = comb * 0.4 + rng.standard_normal(n)
        y = np.digitize(yc, np.quantile(yc, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
        comb_binned = _quantile_bin_njit(comb, bins)
        for _p in range(50):
            yp = np.ascontiguousarray(y[rng.permutation(n)])
            old = _plugin_mi_classif_njit(comb, yp, bins)      # re-bins comb (the pre-optimization path)
            new = _plugin_mi_from_binned_njit(comb_binned, yp, bins)  # bin-once path
            d = abs(old - new)
            max_abs_delta = max(max_abs_delta, d)
            if old != new:
                mismatches += 1
    assert mismatches == 0, f"bin-once perm-null diverged from re-binning on {mismatches} shuffles (max|d|={max_abs_delta:.2e})"


def test_optimise_hermite_pair_runs_with_noise_floor():
    """End-to-end smoke: the optimiser (which now uses the bin-once null on the discrete path) still returns a result."""
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair
    rng = np.random.default_rng(1)
    n = 4000
    xa = rng.standard_normal(n)
    xb = rng.standard_normal(n) + 0.3
    yc = xa * xa * xb + rng.standard_normal(n) * 0.4
    y = np.digitize(yc, np.quantile(yc, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
    r = optimise_hermite_pair(xa, xb, y, n_trials=60, min_degree=3, max_degree=5,
                              optimizer="cma_batch", discrete_target=True, noise_floor_n_perms=30)
    assert r is None or r.mi >= 0.0  # either rejected by the null, or a valid non-negative MI
