"""Regression: fused idx-aware bootstrap-ECE resampler (idx-aware-ece lead).

``_bootstrap_ece_with_indices`` now fuses the per-resample ``y[idx]`` / ``p[idx]``
gather INTO the njit ECE kernel (``_ece_score_idx_numba_serial``) instead of
materialising a Python-level fancy-index slice per resample. Equal-width ECE
binning is order-independent (no argsort / tie-break), so gathering inside the
loop is BIT-IDENTICAL to slicing outside. These tests pin:

  * the fused kernel == the slice-based kernel on every resample (incl. tied /
    discrete / NaN-laden probabilities and unsorted idx);
  * the n_bins-fused ``_bootstrap_ece_with_indices`` path produces point/lo/hi
    BIT-IDENTICAL to the legacy metric_fn-slice path AND to ``bootstrap_metric``;
  * the lead-ece-wrapper contiguous-float64 fast path == the coercion path.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.fast]


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("kind", ["continuous", "discrete", "with_nan"])
def test_fused_kernel_bit_identical_to_slice(seed, kind):
    from mlframe.calibration.policy import _ece_score, _ece_score_idx_numba_serial

    rng = np.random.default_rng(seed)
    n = 1500
    if kind == "continuous":
        p = rng.uniform(0, 1, n)
    elif kind == "discrete":
        p = rng.integers(0, 5, n).astype(np.float64) / 4.0  # heavy ties
    else:
        p = rng.uniform(0, 1, n)
        p[rng.integers(0, n, 30)] = np.nan
    p = np.ascontiguousarray(p)
    y = np.ascontiguousarray((rng.uniform(0, 1, n) < 0.4).astype(np.int64))

    for _ in range(20):
        idx = rng.integers(0, n, n).astype(np.int64)  # unsorted resample
        fused = _ece_score_idx_numba_serial(y, p, idx, 15)
        sliced = _ece_score(y[idx], p[idx], n_bins=15)
        if np.isnan(sliced):
            assert np.isnan(fused)
        else:
            assert fused == sliced


@pytest.mark.parametrize("stratified", [True, False])
def test_fused_bootstrap_bit_identical_to_metric_fn_and_bootstrap_metric(stratified):
    from mlframe.calibration import policy
    from mlframe.evaluation.bootstrap import bootstrap_metric

    rng = np.random.default_rng(3)
    n = 2500
    y = (rng.uniform(0, 1, n) < 0.3).astype(np.int64)
    p = np.clip(rng.uniform(0, 1, n) + 0.1 * y, 0.0, 1.0)
    strat = y if stratified else None
    mf = lambda a, b: policy._ece_score(a, b, n_bins=15)

    idx = policy._build_resample_indices(n, 400, strat, 7)
    fused = policy._bootstrap_ece_with_indices(y, p, idx, mf, 0.05, n_bins=15)
    sliced = policy._bootstrap_ece_with_indices(y, p, idx, mf, 0.05, n_bins=None)
    ref = bootstrap_metric(y, p, metric_fn=mf, n_bootstrap=400, alpha=0.05, stratify=strat, random_state=7)

    assert fused["point"] == sliced["point"] == ref["point"]
    assert fused["lo"] == sliced["lo"] == ref["lo"]
    assert fused["hi"] == sliced["hi"] == ref["hi"]


def test_wrapper_fast_path_matches_coercion_path():
    """lead-ece-wrapper: contiguous-float64 inputs skip coercion, same result."""
    from mlframe.calibration.policy import _ece_score

    rng = np.random.default_rng(5)
    n = 2000
    p = np.ascontiguousarray(rng.uniform(0, 1, n))  # float64 contiguous -> fast path
    y = np.ascontiguousarray((rng.uniform(0, 1, n) < 0.4).astype(np.int64))

    fast = _ece_score(y, p, n_bins=15)
    # force the coercion path via a non-contiguous / 2-col view
    p2 = np.column_stack([1.0 - p, p])  # 2-D -> coercion path picks col 1
    slow = _ece_score(y.astype(np.float64), p2, n_bins=15)
    assert fast == slow
