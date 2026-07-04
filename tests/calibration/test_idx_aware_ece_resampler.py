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


def test_normalize_binary_labels_fast_path_equals_general_path():
    """_normalize_binary_labels short-circuits already-{0,1} integer/bool labels (min==0/max==1 => exactly the two
    0/1 values, both present) to skip the np.unique sort that ran on every bootstrap-ECE resample. Regression sensor:
    the fast path must equal the np.unique general path on 0/1 inputs, still remap non-0/1 encodings ({1,2}, {-1,+1})
    via the general path, and preserve the <2-distinct-values raise."""
    import numpy as np
    from mlframe.calibration.policy import _normalize_binary_labels

    rng = np.random.default_rng(0)
    for _ in range(30):
        y01 = rng.integers(0, 2, int(rng.integers(10, 4000)))
        if np.unique(y01).size < 2:
            y01[0], y01[-1] = 0, 1
        out = _normalize_binary_labels(y01)
        assert out.dtype == np.int64 and np.array_equal(out, y01.astype(np.int64))

    # bool 0/1 also hits the fast path
    yb = np.array([True, False, True, False])
    assert np.array_equal(_normalize_binary_labels(yb), np.array([1, 0, 1, 0]))

    # Non-0/1 encodings still remap (larger -> 1) via the general path.
    assert np.array_equal(_normalize_binary_labels(np.array([1, 2, 2, 1])), np.array([0, 1, 1, 0]))
    assert np.array_equal(_normalize_binary_labels(np.array([-1, 1, 1, -1])), np.array([0, 1, 1, 0]))

    # <2 distinct finite values must still raise.
    import pytest
    with pytest.raises(ValueError):
        _normalize_binary_labels(np.array([1, 1, 1]))


def test_normalize_binary_labels_f64_fast_path():
    """float64 labels (the bootstrap-ECE hot path: honest_diagnostics casts to float64 ONCE, so every resample
    reaches _normalize_binary_labels as float64) short-circuit via the njit _scan_binary01_f64 scan instead of the
    np.unique O(n log n) sort. Sensor: the fast path must (a) return an all-{0,1} float64 vector unchanged, yielding
    ECE bit-identical to the int-cast general path; (b) NOT falsely fast-path non-0/1 floats (0.5 present, single
    class); (c) still remap {1.0,2.0}/{-1.,1.} and raise on <2 distinct finite values."""
    import numpy as np
    from mlframe.calibration.policy import _normalize_binary_labels, _scan_binary01_f64, _ece_score

    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(50, 5000))
        y = rng.integers(0, 2, n).astype(np.float64)
        y[0], y[-1] = 0.0, 1.0  # guarantee both classes
        p = rng.random(n)
        assert _scan_binary01_f64(y) == 1
        out = _normalize_binary_labels(y)
        # value-equal to the int general path, and ECE bit-identical either way
        assert np.array_equal(out.astype(np.int64), y.astype(np.int64))
        e_int = _ece_score(y.astype(np.int64).astype(np.float64), p, n_bins=10)
        e_fast = _ece_score(y, p, n_bins=10)
        assert e_int == e_fast

    # must NOT fast-path a float64 with a non-0/1 value present
    assert _scan_binary01_f64(np.array([0.0, 0.5, 1.0])) == 0
    # single class present -> not fast-pathed (0), general path raises
    assert _scan_binary01_f64(np.array([0.0, 0.0, 0.0])) == 0
    # NaN ignored, both classes still present -> fast-pathed
    assert _scan_binary01_f64(np.array([0.0, np.nan, 1.0])) == 1
    # +inf is not finite-0/1 -> not fast-pathed; general isfinite path handles it
    assert _scan_binary01_f64(np.array([0.0, 1.0, np.inf])) == 0

    # non-0/1 float encodings still remap (larger -> 1) via the general path
    assert np.array_equal(_normalize_binary_labels(np.array([1.0, 2.0, 2.0, 1.0])), np.array([0, 1, 1, 0]))
    assert np.array_equal(_normalize_binary_labels(np.array([-1.0, 1.0, 1.0, -1.0])), np.array([0, 1, 1, 0]))

    import pytest
    with pytest.raises(ValueError):
        _normalize_binary_labels(np.array([1.0, 1.0, 1.0]))
