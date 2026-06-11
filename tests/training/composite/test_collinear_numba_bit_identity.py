"""Bit-identity + correctness regression tests for the numba near-collinear dedup.

The numba-JIT pair walk in ``_collinear_numba.near_collinear_keep_mask_fast`` must
return a keep-mask BIT-IDENTICAL to the pure-numpy reference
``_eval_stats._near_collinear_keep_mask_numpy`` for every input class: continuous,
discrete / tied, NaN-holed, exact-duplicate (corr exactly 1.0), and degenerate
(constant) columns. These tests pin that across many seeds and the degenerate
case so a future kernel change cannot silently diverge the selection.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._collinear_numba import (
    _HAS_NUMBA,
    _MIN_COLS,
    _MIN_ROWS,
    near_collinear_keep_mask_fast,
)
from mlframe.training.composite.discovery._eval_stats import (
    _near_collinear_keep_mask_numpy,
    near_collinear_keep_mask,
)


def _fast(fm: np.ndarray, thr: float) -> np.ndarray:
    return near_collinear_keep_mask_fast(
        fm, corr_threshold=thr, reference_fn=_near_collinear_keep_mask_numpy,
    )


def _make_matrix(seed: int) -> tuple[np.ndarray, float]:
    """A mixed matrix big enough to hit the JIT path: some near-duplicate columns
    of a few latent bases, some independent columns, sometimes NaN-holed."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(_MIN_ROWS, 3000))
    n_cols = int(rng.integers(_MIN_COLS, 45))
    latent = rng.normal(size=(n, 4))
    cols = []
    for _ in range(n_cols):
        r = rng.random()
        if r < 0.4:
            noise = rng.choice([1e-5, 1e-3, 1e-2, 0.4])
            cols.append(latent[:, rng.integers(0, 4)] + noise * rng.normal(size=n))
        elif r < 0.5:
            # discrete / tied column (low cardinality) -- the ULP-flip danger zone.
            cols.append(rng.integers(0, 5, size=n).astype(np.float64))
        else:
            cols.append(rng.normal(size=n))
    fm = np.column_stack(cols)
    if rng.random() < 0.4:
        holes = rng.random((n, n_cols)) < 0.08
        fm[holes] = np.nan
    thr = float(rng.choice([0.9, 0.95, 0.99]))
    return fm, thr


@pytest.mark.parametrize("seed", range(40))
def test_fast_mask_bit_identical_to_numpy_reference(seed: int) -> None:
    """JIT keep-mask equals the numpy reference across continuous/discrete/NaN seeds."""
    fm, thr = _make_matrix(seed)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=thr)
    fast = _fast(fm, thr)
    assert np.array_equal(ref, fast), (
        f"seed={seed} kept ref={ref.sum()} fast={fast.sum()}"
    )


def test_exact_duplicate_columns_dropped_identically() -> None:
    """Exact-duplicate columns (corr exactly 1.0) drop identically to the reference."""
    rng = np.random.default_rng(7)
    n = 1000
    base = rng.normal(size=(n, 3))
    # cols 0..2 latent, 3=dup of 0, 4=dup of 1, plus independents to reach JIT size.
    cols = [base[:, 0], base[:, 1], base[:, 2], base[:, 0].copy(), base[:, 1].copy()]
    cols += [rng.normal(size=n) for _ in range(_MIN_COLS)]
    fm = np.column_stack(cols)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=0.99)
    fast = _fast(fm, 0.99)
    assert np.array_equal(ref, fast)
    # The two exact duplicates must be dropped.
    assert not fast[3] and not fast[4]


def test_degenerate_constant_column_kept_identically() -> None:
    """A constant (zero-variance) column never correlates -> always kept; identical."""
    rng = np.random.default_rng(11)
    n = 1500
    a = rng.normal(size=n)
    cols = [a, a + 1e-9 * rng.normal(size=n)]  # near-duplicate pair
    cols.append(np.full(n, 3.14))  # constant column
    cols.append(np.zeros(n))  # another constant
    cols += [rng.normal(size=n) for _ in range(_MIN_COLS)]
    fm = np.column_stack(cols)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=0.99)
    fast = _fast(fm, 0.99)
    assert np.array_equal(ref, fast)
    assert fast[2] and fast[3], "constant columns must be kept"


def test_public_dispatch_matches_reference_on_large_input() -> None:
    """The public ``near_collinear_keep_mask`` (dispatcher) equals the numpy reference."""
    fm, thr = _make_matrix(123)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=thr)
    pub = near_collinear_keep_mask(fm, corr_threshold=thr)
    assert np.array_equal(ref, pub)


def test_small_input_uses_reference_path() -> None:
    """Below the size gate the dispatcher returns the exact reference mask."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=400)
    fm = np.column_stack([a, a + 1e-4 * rng.normal(size=400), rng.normal(size=400)])
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=0.99)
    fast = _fast(fm, 0.99)
    assert np.array_equal(ref, fast)
    assert fast.tolist() == [True, False, True]


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba required for the JIT path")
def test_threshold_disabled_and_degenerate_shapes() -> None:
    """thr>=1.0 disables dedup; 1-column / <3-row matrices keep everything."""
    rng = np.random.default_rng(9)
    a = rng.normal(size=_MIN_ROWS + 100)
    fm = np.column_stack([a] * _MIN_COLS)
    assert _fast(fm, 1.0).all()  # threshold disables -> all kept
    assert _fast(a.reshape(-1, 1), 0.5).all()
    assert _fast(fm[:2], 0.5).all()
