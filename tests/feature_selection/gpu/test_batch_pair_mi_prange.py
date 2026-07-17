"""Numerical-equivalence test for ``batch_pair_mi_prange`` vs the legacy
``merge_vars + compute_mi_from_classes`` per-pair path that ``MRMR._run_fe_step``
invoked through joblib.

The new njit-prange kernel processes all (a, b) pairs in one shot with shared-
memory parallelism; the legacy path computed each pair in a separate joblib
worker. Both paths use the same joint-class encoding (``a * nbins[b] + b``) and
the same ``compute_mi_from_classes`` arithmetic, so MIs must match to fp tolerance.

Test design:
  * Synthesise pre-binned (int) factor data of various shapes (different nbins
    per column, classes_y cardinalities).
  * Enumerate all unique (a, b) pairs.
  * Compute pair MIs via:
      (a) ``batch_pair_mi_prange``: one batch call.
      (b) ``merge_vars`` + ``compute_mi_from_classes`` per pair: the legacy loop.
  * Assert MIs equal to fp tolerance.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest


def _legacy_pair_mi(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y, dtype):
    """Per-pair reference implementation matching the legacy path."""
    from mlframe.feature_selection.filters.info_theory import merge_vars, compute_mi_from_classes

    n_pairs = pair_a.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    for p in range(n_pairs):
        vars_indices = np.array([pair_a[p], pair_b[p]], dtype=np.int64)
        classes_x, freqs_x_norm, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=vars_indices,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        out[p] = compute_mi_from_classes(
            classes_x=classes_x,
            freqs_x=freqs_x_norm,
            classes_y=classes_y,
            freqs_y=freqs_y,
            dtype=dtype,
        )
    return out


def _build_factor_data(n_samples: int, nbins_per_col: list[int], seed: int):
    """Returns (factors_data, nbins_array)."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nb, size=n_samples) for nb in nbins_per_col]
    data = np.column_stack(cols).astype(np.int32)
    return data, np.asarray(nbins_per_col, dtype=np.int32)


@pytest.mark.parametrize(
    "n_samples,nbins_per_col,n_classes_y,seed",
    [
        (500, [4, 4, 4, 4], 2, 1),  # uniform small
        (500, [3, 5, 7, 4], 3, 2),  # heterogeneous bins, 3-class y
        (2000, [5, 5, 5, 5, 5, 5], 4, 3),  # 6 features -> 15 pairs, 4-class y
        (1000, [2, 3, 4, 5], 2, 4),  # binary-cardinality cats included
    ],
)
def test_batch_pair_mi_prange_matches_legacy(n_samples, nbins_per_col, n_classes_y, seed):
    from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange

    data, nbins = _build_factor_data(n_samples, nbins_per_col, seed)
    rng = np.random.default_rng(seed + 100)
    # Build classes_y + marginal freq vector. classes_y must use 0..n_classes_y-1.
    y_raw = rng.integers(0, n_classes_y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(y_raw, minlength=n_classes_y).astype(np.int32) / n_samples

    # Enumerate ALL pairs (a < b).
    n_cols = len(nbins_per_col)
    pairs = list(itertools.combinations(range(n_cols), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)

    mi_batch = batch_pair_mi_prange(
        factors_data=data,
        pair_a=pair_a,
        pair_b=pair_b,
        nbins=nbins,
        classes_y=y_raw,
        freqs_y=freqs_y,
    )
    mi_legacy = _legacy_pair_mi(
        factors_data=data,
        pair_a=pair_a,
        pair_b=pair_b,
        nbins=nbins,
        classes_y=y_raw,
        freqs_y=freqs_y,
        dtype=np.int32,
    )

    np.testing.assert_allclose(
        mi_batch,
        mi_legacy,
        atol=1e-9,
        rtol=1e-9,
        err_msg=(
            f"batch_pair_mi_prange MIs diverged from legacy merge_vars+compute_mi path: "
            f"shapes=(n={n_samples}, nbins={nbins_per_col}, n_classes_y={n_classes_y}). "
            f"batch[:5]={mi_batch[:5]}, legacy[:5]={mi_legacy[:5]}"
        ),
    )


def test_batch_pair_mi_prange_handles_singleton_pair_set():
    """One-pair input must still return a (1,) ndarray, not a scalar."""
    from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange

    data, nbins = _build_factor_data(200, [4, 4], seed=7)
    y = np.zeros(200, dtype=np.int32)
    y[100:] = 1
    freqs_y = np.array([0.5, 0.5], dtype=np.float64)

    mi = batch_pair_mi_prange(
        factors_data=data,
        pair_a=np.array([0], dtype=np.int64),
        pair_b=np.array([1], dtype=np.int64),
        nbins=nbins,
        classes_y=y,
        freqs_y=freqs_y,
    )
    assert mi.shape == (1,)
    assert mi[0] >= 0.0  # MI is non-negative
