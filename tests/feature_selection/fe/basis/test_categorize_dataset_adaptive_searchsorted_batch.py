"""Wave 11 (Category 3) M8: ``categorize_dataset``'s adaptive-``nbins_strategy`` (MDLP/optimal_joint/...)
path called ``np.searchsorted`` once per column instead of the padded-edge-matrix batched
``_searchsorted_2d_right_njit_parallel`` kernel that already existed. Pins the discretised codes at every
originally-finite position against the pre-fix per-column ``np.searchsorted`` loop (the batching + padding
approach is verified NOT to disturb the separate NaN-bin re-routing, which is applied on top either way).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.discretization import categorize_dataset
import mlframe.feature_selection.filters._adaptive_nbins as _adaptive_nbins_mod


def _old_adaptive_searchsorted(arr, edges_per_col, dtype):
    """Old adaptive searchsorted."""
    n_rows, n_cols = arr.shape
    data = np.empty((n_rows, n_cols), dtype=dtype)
    for j in range(n_cols):
        ej = edges_per_col[j]
        if ej.size == 0:
            data[:, j] = 0
        else:
            data[:, j] = np.searchsorted(ej, arr[:, j].astype(np.float64), side="right").astype(dtype)
    return data


def _make_df(n, p, seed, nan_frac=0.0):
    """Make df."""
    rng = np.random.default_rng(seed)
    cols = {f"c{j}": rng.normal(scale=rng.choice([0.01, 1.0, 100.0]), size=n) + j for j in range(p)}
    df = pd.DataFrame(cols)
    if nan_frac > 0:
        mask = rng.random(df.shape) < nan_frac
        df = df.mask(mask)
    return df


def test_adaptive_categorize_dataset_matches_per_column_searchsorted_at_finite_positions():
    """Adaptive categorize dataset matches per column searchsorted at finite positions."""
    orig_pfe = _adaptive_nbins_mod.per_feature_edges
    captured = {}

    def _capture_pfe(arr, **kwargs):
        """Capture pfe."""
        edges = orig_pfe(arr, **kwargs)
        captured["arr"] = arr.copy()
        captured["edges"] = [e.copy() for e in edges]
        return edges

    n_checks = 0
    for seed in range(4):
        for nan_frac in (0.0, 0.1):
            df = _make_df(n=800, p=15, seed=seed, nan_frac=nan_frac)
            _adaptive_nbins_mod.per_feature_edges = _capture_pfe
            try:
                data_new, _cols_new, _nbins_new = categorize_dataset(
                    df,
                    nbins_strategy="fd",
                    missing_strategy="separate_bin",
                    dtype=np.int32,
                )
            finally:
                _adaptive_nbins_mod.per_feature_edges = orig_pfe
            arr = captured["arr"]
            edges_per_col = captured["edges"]
            data_old = _old_adaptive_searchsorted(arr, edges_per_col, np.int32)

            # Ground-truth NaN mask from the ORIGINAL df (categorize_dataset median-fills NaN in place
            # before per_feature_edges runs, so np.isnan(arr) at capture time would be all-False).
            orig_vals = df.to_numpy(dtype=np.float64)
            finite_pos = ~np.isnan(orig_vals)

            n_checks += 1
            assert np.array_equal(data_old[finite_pos], np.asarray(data_new)[finite_pos]), f"seed={seed} nan_frac={nan_frac}"
    assert n_checks == 8
