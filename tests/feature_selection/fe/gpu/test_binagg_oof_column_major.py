"""The binned-aggregate OOF builder's column-major output is the transpose of its row-major output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as res


def _specs(df, n_bins=6):
    """Column specs for two (group, agg) pairs and all four stats."""
    specs = []
    for g, a in (("g1", "a1"), ("g2", "a2")):
        edges = np.quantile(df[g], np.linspace(0, 1, n_bins + 1)[1:-1])
        for stat in ("mean", "std", "skew", "kurt"):
            specs.append({"name": f"{a}_{stat}_by_{g}", "group_col": g, "agg_col": a, "stat": stat, "edges": edges, "global": float(df[a].mean())})
    return specs


def test_column_major_is_the_transpose():
    """Same values (to summation-order rounding), shape (K, n), C-contiguous."""
    rng = np.random.default_rng(0)
    n = 12_000
    df = pd.DataFrame({"g1": rng.normal(size=n), "g2": rng.uniform(size=n), "a1": rng.normal(size=n), "a2": rng.exponential(size=n)})
    df.loc[::37, "a1"] = np.nan
    specs = _specs(df)
    folds = res.binagg_fold_ids(n, 5, 0)
    row = res.build_binagg_oof_matrix_gpu(cp, df, specs, folds, 5)
    col = res.build_binagg_oof_matrix_gpu(cp, df, specs, folds, 5, column_major=True)
    assert row.shape == (n, len(specs)) and col.shape == (len(specs), n) and col.flags.c_contiguous
    # the per-cell moments are summed with atomics in an unspecified order, so two builds agree to summation-order rounding, with the same NaN pattern
    np.testing.assert_allclose(cp.asnumpy(col), cp.asnumpy(row).T, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_empty_spec_list_has_matching_empty_shapes():
    """No columns: (n, 0) and (0, n)."""
    df = pd.DataFrame({"x": np.arange(10.0)})
    folds = res.binagg_fold_ids(10, 2, 0)
    assert res.build_binagg_oof_matrix_gpu(cp, df, [], folds, 2).shape == (10, 0)
    assert res.build_binagg_oof_matrix_gpu(cp, df, [], folds, 2, column_major=True).shape == (0, 10)
