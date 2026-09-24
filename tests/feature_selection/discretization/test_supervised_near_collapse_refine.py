"""A supervised (MDLP) binning that leaves one bin holding almost every row must be refined with the
unsupervised fallback edges, like a fully collapsed column already is; otherwise a zero-marginal synergy
operand loses its joint signal and the pair screen drops a genuine interaction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._supervised_collapse_refine import refine_near_collapsed_supervised_edges


def test_sliver_split_gets_fallback_edges_and_keeps_the_supervised_cut():
    """A sliver supervised split gets fallback edges while keeping the supervised cut, and no bin holds over 30% of rows."""
    x = np.random.default_rng(0).standard_normal(2000)
    cut = np.sort(x)[9] + 1e-9  # 10 rows on the left
    out = refine_near_collapsed_supervised_edges(np.array([cut]), x, "quantile", 5)
    assert cut in out
    assert out.size >= 4
    counts = np.bincount(np.searchsorted(out, x, side="right"))
    assert counts.max() <= 0.3 * x.size


def test_balanced_supervised_split_is_untouched():
    """A balanced supervised split is returned unchanged (same object)."""
    x = np.random.default_rng(1).standard_normal(2000)
    edges = np.array([0.0])
    assert refine_near_collapsed_supervised_edges(edges, x, "quantile", 5) is edges


def test_mdlp_categorize_keeps_synergy_operand_resolution():
    """End to end on the wide-synergy fixture: MDLP gives the pure-synergy operand ``x3`` a spurious 10-vs-1990 tail
    cut. After the refinement the discretised (x3, x4) pair carries its sign-product joint MI again."""
    from sklearn.metrics import mutual_info_score

    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_order2_maxt_floor import _wide_synergy_frame

    # n pinned: the borrowed fixture shrinks to 1600 rows under MLFRAME_FAST to keep the MRMR test it was written for
    # cheap, and at 1600 the pair's joint MI lands at ~0.043, under a threshold calibrated at 2000. This check only
    # discretises two columns, so the full size costs nothing. Measured at 2000: 0.054 with the refinement, 0.004 without.
    X, y = _wide_synergy_frame(n=2000, n_noise=4)
    df = X[["x3", "x4"]].copy()
    df["y"] = y.to_numpy()
    data, cols, nbins = categorize_dataset(df=df, method="quantile", n_bins=10, dtype=np.int16, nbins_strategy="mdlp", y_for_strategy=y.to_numpy())
    i3, i4 = cols.index("x3"), cols.index("x4")
    assert int(nbins[i3]) >= 3, f"x3 still near-collapsed: nbins={int(nbins[i3])}"
    joint = pd.Series(data[:, i3].astype(np.int64) * 1000 + data[:, i4].astype(np.int64))
    assert mutual_info_score(joint, y.to_numpy()) > 0.05
