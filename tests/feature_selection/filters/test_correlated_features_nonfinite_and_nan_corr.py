"""SU gives -inf its own bin, and a NaN pairwise correlation does not win the medoid pick."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import correlated_features as cf


def test_su_neg_inf_gets_the_non_finite_bin_not_the_lowest_code():
    """b = -inf where a == 0 else 1 is a bijection of a, so SU is 1; -inf coded as the lowest finite value made b constant (SU 0)."""
    a = np.tile([0.0, 1.0], 100)
    b = np.where(a == 0, -np.inf, 1.0)
    su = cf._su_redundancy_matrix(pd.DataFrame({"a": a, "b": b}))
    assert su[0, 1] > 0.99


def test_medoid_not_picked_by_nan_correlation():
    """a and c never overlap (NaN corr); b links both and must be the medoid, not the first NaN-scored member."""
    n = 100
    x = np.arange(n, dtype=float)
    a = x.copy()
    a[50:] = np.nan
    c = x.copy()
    c[:50] = np.nan
    X = pd.DataFrame({"a": a, "b": x, "c": c})
    cid = cf.cluster_features_by_correlation(X, threshold=0.9, method="pearson")
    assert len(set(cid.tolist())) == 1
    assert cf._cluster_medoids(X, cid, method="pearson") == [1]
