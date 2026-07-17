"""biz_value tests for GroupAwareMRMR decision params.

Each test pins a quantitative win of a specific constructor param against its baseline / OFF value on a synthetic where the
param should clearly succeed: corr_method='su' (catches non-monotone redundancy pearson misses), corr_threshold (collapses a
collinear cluster), expand (recovers cluster members), min_reduction (gates the medoid bypass).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from mlframe.feature_selection.filters.group_aware import (
    GroupAwareMRMR,
    _redundancy_matrix,
    cluster_features_by_correlation,
)

pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


class _TopKByCorr(BaseEstimator):
    """Inner selector keeping the k columns most abs-correlated with y; exposes support_ like an mRMR selector."""

    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X, y):
        """Helper that fit."""
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        yv = np.asarray(y)
        cors = np.array([abs(np.corrcoef(Xv[:, j], yv)[0, 1]) if np.std(Xv[:, j]) > 0 else 0.0 for j in range(Xv.shape[1])])
        self.support_ = np.argsort(-np.nan_to_num(cors))[: self.k]
        return self


def _collinear_signal_frame(seed: int = 1, n: int = 1000):
    """4 near-duplicate copies of a relevant signal g (tight collinear cluster) + 3 independent noise columns; y = 1[g>0]."""
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, n)
    cols = {f"g{i}": g + rng.normal(0, 0.05, n) for i in range(4)}
    cols.update({f"n{i}": rng.normal(0, 1, n) for i in range(3)})
    X = pd.DataFrame(cols)
    y = (g > 0).astype(int)
    return X, y


def test_biz_val_group_aware_corr_method_su_catches_nonmonotone_redundancy():
    """corr_method='su' merges a non-monotone redundant pair (b=(a-0.5)^2) that pearson scores ~0 and leaves unclustered."""
    rng = np.random.default_rng(0)
    n = 800
    a = rng.uniform(0, 1, n)
    b = (a - 0.5) ** 2 + rng.normal(0, 0.001, n)
    c = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "b": b, "c": c})

    su_ab = _redundancy_matrix(X, "su")[0, 1]
    pe_ab = _redundancy_matrix(X, "pearson")[0, 1]
    assert su_ab >= 0.45, f"SU should expose the non-monotone redundancy; got {su_ab:.3f}"
    assert pe_ab <= 0.20, f"pearson should miss it; got {pe_ab:.3f}"

    cl_pe = cluster_features_by_correlation(X, threshold=0.5, method="pearson")
    cl_su = cluster_features_by_correlation(X, threshold=0.5, method="su")
    assert len(np.unique(cl_pe)) == 3, "pearson leaves a,b,c as 3 singletons"
    assert cl_su[0] == cl_su[1], "SU merges a and b into one cluster"
    assert len(np.unique(cl_su)) == 2, "SU yields 2 clusters (ab, c)"


def test_biz_val_group_aware_corr_threshold_collapses_collinear_cluster():
    """A tight threshold collapses the 4-member collinear cluster (reduction>0); a near-1 threshold leaves it intact (no reduction)."""
    X, y = _collinear_signal_frame()
    tight = GroupAwareMRMR(_TopKByCorr(k=1), corr_threshold=0.9, corr_method="pearson", expand=True, min_reduction=0.05).fit(X, y)
    loose = GroupAwareMRMR(_TopKByCorr(k=1), corr_threshold=0.999, corr_method="pearson", expand=True, min_reduction=0.05).fit(X, y)

    assert tight.reduction_ >= 0.40, f"tight threshold should collapse the collinear cluster; reduction={tight.reduction_:.3f}"
    assert tight.reduced_ is True
    assert loose.reduction_ <= 0.01, f"near-1 threshold should merge nothing; reduction={loose.reduction_:.3f}"
    assert loose.reduced_ is False
    assert tight.reduction_ - loose.reduction_ >= 0.35


def test_biz_val_group_aware_expand_recovers_cluster_members():
    """expand=True returns ALL members of a selected cluster (4 g-copies); expand=False returns only the medoid (1)."""
    X, y = _collinear_signal_frame()
    g_idx = {i for i, c in enumerate(X.columns) if c.startswith("g")}

    exp = GroupAwareMRMR(_TopKByCorr(k=1), corr_threshold=0.9, expand=True, min_reduction=0.05).fit(X, y)
    noexp = GroupAwareMRMR(_TopKByCorr(k=1), corr_threshold=0.9, expand=False, min_reduction=0.05).fit(X, y)

    exp_sup = set(exp.support_.tolist())
    noexp_sup = set(noexp.support_.tolist())
    assert g_idx.issubset(exp_sup), f"expand=True must recover all 4 collinear members; got {sorted(exp_sup)}"
    assert len(exp_sup) - len(noexp_sup) >= 3, "expand must add >=3 cluster members vs medoid-only"
    assert len(noexp_sup) == 1, "expand=False keeps only the selected cluster medoid"


def test_biz_val_group_aware_min_reduction_gates_medoid_bypass():
    """min_reduction is the bypass gate: below it the wrapper fits inner on FULL X (reduced_=False); above it uses medoids (reduced_=True)."""
    X, y = _collinear_signal_frame()
    # Same data + threshold; only min_reduction flips. Achieved reduction is ~0.43.
    high_gate = GroupAwareMRMR(_TopKByCorr(k=1), corr_threshold=0.9, min_reduction=0.9).fit(X, y)
    low_gate = GroupAwareMRMR(_TopKByCorr(k=1), corr_threshold=0.9, min_reduction=0.05).fit(X, y)

    assert high_gate.reduced_ is False, "min_reduction above achieved reduction must bypass the medoid path"
    assert low_gate.reduced_ is True, "min_reduction below achieved reduction must engage the medoid path"
    # When engaged + expand, the full collinear group is recovered; when bypassed, inner fits raw X (single best column).
    assert len(low_gate.support_) >= 4
    assert len(high_gate.support_) == 1
