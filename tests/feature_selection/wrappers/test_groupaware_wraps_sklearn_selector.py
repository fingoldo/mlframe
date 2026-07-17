"""Regression guard: GroupAwareMRMR can wrap a sklearn-style wrapper selector
that exposes a BOOLEAN support_ mask (audit integration-defaults-3).

GroupAwareMRMR collapses correlated clusters to medoids, fits the inner selector
on the medoids, then expands the support back to whole clusters. It previously
iterated ``inner.support_`` as if it were an index array (mRMR-family); sklearn
RFECV exposes a boolean mask, which iterated as 0/1 -> wrong clusters. The fix
normalises both conventions, enabling the measured ~3x wall-clock win of running
the wrapper on medoids instead of every redundant column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR


def test_wraps_boolean_mask_selector_and_expands_cluster():
    """GroupAwareMRMR normalises a wrapped sklearn selector's boolean support_ mask (vs mRMR-family index arrays) and expands the medoid pick back to its whole correlated cluster."""
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 800
    z = rng.standard_normal(n)
    cols = {f"sig{i}": z + 0.2 * rng.standard_normal(n) for i in range(4)}  # one corr cluster
    for i in range(6):
        cols[f"noise{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = pd.Series((z > 0).astype(int))

    g = GroupAwareMRMR(
        RFECV(LogisticRegression(max_iter=500), cv=3, min_features_to_select=1),
        corr_threshold=0.7,
        corr_method="pearson",
    ).fit(X, y)

    names = list(X.columns)
    support_names = {names[i] for i in g.support_}
    # The fix must (a) not crash on the boolean mask and (b) expand the selected
    # medoid back to its whole cluster (all 4 strongly-predictive sig features).
    assert len(g.support_) >= 1
    assert all(f"sig{i}" in support_names for i in range(4)), f"signal cluster must be selected and expanded to all members; got {sorted(support_names)}"
    # The inner selector saw FEWER columns than the original (medoid reduction).
    assert len(g.cluster_medoid_indices_) < X.shape[1]
