"""Validate GroupAwareMRMR as a faithful drop-in around the REAL mlframe RFECV
(audit integration-defaults-3), plus the safety/usefulness guard. This is the
real-path check before defaulting the cluster-medoid reduction ON in the suite.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _wide_corr(n=900, n_groups=5, per=6, n_noise=10, seed=0):
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(n_groups)]
    cols = {}
    for gi, z in enumerate(latents):
        for r in range(per):
            cols[f"L{gi}_{r}"] = z + 0.1 * rng.standard_normal(n)
    for j in range(n_noise):
        cols[f"noise{j}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    sig = latents[0] + latents[1]
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * sig))).astype(int))
    return X, y


def test_groupaware_wraps_real_mlframe_rfecv():
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _wide_corr()
    g = GroupAwareMRMR(
        RFECV(estimator=LogisticRegression(max_iter=500), cv=3, verbose=0),
        corr_threshold=0.9, corr_method="pearson",
    ).fit(X, y)
    # Faithful selector API: support_, transform, get_feature_names_out all work.
    assert hasattr(g, "support_") and len(g.support_) >= 1
    Xt = g.transform(X)
    assert Xt.shape[1] == len(g.support_)
    names = list(g.get_feature_names_out())
    assert len(names) == len(g.support_)
    assert set(names).issubset(set(X.columns))
    # The reduction actually engaged on this correlated frame.
    assert g.reduced_ is True and g.reduction_ > 0.0
    # transform column names match get_feature_names_out.
    assert list(Xt.columns) == names


def test_guard_bypasses_on_uncorrelated_data():
    # No correlated clusters -> reduction below min_reduction -> bypass to a full
    # inner fit. support_ must equal what the bare inner selector would give.
    from sklearn.feature_selection import RFECV as SkRFECV
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

    rng = np.random.default_rng(1)
    n = 800
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(20)})
    # a couple of features carry signal; none are mutually correlated
    y = pd.Series((X["f0"] + X["f1"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    est = SkRFECV(LogisticRegression(max_iter=500), cv=3, min_features_to_select=1)
    g = GroupAwareMRMR(est, corr_threshold=0.9, corr_method="pearson",
                       min_reduction=0.05).fit(X, y)
    assert g.reduced_ is False, "uncorrelated data must bypass the medoid path"
    # bare inner selection on full X (reference)
    from sklearn.base import clone
    ref = clone(est).fit(X, y)
    ref_idx = np.where(ref.support_)[0]
    assert list(g.support_) == list(ref_idx), (
        "bypass path must reproduce the bare inner selector's support"
    )


def test_registry_rfecv_is_cluster_reduced_by_default():
    # The suite instantiates RFECV via the registry; medoid reduction must be
    # DEFAULT-ON (returns a GroupAwareMRMR wrapping RFECV).
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection import registry
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

    sel = registry.get("RFECV").instantiate(
        estimator=LogisticRegression(max_iter=300), cv=3, verbose=0)
    assert isinstance(sel, GroupAwareMRMR), "RFECV must be cluster-reduced by default"
    X, y = _wide_corr(n=700)
    sel.fit(X, y)
    assert len(list(sel.get_feature_names_out())) == len(sel.support_) >= 1
    # __getattr__ passthrough: an attribute set on the inner selector (e.g. the
    # suite's markers RFECV stamps on itself) is visible on the wrapper.
    sel.estimator_._mlframe_demo_marker_ = "RFECV"
    assert sel._mlframe_demo_marker_ == "RFECV"


def test_registry_rfecv_cluster_reduce_false_returns_bare():
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection import registry
    from mlframe.feature_selection.wrappers import RFECV

    sel = registry.get("RFECV").instantiate(
        estimator=LogisticRegression(max_iter=300), cv=3, cluster_reduce=False)
    assert isinstance(sel, RFECV) and not hasattr(sel, "min_reduction")
