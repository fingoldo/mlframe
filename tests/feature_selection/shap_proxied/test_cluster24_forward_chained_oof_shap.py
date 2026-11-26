"""Cluster 24 regression: the out-of-fold SHAP stage is forward-chained under a temporal policy (audit 2026-11-20, finding 24-F45, SHAP part).

The partition trained each fold on every other block, including later ones, so a row's attribution came from a model that had seen its future. Now fold i
trains on the blocks before block i only. The first block has no earlier data and so no out-of-fold value: its rows are NaN in phi and base (not 0,
which would read as an exact zero attribution). Without a temporal policy every row is still explained.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.cv_policy import CVPolicy
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator


def _data(n=240, f=5, seed=0):
    """A small regression problem whose target depends on two of the columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"c{i}" for i in range(f)])
    y = (2.0 * X["c0"] - X["c1"] + 0.1 * rng.normal(size=n)).to_numpy()
    return X, y


def test_temporal_policy_leaves_the_first_block_unexplained_and_the_rest_finite():
    """Rows of the first block are NaN in phi and base; every later row has a finite attribution (24-F45)."""
    n, k = 240, 4
    X, y = _data(n)
    policy = CVPolicy("temporal", "test", timestamps=np.arange(n, dtype=np.float64))
    model = make_default_estimator(classification=False, n_estimators=15, random_state=0)
    phi, base, _y = compute_shap_matrix(
        model, X, y, classification=False, out_of_fold=True, n_splits=k, n_models=1, rng=np.random.default_rng(0), n_jobs=1, cv_policy=policy
    )
    first_block = np.arange(n // k)
    later = np.arange(n // k, n)
    assert np.isnan(base[first_block]).all() and np.isnan(phi[first_block]).all()
    assert np.isfinite(base[later]).all() and np.isfinite(phi[later]).all()


def test_no_policy_still_explains_every_row():
    """Without a temporal policy the partition covers all rows, as before (24-F45)."""
    X, y = _data(180)
    model = make_default_estimator(classification=False, n_estimators=15, random_state=0)
    phi, base, _y = compute_shap_matrix(model, X, y, classification=False, out_of_fold=True, n_splits=3, n_models=1, rng=np.random.default_rng(0), n_jobs=1)
    assert np.isfinite(base).all() and np.isfinite(phi).all()


def test_full_fit_under_a_temporal_policy_stays_aligned_and_finds_the_signal():
    """A real ShapProxiedFS fit on a temporal policy drops the first block's rows from every search-side array and still selects the signal (24-F45)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(3)
    n = 400
    ts = np.arange(n, dtype=np.float64)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    sel = ShapProxiedFS(classification=True, random_state=0, verbose=False, holdout_size=0.25, n_splits=3)
    sel._mlframe_cv_policy_ = CVPolicy("temporal", "test", ts)
    sel.fit(X, y)
    assert "f1" in list(sel.selected_features_)
    policy = sel._search_cv_policy
    assert policy is not None and len(policy.timestamps) < 300, "the search policy follows the rows that were kept"
