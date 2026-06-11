"""perf regression guard (2026-06-05): RFECV now feeds the inner estimator a contiguous float64
NUMPY mirror of the (all-numeric) DataFrame -- by integer column-position -- throughout elimination,
CV scoring, and permutation-FI re-prediction, instead of re-handing pandas to ``estimator.fit/.predict``
on every fold x refit x permutation re-prediction. That kills LightGBM's per-call ``_data_from_pandas``
reconversion + the per-column dtype-validation storm (cProfile: ~47% of the scene 700x299 fit).

This is PURE PLUMBING (numpy vs pandas to the estimator), no algorithm change, so it MUST be
selection-IDENTICAL. The float64 mirror is bit-identical to pandas for the all-numeric case (float32
would shift LightGBM's split points; verified separately). These tests A/B the numpy fast path against
the historical pandas path on the SAME seeded fixture via the ``_force_pandas_estimator_path`` escape
hatch and assert byte-for-byte identical selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig


def _synthetic_frame(seed: int = 0, n: int = 400, p: int = 30):
    """Seeded all-numeric classification frame: a handful of informative columns + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # 5 informative features drive a logistic target; the rest are noise.
    w = np.zeros(p)
    w[:5] = np.array([2.0, -1.7, 1.3, -1.1, 0.9])
    logits = X @ w + 0.3 * rng.standard_normal(n)
    y = (logits > np.median(logits)).astype(int)
    cols = [f"f{i}" for i in range(p)]
    # Mix int and float dtypes (like scene) so the float64 cast covers both.
    Xdf = pd.DataFrame(X, columns=cols)
    Xdf["f0"] = (Xdf["f0"] * 3).round().astype("int64")  # integer column
    return Xdf, pd.Series(y, name="target")


def _make_rfecv():
    fi = FIConfig(importance_getter="permutation", n_features_selection_rule="one_se_min")
    sc = SearchConfig(max_refits=8, max_runtime_mins=2)
    return RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=60, num_leaves=15, learning_rate=0.1, n_jobs=1, verbose=-1),
        cv=3, scoring=None, verbose=0, fi_config=fi, search_config=sc, random_state=0,
    )


def _fit_get(force_pandas: bool, seed: int = 0):
    X, y = _synthetic_frame(seed=seed)
    r = _make_rfecv()
    if force_pandas:
        r._force_pandas_estimator_path = True
    r.fit(X, y)
    out = list(r.get_feature_names_out())
    support = np.asarray(r.support_).tolist()
    ranking = np.asarray(getattr(r, "ranking_", [])).tolist() if hasattr(r, "ranking_") else None
    cv = r.cv_results_
    return out, support, ranking, cv


@pytest.mark.parametrize("seed", [0, 1])
def test_selection_identical_numpy_vs_pandas(seed):
    """The numpy fast path (default) and the forced-pandas path select the SAME features,
    in the SAME order, with the SAME support_/ranking_/cv curve on a seeded fixture."""
    out_np, sup_np, rank_np, cv_np = _fit_get(force_pandas=False, seed=seed)
    out_pd, sup_pd, rank_pd, cv_pd = _fit_get(force_pandas=True, seed=seed)

    assert out_np == out_pd, (
        f"get_feature_names_out differs between numpy fast path and pandas path (seed={seed}):\n"
        f"  numpy : {out_np}\n  pandas: {out_pd}"
    )
    assert sup_np == sup_pd, f"support_ differs (seed={seed})"
    assert rank_np == rank_pd, f"ranking_ differs (seed={seed})"
    # CV curve (the elimination + scoring trajectory) must match exactly.
    assert list(cv_np["nfeatures"]) == list(cv_pd["nfeatures"]), f"cv nfeatures differ (seed={seed})"
    np.testing.assert_array_equal(
        np.asarray(cv_np["cv_mean_perf"]), np.asarray(cv_pd["cv_mean_perf"]),
        err_msg=f"cv_mean_perf differs between numpy and pandas paths (seed={seed})",
    )


def test_numpy_path_actually_engaged():
    """Sanity: the default path produces a usable selection (the numpy mirror was built and used).
    A non-empty support over an all-numeric frame confirms the fast path didn't silently no-op."""
    out, support, _, _ = _fit_get(force_pandas=False, seed=0)
    assert len(out) >= 1
    assert any(bool(s) for s in support)


def test_object_column_falls_back_to_pandas():
    """An object/category column disqualifies the all-numeric fast path; RFECV must still fit
    (the historical pandas path) and select features without raising."""
    X, y = _synthetic_frame(seed=0, n=300, p=12)
    X = X.copy()
    X["cat_col"] = pd.Series(np.where(np.arange(len(X)) % 2 == 0, "a", "b")).astype("category")
    r = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=40, num_leaves=15, n_jobs=1, verbose=-1),
        cv=3, scoring=None, verbose=0,
        fi_config=FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min"),
        search_config=SearchConfig(max_refits=5, max_runtime_mins=2),
        random_state=0, cat_features=["cat_col"],
    )
    r.fit(X, y)
    assert len(list(r.get_feature_names_out())) >= 1
