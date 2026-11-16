"""Our TreeSHAP against the official shap on XGBoost models: families, tree methods, base_score, weights, NaN, and the documented multiclass fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.shap_proxied.treeshap_vs_shap_common import (
    assert_matches_shap,
    assert_same_rank,
    fit_xgb,
    make_data,
    margin,
    ours,
    shap_ref,
    ts,
    xgb_ref64,
)

KINDS = [("reg", "reg"), ("bin", "bin")]


@pytest.mark.parametrize("which", ["numba", "path", "table"])
@pytest.mark.parametrize("kind", ["reg", "bin"])
def test_regressor_and_binary_match_shap_float64_kernel(kind, which):
    """Regressor and binary classifier agree with shap's float64 C++ kernel to 1e-9 on every CPU kernel."""
    X, y = make_data(kind=kind)
    assert_matches_shap(fit_xgb(X, y, kind), X, which)


@pytest.mark.parametrize("kind", ["reg", "bin"])
def test_xgboost_matches_shap_native_float32(kind):
    """Against ``TreeExplainer(xgb_model)`` itself (xgboost's native float32 pred_contribs) the difference stays below 1e-5, ranks equal."""
    X, y = make_data(kind=kind)
    m = fit_xgb(X, y, kind)
    ref, _ = shap_ref(m, X)
    phi, base = ours(m, X)
    np.testing.assert_allclose(phi, ref, rtol=0, atol=1e-5)
    assert_same_rank(phi, ref)
    np.testing.assert_allclose(base + phi.sum(1), margin(m, X), rtol=0, atol=1e-5)


@pytest.mark.parametrize("method", ["exact", "hist", "approx"])
def test_tree_methods(method):
    """exact / hist / approx tree builders all extract and agree with shap."""
    X, y = make_data()
    assert_matches_shap(fit_xgb(X, y, tree_method=method), X)


@pytest.mark.parametrize("base_score", [0.0, 0.7, -3.0])
def test_base_score(base_score):
    """An explicit base_score is carried in the base value: additivity and values match shap."""
    X, y = make_data()
    assert_matches_shap(fit_xgb(X, y, base_score=base_score), X)


def test_binary_base_score_is_in_log_odds():
    """A binary model with base_score 0.8 (a probability) has the log-odds margin as its base."""
    X, y = make_data(kind="bin")
    assert_matches_shap(fit_xgb(X, y, "bin", base_score=0.8), X)


def test_scale_pos_weight():
    """scale_pos_weight changes the covers (hessian sums); the cover-weighted values still match shap."""
    X, y = make_data(kind="bin")
    assert_matches_shap(fit_xgb(X, y, "bin", scale_pos_weight=4.0), X)


def test_sample_weights_change_covers():
    """Sample weights enter the node covers; values match shap and differ from the unweighted fit."""
    X, y = make_data()
    w = np.random.default_rng(1).uniform(0.1, 5, len(y))
    m = fit_xgb(X, y, sample_weight=w)
    phi, _ = assert_matches_shap(m, X)
    assert not np.allclose(phi, ours(fit_xgb(X, y), X)[0])


@pytest.mark.parametrize("frac", [0.2, 1.0])
def test_nan_fraction_in_scoring_rows(frac):
    """NaN in 20 percent / 100 percent of the scored cells follows the default child like shap."""
    X, y = make_data()
    m = fit_xgb(np.where(np.random.default_rng(2).random(X.shape) < 0.2, np.nan, X), y)
    Xs = np.where(np.random.default_rng(3).random(X.shape) < frac, np.nan, X).astype(np.float32)
    assert_matches_shap(m, Xs)


def test_nan_in_training_and_scoring_zero_percent():
    """A model trained with missing values scored on complete rows matches shap."""
    X, y = make_data()
    Xn = np.where(np.random.default_rng(2).random(X.shape) < 0.2, np.nan, X).astype(np.float32)
    assert_matches_shap(fit_xgb(Xn, y), X)


def test_xgboost_rejects_inf_at_the_booster_not_in_our_code():
    """xgboost refuses infinities in a DMatrix, so shap and the margin reference cannot score them; our kernel is not asked to."""
    import xgboost as xgb

    X, _ = make_data()
    X[0, 0] = np.inf
    with pytest.raises(xgb.core.XGBoostError):
        xgb.DMatrix(X)


@pytest.mark.parametrize("kw", [dict(max_depth=1), dict(max_depth=2, n_estimators=30)])
def test_stumps_and_shallow(kw):
    """Depth-1 stumps (one feature per tree) match shap."""
    X, y = make_data()
    assert_matches_shap(fit_xgb(X, y, **kw), X)


def test_constant_target_single_leaf_trees():
    """A constant target gives single-leaf trees: all phi are zero, the base is the margin."""
    X, _ = make_data()
    m = fit_xgb(X, np.ones(len(X)))
    phi, _ = assert_matches_shap(m, X)
    assert np.all(phi == 0)


def test_deep_trees_depth_14():
    """Depth 14 trees (min_child_weight 0) match shap on the numba path and table kernels."""
    X, y = make_data(n=3000)
    m = fit_xgb(X, y, max_depth=14, n_estimators=3, min_child_weight=0, reg_lambda=0)
    for which in ("path", "table"):
        assert_matches_shap(m, X, which)


def test_feature_repeated_along_a_path():
    """Two features and deep trees repeat features along a path; the unwind of repeated features matches shap."""
    X, y = make_data(f=2)
    assert_matches_shap(fit_xgb(X, y, max_depth=8, n_estimators=10), X)


def test_many_trees_few_features():
    """500 trees on 2 features."""
    X, y = make_data(f=2)
    assert_matches_shap(fit_xgb(X, y, n_estimators=500, max_depth=6), X)


def test_few_trees_wide_matrix():
    """Five trees over 1000 columns."""
    X = np.random.default_rng(3).normal(size=(200, 1000)).astype(np.float32)
    assert_matches_shap(fit_xgb(X, X[:, 0] + X[:, 5] * X[:, 7], n_estimators=5), X)


def test_duplicated_and_constant_columns():
    """Duplicated and constant columns share or never get credit exactly like shap."""
    X, y = make_data()
    X[:, 4] = X[:, 0]
    X[:, 5] = 3.0
    m = fit_xgb(X, y)
    phi, _ = assert_matches_shap(m, X)
    assert np.all(phi[:, 5] == 0)


def test_named_dataframe_columns():
    """A booster with feature names (pandas fit) is extracted by name."""
    X, y = make_data()
    df = pd.DataFrame(X, columns=list("abcdef"))
    assert_matches_shap(fit_xgb(df, y), df.to_numpy())


def test_multiclass_is_declined():
    """Multiclass is not modelled: the support check says no and no ensemble is extracted."""
    X, y = make_data(kind="multi")
    m = fit_xgb(X, y, "bin")
    assert not ts.is_supported_xgboost(m)
    assert ts.extract_ensemble(m) is None


def test_multiclass_dispatcher_matches_shap_or_raises_clearly():
    """The dispatcher on a multiclass model never returns our kernel's numbers: it takes the shap route (3-D output rejected with a clear message)."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as spe

    X, y = make_data(kind="multi")
    m = fit_xgb(X, y, "bin")
    with pytest.raises(ValueError, match="binary|multiclass|classes"):
        spe._shap_phi_and_base(m, pd.DataFrame(X), backend="treeshap_numba")


def test_reference_float64_helper_is_additive():
    """Sanity of the reference itself: shap's float64 kernel on the parsed trees is additive up to xgboost's float32 margin."""
    X, y = make_data()
    m = fit_xgb(X, y)
    ref = xgb_ref64(m, X)
    _, base = ours(m, X)
    np.testing.assert_allclose(base + ref.sum(1), margin(m, X), rtol=0, atol=5e-6)
