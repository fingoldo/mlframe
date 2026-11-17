"""Our TreeSHAP against the official shap on LightGBM models: boosting types, missing-value modes, bins, categoricals and the unmodelled modes that must decline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.shap_proxied.treeshap_vs_shap_common import (
    assert_matches_shap,
    fit_lgb,
    make_data,
    margin,
    shap_ref,
    ts,
)


@pytest.mark.parametrize("which", ["numba", "path", "table"])
@pytest.mark.parametrize("kind", ["reg", "bin"])
def test_regressor_and_binary(kind, which):
    """Regressor and binary classifier agree with shap to 1e-9 on every CPU kernel."""
    X, y = make_data(kind=kind)
    assert_matches_shap(fit_lgb(X, y, kind), X, which)


@pytest.mark.parametrize("boosting", ["gbdt", "dart", "goss", "rf"])
def test_boosting_types(boosting):
    """gbdt / dart / goss / rf boosters agree with shap (dart weights are folded into the leaf values by LightGBM)."""
    X, y = make_data()
    kw = dict(boosting_type=boosting)
    if boosting == "rf":
        kw.update(subsample=0.7, subsample_freq=1, colsample_bytree=0.8)
    assert_matches_shap(fit_lgb(X, y, **kw), X)


def test_binary_rf_boosting():
    """A random-forest binary classifier (averaged trees)."""
    X, y = make_data(kind="bin")
    assert_matches_shap(fit_lgb(X, y, "bin", boosting_type="rf", subsample=0.7, subsample_freq=1), X)


def test_use_missing_false():
    """use_missing=False models agree with shap on rows with NaN."""
    X, y = make_data()
    Xn = np.where(np.random.default_rng(5).random(X.shape) < 0.2, np.nan, X).astype(np.float32)
    assert_matches_shap(fit_lgb(X, y, use_missing=False), Xn)


@pytest.mark.parametrize("sign", [1, -1])
def test_default_left_both_directions(sign):
    """NaN learned to the left or to the right of a split (default_left true / false) routes like shap."""
    X, y = make_data()
    Xn = X.copy()
    Xn[sign * Xn[:, 0] > 1, 1] = np.nan
    yn = y + np.where(np.isnan(Xn[:, 1]), 3 * sign, 0)
    Xs = np.where(np.random.default_rng(5).random(X.shape) < 0.1, np.nan, Xn).astype(np.float32)
    assert_matches_shap(fit_lgb(Xn, yn), Xs)


@pytest.mark.parametrize("frac", [0.0, 0.2, 1.0])
def test_nan_fraction(frac):
    """NaN in 0 / 20 / 100 percent of the scored cells."""
    X, y = make_data()
    m = fit_lgb(np.where(np.random.default_rng(2).random(X.shape) < 0.2, np.nan, X), y)
    Xs = np.where(np.random.default_rng(3).random(X.shape) < frac, np.nan, X).astype(np.float32)
    assert_matches_shap(m, Xs)


def test_infinities():
    """Plus and minus infinity follow the extreme branches like shap."""
    X, y = make_data()
    Xi = X.copy()
    Xi[::7, 0] = np.inf
    Xi[::11, 1] = -np.inf
    assert_matches_shap(fit_lgb(X, y), Xi)


@pytest.mark.parametrize("kw", [dict(max_bin=4, min_data_in_bin=1), dict(max_bin=255, min_data_in_bin=1), dict(min_data_in_bin=1, min_child_samples=1)])
def test_tiny_min_data_in_bin_and_small_max_bin(kw):
    """Coarse and very fine binning move the thresholds; values still match shap."""
    X, y = make_data()
    assert_matches_shap(fit_lgb(X, y, **kw), X)


def test_monotone_constraints_are_exact():
    """Monotone constraints only affect the fit; the trees are ordinary, values match shap."""
    X, y = make_data()
    assert_matches_shap(fit_lgb(X, y, monotone_constraints=[1, 0, 0, 0, 0, 0]), X)


def test_stumps_and_single_leaf():
    """num_leaves=2 stumps and a constant target (single-leaf trees)."""
    X, y = make_data()
    assert_matches_shap(fit_lgb(X, y, num_leaves=2), X)
    phi, _ = assert_matches_shap(fit_lgb(X, np.ones(len(X))), X)
    assert np.all(phi == 0)


def test_very_deep_trees():
    """Leaf-wise trees far deeper than 12 levels."""
    X, y = make_data(n=3000)
    m = fit_lgb(X, y, num_leaves=1000, max_depth=-1, n_estimators=3, min_child_samples=1)
    assert m.booster_.dump_model()["tree_info"][0]["num_leaves"] > 200
    assert_matches_shap(m, X)


def test_feature_repeated_many_times_along_a_path():
    """Two features, deep leaf-wise trees: one feature occurs many times on one path."""
    X, y = make_data(n=1000, f=2)
    assert_matches_shap(fit_lgb(X, y, num_leaves=127, n_estimators=10, min_child_samples=2), X)


def test_many_trees_few_features_and_wide():
    """500 trees over 2 features; 5 trees over 1000 columns."""
    X, y = make_data(f=2)
    assert_matches_shap(fit_lgb(X, y, n_estimators=500), X)
    Xw = np.random.default_rng(3).normal(size=(200, 1000)).astype(np.float32)
    assert_matches_shap(fit_lgb(Xw, Xw[:, 0] + Xw[:, 5] * Xw[:, 7], n_estimators=5), Xw)


def test_duplicated_and_constant_columns():
    """A copy of a column and a constant column."""
    X, y = make_data()
    X[:, 4] = X[:, 0]
    X[:, 5] = 3.0
    phi, _ = assert_matches_shap(fit_lgb(X, y), X)
    assert np.all(phi[:, 5] == 0)


def test_very_large_magnitudes():
    """Features of magnitude 1e30 (near the float32 range)."""
    X, y = make_data()
    assert_matches_shap(fit_lgb(X * 1e30, y), X * 1e30)


# ---- unmodelled modes: the support check declines and the dispatcher falls back to shap ----


def _categorical():
    """A frame with a pandas categorical column and its target."""
    X, y = make_data()
    df = pd.DataFrame({"a": X[:, 0], "c": pd.Categorical(np.random.default_rng(1).integers(0, 5, len(X))), "b": X[:, 1]})
    return df, y + df["c"].cat.codes.to_numpy()


def _assert_declined_and_dispatcher_equals_shap(model, df):
    """The model is unsupported, extract_ensemble gives None, and the forced-numba dispatcher returns shap's values."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as spe

    assert not ts.is_supported_lightgbm(model)
    assert ts.extract_ensemble(model) is None
    phi, base = spe._shap_phi_and_base(model, df, backend="treeshap_numba")
    ref, ref_base = shap_ref(model, df)
    np.testing.assert_allclose(phi, ref, rtol=0, atol=1e-12)
    assert base == pytest.approx(ref_base, abs=1e-12)
    np.testing.assert_allclose(base + phi.sum(1), margin(model, df), rtol=0, atol=1e-9)


def test_categorical_features_decline_and_fall_back():
    """Categorical == splits are not modelled: decline cleanly, dispatcher equals shap."""
    df, y = _categorical()
    _assert_declined_and_dispatcher_equals_shap(fit_lgb(df, y), df)


def test_zero_as_missing_declines_and_falls_back():
    """zero_as_missing sends exact 0.0 down default_left: decline, dispatcher equals shap on rows holding zeros."""
    X, y = make_data()
    X[::3, 0] = 0.0
    _assert_declined_and_dispatcher_equals_shap(fit_lgb(X, y, zero_as_missing=True), pd.DataFrame(X))


def test_multiclass_declines():
    """Multiclass LightGBM is out of scope: no ensemble extracted."""
    X, y = make_data(kind="multi")
    m = fit_lgb(X, y, "bin")
    assert not ts.is_supported_lightgbm(m)
    assert ts.extract_ensemble(m) is None
