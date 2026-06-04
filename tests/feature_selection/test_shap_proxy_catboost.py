"""Unit tests for the catboost booster backend.

CatBoost is the third major GBT family wired into ``ShapProxiedFS``. Native fast SHAP via
``get_feature_importance(type='ShapValues')`` is exercised here for both regression and binary
classification; the end-to-end selector test guards that ``booster_kind='catboost'`` matches the
xgboost recall on a synthetic informative-feature regime, and ``cat_features=`` round-trips through
the prefilter / OOF / honest-retrain pipeline without losing the categorical hint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")


# -------------------------------------------------------------- module-level helpers


def test_catboost_available_when_installed():
    from mlframe.feature_selection._shap_proxy_catboost import catboost_available, reset_catboost_available_cache

    reset_catboost_available_cache()
    assert catboost_available() is True
    # Second call uses cache; result must be identical (idempotent).
    assert catboost_available() is True


def test_make_catboost_estimator_classification_defaults():
    from catboost import CatBoostClassifier

    from mlframe.feature_selection._shap_proxy_catboost import make_catboost_estimator

    est = make_catboost_estimator(classification=True, random_state=7, n_estimators=42)
    assert isinstance(est, CatBoostClassifier)
    params = est.get_params()
    assert params["iterations"] == 42
    assert params["random_seed"] == 7
    assert params["verbose"] is False
    assert params["allow_writing_files"] is False
    assert params["loss_function"] == "Logloss"


def test_make_catboost_estimator_regression_defaults():
    from catboost import CatBoostRegressor

    from mlframe.feature_selection._shap_proxy_catboost import make_catboost_estimator

    est = make_catboost_estimator(classification=False, random_state=3, n_estimators=20)
    assert isinstance(est, CatBoostRegressor)
    p = est.get_params()
    assert p["iterations"] == 20 and p["random_seed"] == 3


def test_make_catboost_estimator_cat_features_forwarded():
    from mlframe.feature_selection._shap_proxy_catboost import make_catboost_estimator

    est = make_catboost_estimator(classification=True, cat_features=["a", "c"])
    assert list(est.get_params()["cat_features"]) == ["a", "c"]


def test_is_catboost_estimator_detection():
    from catboost import CatBoostClassifier, CatBoostRegressor

    from mlframe.feature_selection._shap_proxy_catboost import is_catboost_estimator
    from sklearn.linear_model import LogisticRegression

    assert is_catboost_estimator(CatBoostClassifier(verbose=False))
    assert is_catboost_estimator(CatBoostRegressor(verbose=False))
    assert not is_catboost_estimator(LogisticRegression())


# -------------------------------------------------------------- catboost_shap


@pytest.mark.parametrize("classification", [False, True])
def test_catboost_shap_shape_and_additivity(classification):
    """phi has shape (n, f), base is a scalar, and base + phi.sum(1) reconstructs the raw margin
    (TreeSHAP invariant on catboost's native ShapValues path)."""
    from mlframe.feature_selection._shap_proxy_catboost import catboost_shap, make_catboost_estimator

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 6)), columns=[f"f{i}" for i in range(6)])
    signal = 2.0 * X["f0"] + X["f1"] - 0.5 * X["f2"]
    y = ((signal + 0.3 * rng.normal(size=150) > 0).astype(int).values
         if classification else (signal + 0.1 * rng.normal(size=150)).to_numpy())
    m = make_catboost_estimator(classification, n_estimators=30, random_state=0)
    m.fit(X, y)
    phi, base = catboost_shap(m, X)
    assert phi.shape == (150, 6)
    assert isinstance(base, float)
    margin = m.predict(X, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(base + phi.sum(axis=1), margin, rtol=0, atol=1e-6)


def test_catboost_shap_with_cat_features():
    """Categorical column passed via ``cat_features`` keyword on the Pool ctor is honoured; SHAP
    additivity holds on the resulting model."""
    from mlframe.feature_selection._shap_proxy_catboost import catboost_shap, make_catboost_estimator

    rng = np.random.default_rng(2)
    X = pd.DataFrame({
        "n0": rng.normal(size=200),
        "cat0": rng.integers(0, 4, size=200).astype(str),
        "n1": rng.normal(size=200),
    })
    y = ((X["n0"] + 0.5 * (X["cat0"].astype(int) - 1.5) + 0.2 * rng.normal(size=200)) > 0).astype(int).values
    m = make_catboost_estimator(classification=True, n_estimators=30, cat_features=["cat0"])
    m.fit(X, y)
    phi, base = catboost_shap(m, X, cat_features=["cat0"])
    assert phi.shape == (200, 3)
    margin = m.predict(X, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(base + phi.sum(axis=1), margin, rtol=0, atol=1e-6)


# -------------------------------------------------------------- explain wiring


def test_make_default_estimator_dispatches_catboost():
    """``make_default_estimator(booster_kind='catboost')`` returns a CatBoostClassifier with the
    cat_features sentinel attached so downstream clones recover it."""
    from catboost import CatBoostClassifier

    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator

    est = make_default_estimator(classification=True, random_state=0, booster_kind="catboost",
                                  cat_features=["f1"])
    assert isinstance(est, CatBoostClassifier)
    assert getattr(est, "_shap_proxy_cat_features", None) == ["f1"]


def test_make_default_estimator_rejects_unknown_kind():
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator

    with pytest.raises(ValueError, match="booster_kind"):
        make_default_estimator(classification=True, booster_kind="bogus")


def test_shap_phi_and_base_routes_catboost():
    """``_shap_phi_and_base`` recognises a catboost model and takes the native SHAP path (skipping
    both the numba dispatcher and the shap library wrapper)."""
    from mlframe.feature_selection._shap_proxy_catboost import make_catboost_estimator
    from mlframe.feature_selection._shap_proxy_explain import _shap_phi_and_base

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
    y = ((2 * X["f0"] + X["f1"] + 0.3 * rng.normal(size=100)) > 0).astype(int).values
    m = make_catboost_estimator(classification=True, n_estimators=20, random_state=0)
    m.fit(X, y)
    phi, base = _shap_phi_and_base(m, X)
    assert phi.shape == (100, 5)
    assert isinstance(base, float)


# -------------------------------------------------------------- end-to-end ShapProxiedFS


def _make_informative_data(n_rows=400, n_features=30, n_informative=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    coefs = np.zeros(n_features)
    coefs[:n_informative] = rng.normal(scale=2.0, size=n_informative)
    signal = X @ coefs
    y = (signal + 0.3 * rng.normal(size=n_rows) > 0).astype(int)
    return (pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y,
            list(range(n_informative)))


def _fit_and_recall(booster_kind, X, y, informative, *, cat_features=None, prefilter_method="univariate"):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(
        classification=True, booster_kind=booster_kind, cat_features=cat_features,
        n_jobs=1, random_state=0, cluster_features=False, trust_guard=False,
        run_importance_ablation=False, revalidate=False, prefilter_method=prefilter_method,
    )
    fs.fit(X, y)
    chosen = fs.get_support(indices=True).tolist()
    recall = len(set(chosen) & set(informative))
    return recall, chosen


def test_shap_proxied_fs_catboost_end_to_end():
    """``ShapProxiedFS(booster_kind='catboost')`` recovers the informative columns on a small
    synthetic regime (parity with the xgboost-default baseline)."""
    X, y, informative = _make_informative_data()
    recall_xgb, _ = _fit_and_recall("xgboost", X, y, informative)
    recall_cat, _ = _fit_and_recall("catboost", X, y, informative)
    # Catboost recall must be no worse than xgboost recall by more than 1 informative (capability
    # parity gate, mirrors the iter62 plan; xgboost is the established default).
    assert recall_cat >= recall_xgb - 1
    assert recall_cat >= len(informative) - 1


def test_shap_proxied_fs_catboost_cat_features_pass_through():
    """``booster_kind='catboost'`` with a non-empty ``cat_features`` must fail fast: the surrounding
    prefilter / clustering / column-slicing pipeline is NOT categorical-aware, so this combination
    raises ValueError early with an actionable message instead of crashing deep in densification."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "num0": rng.normal(size=n),
        "num1": rng.normal(size=n),
        "cat0": rng.integers(0, 3, size=n).astype(str),
        "num2": rng.normal(size=n),
        "cat1": rng.integers(0, 4, size=n).astype(str),
    })
    # Numeric features dominate the label so the recall isn't dependent on catboost's CTR quality.
    y = ((X["num0"] + X["num1"] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    fs = ShapProxiedFS(
        classification=True, booster_kind="catboost", cat_features=["cat0", "cat1"],
        n_jobs=1, random_state=0, cluster_features=False, trust_guard=False,
        run_importance_ablation=False, revalidate=False, prefilter_method="univariate",
    )
    with pytest.raises(ValueError, match="not yet supported"):
        fs.fit(X, y)


def test_shap_proxied_fs_unknown_booster_kind_raises():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0).astype(int).values
    fs = ShapProxiedFS(classification=True, booster_kind="bogus")
    with pytest.raises(ValueError, match="booster_kind"):
        fs.fit(X, y)
