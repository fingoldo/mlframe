"""Unit tests for CompositeClassificationDiscovery (base-margin auto-selection)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from mlframe.training.composite import (
    CompositeClassificationDiscovery,
    discover_and_wrap_classification,
)


def _dominant_feature_data(n: int = 3000, seed: int = 0, n_noise: int = 6):
    """Binary y whose logit is dominated by x0, plus a weak x1*x2 interaction and noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3 + n_noise))
    logit = 2.5 * X[:, 0] + 0.6 * X[:, 1] * X[:, 2]
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def _tiny_capacity_inner(seed: int = 0):
    """Capacity-constrained booster: few trees/leaves, so it cannot cheaply
    approximate a steep continuous linear ramp the way a linear margin can --
    see test_biz_val_classification_discovery.py for the measured OOS effect."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=25, num_leaves=7, learning_rate=0.1, random_state=seed, verbose=-1, n_jobs=1, min_child_samples=15)


def _steep_ramp_data(n: int = 1200, seed: int = 0, n_noise: int = 18):
    """Steep single-feature logit + many noise columns; n=1200 measured (10/10
    seeds) to give the stage-2 paired CV + honest holdout enough rows to
    reliably accept the composite anchor (see test_biz_val_classification_discovery.py)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 1 + n_noise))
    logit = 8.0 * X[:, 0]
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def test_discovery_picks_dominant_column():
    X, y = _steep_ramp_data()
    disc = CompositeClassificationDiscovery(inner_estimator=_tiny_capacity_inner(), random_state=0, holdout_frac=0.25).fit(X, y)
    rec = disc.recommend()
    assert rec is not None, "steep-ramp synthetic must yield a recommendation"
    assert rec["column"] == "col_0"
    assert disc.best_estimator_ is not None
    proba = disc.best_estimator_.predict_proba(X[:50])
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_stage1_table_is_ranked_and_complete():
    X, y = _dominant_feature_data(n=1500)
    disc = CompositeClassificationDiscovery(random_state=0).fit(X, y)
    res = disc.result_
    gains = [r["margin_gain"] for r in res.candidates]
    assert gains == sorted(gains, reverse=True)
    assert res.baseline_logloss > 0
    assert any(r["column"] == "col_0" for r in res.candidates)


def test_leak_column_rejected():
    X, y = _dominant_feature_data(n=1500)
    # append a perfect-leak column: the label itself, jittered infinitesimally
    leak = y.astype(np.float64) + np.random.default_rng(1).normal(0, 1e-9, y.size)
    X_leak = np.column_stack([X, leak])
    disc = CompositeClassificationDiscovery(random_state=0).fit(X_leak, y)
    leak_name = f"col_{X_leak.shape[1] - 1}"
    row = next(r for r in disc.result_.candidates if r["column"] == leak_name)
    assert row["suspected_leak"] is True
    rec = disc.recommend()
    assert rec is None or rec["column"] != leak_name


def test_pure_noise_recommends_plain_model():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (1200, 5))
    y = rng.integers(0, 2, 1200)
    disc = CompositeClassificationDiscovery(random_state=0).fit(X, y)
    assert disc.recommend() is None
    assert disc.best_estimator_ is None


def test_multiclass_supported():
    rng = np.random.default_rng(4)
    n = 2400
    X = rng.normal(0, 1, (n, 5))
    # 3-class: class score k driven by x0 thresholds -> x0 is the dominant margin source
    y = np.digitize(1.5 * X[:, 0] + rng.normal(0, 0.7, n), [-1.0, 1.0])
    disc = CompositeClassificationDiscovery(random_state=0).fit(X, y)
    assert any("cv_gain" in r for r in disc.result_.candidates)
    if disc.best_estimator_ is not None:
        proba = disc.best_estimator_.predict_proba(X[:20])
        assert proba.shape == (20, 3)


def test_discover_and_wrap_always_returns_usable_model():
    X, y = _steep_ramp_data()
    model, result = discover_and_wrap_classification(X, y, inner_estimator=_tiny_capacity_inner(), random_state=0, holdout_frac=0.25)
    assert hasattr(model, "predict_proba")
    assert model.predict_proba(X[:10]).shape == (10, 2)
    assert result.baseline_logloss > 0
    # pure-noise path returns the plain fitted model, never None
    rng = np.random.default_rng(5)
    Xn, yn = rng.normal(0, 1, (900, 4)), rng.integers(0, 2, 900)
    model_n, result_n = discover_and_wrap_classification(Xn, yn, random_state=0)
    assert hasattr(model_n, "predict_proba")
    assert result_n.best is None


def test_pandas_input_and_forbidden_patterns():
    pd = pytest.importorskip("pandas")
    X, y = _steep_ramp_data()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target_copy"] = y  # name-forbidden even though it is a perfect leak
    disc = CompositeClassificationDiscovery(inner_estimator=_tiny_capacity_inner(), random_state=0, holdout_frac=0.25).fit(df, y)
    assert all(r["column"] != "target_copy" for r in disc.result_.candidates)
    rec = disc.recommend()
    assert rec is not None and rec["column"] == "f0"


def test_single_class_raises():
    X = np.random.default_rng(0).normal(0, 1, (100, 3))
    with pytest.raises(ValueError, match=">= 2 classes"):
        CompositeClassificationDiscovery().fit(X, np.zeros(100))
