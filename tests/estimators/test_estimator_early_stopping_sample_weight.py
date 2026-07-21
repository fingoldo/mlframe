"""Regression tests for EstimatorWithEarlyStopping sample-weight propagation.

The CatBoost early-stopping branch splits X/y into train/val internally. A
full-length ``sample_weight`` passed through ``fit_params`` must be split the
same way: the train-fold weights go to ``fit``, the val-fold weights feed the
``eval_set`` Pool. Before the fix the full-length weight reached the smaller
train fold and CatBoost raised a length-mismatch error (a silent desync that
also misaligned weights to rows when lengths happened to match).
"""

import numpy as np
import pytest

from mlframe.estimators.base import ClassifierWithEarlyStopping, RegressorWithEarlyStopping

CatBoostClassifier = pytest.importorskip("catboost").CatBoostClassifier
CatBoostRegressor = pytest.importorskip("catboost").CatBoostRegressor


def _data(n=200, seed=0):
    """Builds seeded synthetic test data; returns ``(X, y, w)``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    w = rng.uniform(0.5, 2.0, size=n)
    return X, y, w


def test_classifier_early_stopping_sample_weight_split_aligns():
    # Pre-fix: full-length weight (n=200) hit the 160-row train fold -> CatBoostError.
    """Classifier early stopping sample weight split aligns."""
    X, y, w = _data()
    est = ClassifierWithEarlyStopping(
        base_estimator=CatBoostClassifier(iterations=10, verbose=False),
        test_size=0.2,
        random_state=0,
    )
    est.fit(X, y, sample_weight=w)
    assert est.predict(X).shape == (len(X),)


def test_regressor_early_stopping_sample_weight_split_aligns():
    """Regressor early stopping sample weight split aligns."""
    X, y, w = _data()
    y = y.astype(float) + np.arange(len(y)) * 0.01
    est = RegressorWithEarlyStopping(
        base_estimator=CatBoostRegressor(iterations=10, verbose=False),
        test_size=0.25,
        random_state=1,
    )
    est.fit(X, y, sample_weight=w)
    assert est.predict(X).shape == (len(X),)


def test_early_stopping_weights_are_honored_not_just_length_matched():
    # Extreme weights must measurably change the fitted model vs uniform weights,
    # proving the train-fold weights actually reach fit (not silently dropped).
    """Early stopping weights are honored not just length matched."""
    X, y, _ = _data(n=300, seed=2)
    w = np.where(y == 1, 50.0, 0.01)

    base = dict(iterations=40, depth=2, verbose=False, random_seed=0)
    weighted = ClassifierWithEarlyStopping(base_estimator=CatBoostClassifier(**base), test_size=0.2, random_state=0).fit(X, y, sample_weight=w)
    uniform = ClassifierWithEarlyStopping(base_estimator=CatBoostClassifier(**base), test_size=0.2, random_state=0).fit(X, y)

    p_w = weighted.predict_proba(X)[:, 1].mean()
    p_u = uniform.predict_proba(X)[:, 1].mean()
    # Up-weighting the positive class should lift its mean predicted probability.
    assert p_w > p_u + 0.05, f"weighted mean P(1)={p_w:.3f} not above uniform {p_u:.3f}"
