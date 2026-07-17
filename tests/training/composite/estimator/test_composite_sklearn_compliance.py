"""sklearn-API compliance for the composite estimators.

Two contracts pinned across the affected estimator classes:

1. Calling predict / transform / predict_quantile / predict_proba BEFORE fit raises ``sklearn.exceptions.NotFittedError``
   (not a bare ``RuntimeError`` / ``AttributeError``), so the standard sklearn ``check_is_fitted`` error contract holds.
2. After fit, ``n_features_in_`` is set to the real feature count of the X seen at fit (the named-column length).

Plus targeted regressions: bagging re-validates moved-out params in fit, the quantile wrapper stores ``alphas`` verbatim
(clone round-trips a list), and the tail composite does not mutate its ``base_estimator`` prototype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

from mlframe.training.composite.panel import CompositePanelEstimator
from mlframe.training.composite.orthogonal import OrthogonalizedCompositeEstimator
from mlframe.training.composite.suite_features import CompositeFeatureGenerator
from mlframe.training.composite.simplex import CompositeSimplexEstimator
from mlframe.training.composite.meta import CompositeOrRawStacker
from mlframe.training.composite.missing import MissingAwareComposite
from mlframe.training.composite.quantile import CompositeQuantileEstimator
from mlframe.training.composite.qrf import CompositeQRFEstimator
from mlframe.training.composite.bagging import BaggedCompositeEstimator
from mlframe.training.composite.survival import CompositeSurvivalEstimator
from mlframe.training.composite.extremes import TailCompositeEstimator
from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.quantile_wrapper import _QuantileMultiOutputWrapper


N = 80
N_FEATURES = 4  # cols: base, f1, f2, f3


def _make_X(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base": rng.normal(size=N),
            "f1": rng.normal(size=N),
            "f2": rng.normal(size=N),
            "f3": rng.normal(size=N),
        }
    )


def _make_y(X: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    return (1.5 * X["base"] + 0.5 * X["f1"] - 0.3 * X["f2"]).to_numpy() + rng.normal(scale=0.1, size=N)


# Each case: (id, factory, fit_callable(est, X, y), predict_callable(est, X), expected_n_features_in_).
# Most estimators report the full X column count (4); panel drops the entity column from the inner matrix, so it reports 3.
def _fit_xy(est, X, y):
    est.fit(X, y)


def _predict(est, X):
    est.predict(X)


_CASES = [
    (
        "panel",
        lambda: CompositePanelEstimator(inner_estimator=Ridge(), entity_column="f3"),
        _fit_xy,
        _predict,
        N_FEATURES - 1,  # entity column dropped from the inner feature matrix
    ),
    (
        "orthogonal",
        lambda: OrthogonalizedCompositeEstimator(inner_estimator=Ridge(), base_column="base", n_folds=3),
        _fit_xy,
        _predict,
        N_FEATURES,
    ),
    (
        "suite_features",
        lambda: CompositeFeatureGenerator(
            wrapper_factory=lambda: CompositeTargetEstimator(base_estimator=Ridge(), base_column="base", transform_name="diff"),
            column_name="composite_feat",
        ),
        _fit_xy,
        lambda est, X: est.transform(X),
        N_FEATURES,
    ),
    (
        "simplex",
        lambda: CompositeSimplexEstimator(base_estimator=Ridge(), transform="ilr"),
        lambda est, X, y: est.fit(X, _simplex_target(y)),
        _predict,
        N_FEATURES,
    ),
    (
        "meta",
        lambda: CompositeOrRawStacker(base_estimator=Ridge(), transform_name="diff", base_column="base", n_splits=3),
        _fit_xy,
        _predict,
        N_FEATURES,
    ),
    (
        "missing",
        lambda: MissingAwareComposite(composite=CompositeTargetEstimator(base_estimator=Ridge(), base_column="base", transform_name="diff")),
        _fit_xy,
        _predict,
        N_FEATURES,
    ),
    (
        "quantile",
        lambda: CompositeQuantileEstimator(
            base_estimator=__import__("lightgbm").LGBMRegressor(n_estimators=10, verbose=-1),
            base_column="base",
            transform_name="linear_residual",
            quantiles=(0.25, 0.5, 0.75),
        ),
        _fit_xy,
        lambda est, X: est.predict_quantile(X),
        N_FEATURES,
    ),
    (
        "qrf",
        lambda: CompositeQRFEstimator(base_column="base", transform_name="linear_residual", n_estimators=20, prefer_quantile_forest=False),
        _fit_xy,
        lambda est, X: est.predict_quantile(X),
        N_FEATURES,
    ),
    (
        "bagging",
        lambda: BaggedCompositeEstimator(
            base_estimator=CompositeTargetEstimator(base_estimator=Ridge(), base_column="base", transform_name="diff"),
            n_estimators=3,
            random_state=0,
        ),
        _fit_xy,
        _predict,
        N_FEATURES,
    ),
    (
        "survival",
        lambda: CompositeSurvivalEstimator(base_estimator=Ridge(), base_column="base", censoring="observed_only"),
        lambda est, X, y: est.fit(X, _positive(y), event=np.ones(N)),
        _predict,
        N_FEATURES,
    ),
    (
        "extremes",
        lambda: TailCompositeEstimator(base_estimator=Ridge(), base_column="base", transform_name="diff"),
        _fit_xy,
        _predict,
        N_FEATURES,
    ),
]


def _simplex_target(y: np.ndarray) -> np.ndarray:
    """Build a valid (n, 3) composition from y (strictly positive rows summing to 1)."""
    a = np.abs(y) + 0.1
    b = np.abs(y - y.mean()) + 0.1
    c = np.ones_like(y) * 0.5
    mat = np.column_stack([a, b, c])
    return mat / mat.sum(axis=1, keepdims=True)


def _positive(y: np.ndarray) -> np.ndarray:
    return np.abs(y) + 1.0


@pytest.mark.parametrize("case_id,factory,fit_fn,predict_fn,expected_nf", _CASES, ids=[c[0] for c in _CASES])
def test_composite_predict_before_fit_raises_not_fitted(case_id, factory, fit_fn, predict_fn, expected_nf):
    """Contract 1: predict / transform before fit raises sklearn NotFittedError, not RuntimeError/AttributeError."""
    rng = np.random.default_rng(0)
    X = _make_X(rng)
    est = factory()
    with pytest.raises(NotFittedError):
        predict_fn(est, X)


@pytest.mark.parametrize("case_id,factory,fit_fn,predict_fn,expected_nf", _CASES, ids=[c[0] for c in _CASES])
def test_composite_sets_n_features_in_after_fit(case_id, factory, fit_fn, predict_fn, expected_nf):
    """Contract 2: after fit, n_features_in_ equals the real feature count of the X seen at fit."""
    rng = np.random.default_rng(1)
    X = _make_X(rng)
    y = _make_y(X, rng)
    est = factory()
    fit_fn(est, X, y)
    assert getattr(est, "n_features_in_", None) == expected_nf, f"{case_id}: n_features_in_={getattr(est, 'n_features_in_', None)!r}, expected {expected_nf}"
    # predict path works after fit (no regression in the happy path).
    predict_fn(est, X)


def test_bagging_set_params_revalidated_in_fit():
    """bagging moved __init__ validation into fit: a bad aggregation set AFTER construction must raise at fit, not __init__."""
    rng = np.random.default_rng(2)
    X = _make_X(rng)
    y = _make_y(X, rng)
    est = BaggedCompositeEstimator(
        base_estimator=CompositeTargetEstimator(base_estimator=Ridge(), base_column="base", transform_name="diff"),
        n_estimators=2,
    )
    # __init__ no longer validates -> set_params with a bad value does not raise here.
    est.set_params(aggregation="not_a_real_aggregation")
    with pytest.raises(ValueError, match="aggregation"):
        est.fit(X, y)
    # trim_fraction likewise validated only at fit.
    est2 = BaggedCompositeEstimator(
        base_estimator=CompositeTargetEstimator(base_estimator=Ridge(), base_column="base", transform_name="diff"),
        n_estimators=2,
    )
    est2.set_params(trim_fraction=0.9)
    with pytest.raises(ValueError, match="trim_fraction"):
        est2.fit(X, y)


def test_bagging_init_does_not_validate():
    """__init__ stores params verbatim (sklearn contract): a bad aggregation/trim_fraction does NOT raise at construction."""
    est = BaggedCompositeEstimator(base_estimator=Ridge(), aggregation="bogus", trim_fraction=5.0)
    assert est.aggregation == "bogus"
    assert est.trim_fraction == 5.0


def test_quantile_wrapper_stores_alphas_verbatim_and_clone_roundtrips():
    """quantile_wrapper stores alphas verbatim (no tuple() transform) so clone / get_params round-trip a list unchanged."""
    alphas_list = [0.1, 0.5, 0.9]
    est = _QuantileMultiOutputWrapper(base_estimator=Ridge(), alphas=alphas_list)
    assert est.alphas == alphas_list
    assert isinstance(est.alphas, list)
    assert est.get_params()["alphas"] == alphas_list
    cloned = clone(est)
    assert cloned.alphas == alphas_list
    assert isinstance(cloned.alphas, list)


def test_extremes_passes_clone_not_prototype_to_inner():
    """extremes must pass a CLONE of base_estimator into the inner body composite, not the caller's prototype object.

    Pins the prototype-isolation fix at the TailCompositeEstimator level (independent of whether the inner happens to
    clone again): the body composite's stored base_estimator must be a DISTINCT object from the prototype, and the
    prototype must stay unfitted after fit.
    """
    rng = np.random.default_rng(3)
    X = _make_X(rng)
    y = _make_y(X, rng)
    proto = Ridge()
    est = TailCompositeEstimator(base_estimator=proto, base_column="base", transform_name="diff")
    est.fit(X, y)
    assert est.body_estimator_.base_estimator is not proto, "TailCompositeEstimator passed the prototype, not a clone"
    assert not hasattr(proto, "coef_"), "base_estimator prototype was mutated (fitted) by TailCompositeEstimator.fit"
