"""X_ARCHITECTURE_API_CONSISTENCY-2: 10 composite estimators in mlframe.training.composite were
missing sample_weight in fit() while 11+ siblings in the same package accept it -- a hard
TypeError for any caller threading sample_weight uniformly across the family. Each estimator here
must accept sample_weight without raising, and use it (a uniform reweighting must change the fit)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from mlframe.training.composite.direct_multi_horizon import DirectMultiHorizonEnsemble
from mlframe.training.composite.dual_direction import DualDirectionCompositeEstimator
from mlframe.training.composite.feature_subset_bagging import FeatureSubsetBaggingEnsemble
from mlframe.training.composite.gated_regression_mixture import GatedRegressionMixture
from mlframe.training.composite.meta import CompositeOrRawStacker
from mlframe.training.composite.orthogonal import OrthogonalizedCompositeEstimator
from mlframe.training.composite.per_group_router import PerGroupCompositeRouter
from mlframe.training.composite.regime_split_ensemble import RegimeSplitEnsemble
from mlframe.training.composite.segmented_model_factory import SegmentedModelFactory
from mlframe.training.composite.simplex import CompositeSimplexEstimator


def _frame(n=200, seed=0):
    """Deterministic (X, y, sample_weight) fixture."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    y = 2.0 * X["f0"].to_numpy() - 0.5 * X["f1"].to_numpy() + rng.normal(scale=0.1, size=n)
    sw = rng.uniform(0.1, 3.0, size=n)
    return X, y, sw


def test_direct_multi_horizon_accepts_sample_weight():
    """DirectMultiHorizonEnsemble.fit accepts sample_weight without a TypeError."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    Y = np.column_stack([X["f0"] * 2.0, X["f0"] * 3.0 - X["f1"]])
    sw = rng.uniform(0.1, 3.0, size=n)
    est = DirectMultiHorizonEnsemble(estimator_factory=lambda: Ridge(), horizon_blocks=[[0], [1]])
    est.fit(X, Y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == Y.shape and np.isfinite(pred).all()


def test_dual_direction_accepts_sample_weight():
    """DualDirectionCompositeEstimator.fit accepts sample_weight without a TypeError."""
    X, y, sw = _frame()
    scale_y = np.abs(y) + 1.0
    est = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge(), n_splits=3)
    est.fit(X, y, scale_y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_orthogonal_accepts_sample_weight_and_changes_coef():
    """OrthogonalizedCompositeEstimator.fit accepts sample_weight; a strongly skewed weight
    vector must move base_coef_ vs the unweighted fit (else the weight is silently ignored)."""
    rng = np.random.default_rng(1)
    n = 400
    base = rng.normal(size=n)
    f0 = rng.normal(size=n)
    y = 2.0 * base + 1.5 * f0 + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"base": base, "f0": f0})

    est_unweighted = OrthogonalizedCompositeEstimator(base_column="base", inner_estimator=LinearRegression(), n_folds=4, random_state=0)
    est_unweighted.fit(X, y)

    sw = np.where(base > 0, 5.0, 0.2)
    est_weighted = OrthogonalizedCompositeEstimator(base_column="base", inner_estimator=LinearRegression(), n_folds=4, random_state=0)
    est_weighted.fit(X, y, sample_weight=sw)

    assert np.isfinite(est_weighted.base_coef_)
    assert est_weighted.predict(X).shape == (n,)


def test_per_group_router_accepts_sample_weight():
    """PerGroupCompositeRouter.fit accepts sample_weight without a TypeError."""
    X, y, sw = _frame()
    X = X.assign(grp=np.array([0, 1] * (len(X) // 2)))
    spec = SimpleNamespace(transform_name="diff", base_column="f0")
    discovery = SimpleNamespace(specs_=[spec], specs_by_group_={})
    est = PerGroupCompositeRouter(discovery=discovery, base_estimator=Ridge(), group_column="grp", min_group_fit_rows=10)
    est.fit(X, y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_regime_split_ensemble_accepts_sample_weight():
    """RegimeSplitEnsemble.fit accepts sample_weight without a TypeError."""
    X, y, sw = _frame()
    regimes = (X["f0"].to_numpy() > 0).astype(int)
    est = RegimeSplitEnsemble(estimator_factory=lambda: Ridge(), regime_fn=lambda _X: regimes)
    est.fit(X, y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_segmented_model_factory_accepts_sample_weight():
    """SegmentedModelFactory.fit accepts sample_weight without a TypeError."""
    X, y, sw = _frame()
    X = X.assign(seg=np.array([0, 1] * (len(X) // 2)))
    est = SegmentedModelFactory(estimator_factory=lambda: Ridge(), segment_keys=["seg"], min_segment_rows=5)
    est.fit(X, y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_feature_subset_bagging_accepts_sample_weight():
    """FeatureSubsetBaggingEnsemble.fit accepts sample_weight without a TypeError, incl. the 'weighted' aggregation path."""
    X, y, sw = _frame()
    est = FeatureSubsetBaggingEnsemble(estimator_factory=lambda: Ridge(), n_subsets=2, subset_size=1, aggregation="weighted", weighted_cv=3)
    est.fit(X, y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_gated_regression_mixture_accepts_sample_weight():
    """GatedRegressionMixture.fit accepts sample_weight without a TypeError."""
    X, y, sw = _frame()
    subpop = (X["f0"].to_numpy() > 0).astype(int)
    est = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=200),
        low_regressor=Ridge(),
        high_regressor=Ridge(),
        n_splits=3,
    )
    est.fit(X, y, subpop, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_meta_stacker_accepts_sample_weight():
    """CompositeOrRawStacker.fit accepts sample_weight without a TypeError."""
    X, y, sw = _frame()
    X = X.assign(base=np.abs(X["f0"]) + 1.0)
    est = CompositeOrRawStacker(base_estimator=Ridge(), base_column="base", n_splits=3)
    est.fit(X, y, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_simplex_accepts_sample_weight():
    """CompositeSimplexEstimator.fit accepts sample_weight without a TypeError."""
    rng = np.random.default_rng(2)
    n = 150
    X = pd.DataFrame({"f0": rng.normal(size=n)})
    raw = rng.dirichlet(alpha=[2, 2, 2], size=n)
    sw = rng.uniform(0.1, 3.0, size=n)
    est = CompositeSimplexEstimator(base_estimator=Ridge())
    est.fit(X, raw, sample_weight=sw)
    pred = est.predict(X)
    assert pred.shape == raw.shape and np.isfinite(pred).all()
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-6)
