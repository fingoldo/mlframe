"""Unit tests for ``BaggedCompositeEstimator`` (composite/bagging.py).

Cover: member count after fit, predict shape, predict_std non-negativity +
exact-zero on identical members, sklearn clone, and determinism under a fixed
seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.bagging import BaggedCompositeEstimator


def _make_composite_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.normal(5.0, 1.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    y = base + 0.5 * feat + rng.normal(0.0, 0.3, size=n)
    X = pd.DataFrame({"lag": base, "feat": feat})
    return X, y


def _make_composite_proto():
    return CompositeTargetEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=4, random_state=0),
        transform_name="diff",
        base_column="lag",
    )


def test_fits_n_members():
    X, y = _make_composite_data()
    bag = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=7, random_state=1,
    ).fit(X, y)
    assert len(bag.estimators_) == 7
    # Prototype stays unfitted (cloned per member).
    assert not hasattr(bag.base_estimator, "estimator_")


def test_predict_shape_and_std_nonneg():
    X, y = _make_composite_data()
    bag = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=5, random_state=2,
    ).fit(X, y)
    pred = bag.predict(X)
    std = bag.predict_std(X)
    assert pred.shape == (len(y),)
    assert std.shape == (len(y),)
    assert np.all(std >= 0.0)


def test_std_zero_for_identical_members():
    # Deterministic linear inner + no bootstrap + no seed variation -> every
    # member is identical -> spread is exactly 0.
    X, y = _make_composite_data()
    proto = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="lag",
    )
    bag = BaggedCompositeEstimator(
        base_estimator=proto,
        n_estimators=4,
        bootstrap=False,
        vary_inner_random_state=False,
        random_state=3,
    ).fit(X, y)
    std = bag.predict_std(X)
    assert np.allclose(std, 0.0, atol=1e-9)


def test_predict_interval_epistemic_brackets_mean():
    X, y = _make_composite_data()
    bag = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=6, random_state=4,
    ).fit(X, y)
    mean = bag.predict(X)
    lo, hi = bag.predict_interval_epistemic(X, z=1.96)
    assert np.all(lo <= mean + 1e-9)
    assert np.all(hi >= mean - 1e-9)
    assert np.all(hi >= lo)


def test_clone_roundtrip():
    proto = _make_composite_proto()
    bag = BaggedCompositeEstimator(
        base_estimator=proto, n_estimators=5, random_state=9, max_samples=0.8,
    )
    cloned = clone(bag)
    assert cloned.n_estimators == 5
    assert cloned.random_state == 9
    assert cloned.max_samples == 0.8
    assert not hasattr(cloned, "estimators_")  # unfitted shell


def test_aggregation_param_defaults_and_validation():
    bag = BaggedCompositeEstimator(base_estimator=_make_composite_proto())
    assert bag.aggregation == "trimmed_mean"
    assert bag.trim_fraction == 0.2
    cloned = clone(BaggedCompositeEstimator(base_estimator=_make_composite_proto(), aggregation="median", trim_fraction=0.15))
    assert cloned.aggregation == "median"
    assert cloned.trim_fraction == 0.15
    # sklearn contract: __init__ stores params verbatim; aggregation / trim_fraction are validated at fit, not construction.
    X, y = _make_composite_data()
    with pytest.raises(ValueError):
        BaggedCompositeEstimator(base_estimator=_make_composite_proto(), aggregation="bogus").fit(X, y)
    with pytest.raises(ValueError):
        BaggedCompositeEstimator(base_estimator=_make_composite_proto(), trim_fraction=0.5).fit(X, y)


def test_pickle_replay_legacy_model_aggregates_by_mean():
    import pickle

    X, y = _make_composite_data()
    bag = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=6, random_state=3,
    ).fit(X, y)
    replayed = pickle.loads(pickle.dumps(bag))
    # Simulate a pre-flip pickle: the aggregation attrs did not exist; predict must fall back to the legacy plain mean.
    del replayed.aggregation, replayed.trim_fraction
    members = replayed._member_predictions(X)
    assert np.array_equal(replayed.predict(X), members.mean(axis=0))


def test_aggregation_variants_match_their_definitions():
    X, y = _make_composite_data()
    bag = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=12, random_state=1, aggregation="median",
    ).fit(X, y)
    members = bag._member_predictions(X)
    assert np.array_equal(bag.predict(X), np.median(members, axis=0))
    bag_mean = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=12, random_state=1, aggregation="mean",
    ).fit(X, y)
    assert np.array_equal(bag_mean.predict(X), bag_mean._member_predictions(X).mean(axis=0))


def test_deterministic_with_fixed_seed():
    X, y = _make_composite_data()
    p1 = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=6, random_state=42,
    ).fit(X, y).predict(X)
    p2 = BaggedCompositeEstimator(
        base_estimator=_make_composite_proto(), n_estimators=6, random_state=42,
    ).fit(X, y).predict(X)
    assert np.array_equal(p1, p2)


def test_fit_validation_errors():
    X, y = _make_composite_data(n=50)
    with pytest.raises(ValueError):
        BaggedCompositeEstimator(base_estimator=None).fit(X, y)
    with pytest.raises(ValueError):
        BaggedCompositeEstimator(
            base_estimator=_make_composite_proto(), n_estimators=0,
        ).fit(X, y)
    with pytest.raises(ValueError):
        BaggedCompositeEstimator(
            base_estimator=_make_composite_proto(), max_samples=1.5,
        ).fit(X, y)


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    bag = BaggedCompositeEstimator(base_estimator=_make_composite_proto())
    with pytest.raises(NotFittedError):
        bag.predict(np.zeros((3, 2)))
