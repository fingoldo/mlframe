"""Zero-inflated targets that are not textbook: a few rows below the atom, a filled constant at the maximum.

A production ``total_charge`` was 74% zeros with a few refunds at -2.17. The hurdle check required 0 to be the exact
minimum, declined silently, and the drift report's p01=0 hid the negatives. The same run filled "not hired" with 5 and
400 in other targets, which put the point mass at the target's MAXIMUM.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.composite._hurdle_dispatch import (
    BELOW_ATOM_TOLERANCE,
    zero_inflated_regression_targets,
    zero_inflation_verdict,
)
from mlframe.training.composite.hurdle import HurdleRegressor


def _charges(n=20_000, zero_share=0.74, n_negative=0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < zero_share, 0.0, np.exp(rng.normal(4.0, 1.2, n)))
    y[:n_negative] = -2.17
    return y


def test_a_sliver_of_refunds_below_zero_still_gets_a_hurdle():
    atom, n_below, reason = zero_inflation_verdict(_charges(n_negative=5))
    assert atom == 0.0 and n_below == 5 and reason is None


def test_many_rows_below_the_atom_are_declined_with_the_reason():
    n = 20_000
    atom, n_below, reason = zero_inflation_verdict(_charges(n=n, n_negative=int(3 * BELOW_ATOM_TOLERANCE * n)))
    assert atom is None and n_below > 0
    assert "below it" in reason and "-2.17" in reason


def test_a_point_mass_at_the_maximum_is_named_as_a_likely_fill_value():
    """``fill_null(5)`` on a feedback score: 97% at 5, which is the maximum."""
    rng = np.random.default_rng(1)
    y = np.where(rng.random(10_000) < 0.97, 5.0, rng.integers(1, 5, 10_000).astype(float))
    atom, _, reason = zero_inflation_verdict(y)
    assert atom is None and "MAXIMUM" in reason and "filled in" in reason


def test_no_point_mass_means_no_verdict():
    assert zero_inflation_verdict(np.random.default_rng(2).normal(size=5_000)) == (None, 0, None)


def test_a_declined_target_is_logged_not_silently_skipped(caplog):
    from mlframe.training._configs_base import TargetTypes

    targets = {TargetTypes.REGRESSION: {"total_charge": _charges(n_negative=2_000)}}
    with caplog.at_level(logging.WARNING):
        assert zero_inflated_regression_targets(targets, None) == {}
    assert any("no HurdleRegressor for total_charge" in r.getMessage() for r in caplog.records)


def test_below_zero_no_event_keeps_the_log_scale():
    """With the refunds counted as events, one negative magnitude forced the whole magnitude model onto the raw scale."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4_000, 3))
    y = _charges(n=4_000, n_negative=3)
    as_event = HurdleRegressor(random_state=0).fit(X, y)
    as_no_event = HurdleRegressor(random_state=0, below_zero="no_event").fit(X, y)
    assert as_event.magnitude_target_ == "raw"
    assert as_no_event.magnitude_target_ == "log"
    assert np.isfinite(as_no_event.predict(X)).all()


def test_below_zero_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="below_zero"):
        HurdleRegressor(below_zero="clip").fit(np.zeros((10, 1)), np.r_[np.zeros(5), np.ones(5)])


def test_the_injected_hurdle_counts_below_atom_rows_as_no_event(caplog):
    from types import SimpleNamespace

    from mlframe.training._configs_base import TargetTypes
    from mlframe.training.composite import _hurdle_dispatch

    targets = {TargetTypes.REGRESSION: {"total_charge": _charges(n_negative=4)}}
    with caplog.at_level(logging.WARNING):
        models = _hurdle_dispatch.maybe_inject_hurdle_for_zero_inflated(
            SimpleNamespace(), {}, [], targets, None, SimpleNamespace(hurdle_for_zero_inflated=True),
        )
    (label, est), = models
    assert est.below_zero == "no_event"
    assert any("lie below the point mass" in r.getMessage() for r in caplog.records)


def test_a_hurdle_of_two_boosters_trains_as_a_tree_model():
    """It was sent through the linear strategy (scaling, one-hot) with a 'No registered strategy' warning."""
    from mlframe.training.strategies import _strategy_for_estimator
    from mlframe.training.strategies.hgb import HGBStrategy
    from mlframe.training.strategies.tree_cb import TreeModelStrategy

    lightgbm = pytest.importorskip("lightgbm")
    lgb_hurdle = HurdleRegressor(classifier=lightgbm.LGBMClassifier(), regressor=lightgbm.LGBMRegressor())
    assert type(_strategy_for_estimator(lgb_hurdle)) is TreeModelStrategy
    assert isinstance(_strategy_for_estimator(HurdleRegressor()), HGBStrategy)  # both halves default to HGB


def test_halves_that_disagree_still_fall_back_to_linear(caplog):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from mlframe.training.strategies import _strategy_for_estimator
    from mlframe.training.strategies.neural import LinearModelStrategy

    with caplog.at_level(logging.WARNING):
        strategy = _strategy_for_estimator(HurdleRegressor(classifier=LogisticRegression(), regressor=LinearRegression()))
    assert isinstance(strategy, LinearModelStrategy)
