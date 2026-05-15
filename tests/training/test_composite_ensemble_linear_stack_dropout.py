"""Regression: linear_stack predict must not be biased when a component
returns None at predict time.

Pre-fix: dropped component's WEIGHT was removed but the original Ridge
intercept (fit assuming ALL components) was still added in full. The
resulting prediction was off by O(intercept) -- a constant bias term.

Post-fix: predict refits Ridge on the surviving subset using the stashed
training matrix; the new intercept is consistent with the surviving
columns.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np

from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble


def _build_stack_with_three_components(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 500
    # Three "components" producing predictions correlated with a target.
    y_train = rng.normal(loc=10.0, scale=2.0, size=n)
    # Each component's predictions are y_train + bias + small noise so
    # Ridge picks a non-trivial intercept.
    p1 = y_train + 1.0 + rng.normal(scale=0.5, size=n)
    p2 = y_train - 0.5 + rng.normal(scale=0.5, size=n)
    p3 = y_train + 2.0 + rng.normal(scale=0.5, size=n)
    component_preds = np.column_stack([p1, p2, p3])

    # Component models are mocks; we'll override .predict per-test.
    models = [MagicMock(name=f"c{i}") for i in range(3)]
    names = [f"c{i}" for i in range(3)]
    ens = CompositeCrossTargetEnsemble.from_linear_stack(
        component_models=models,
        component_names=names,
        component_predictions=component_preds,
        y_train=y_train,
        ridge_alpha=1.0,
    )
    return ens, models, y_train


def test_linear_stack_all_components_ok_no_warning(caplog):
    ens, models, y_train = _build_stack_with_three_components(seed=0)
    # All components produce the same training-time mean pattern.
    test_y = np.array([10.0, 11.0, 9.0])
    for i, m in enumerate(models):
        m.predict.return_value = test_y + (i - 1) * 0.1  # tiny shift
    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite_ensemble"):
        preds = ens.predict("X_dummy")
    assert preds.shape == (3,)
    assert all("dropped out" not in rec.message for rec in caplog.records)


def test_linear_stack_one_component_dropped_refits_and_warns(caplog):
    """When one component returns None at predict, predict must:
       (a) emit a refit-warning, and
       (b) return predictions consistent with a fresh Ridge fit on the
           surviving subset (i.e. NOT include the original intercept,
           which was calibrated against all 3 components).
    """
    ens, models, y_train = _build_stack_with_three_components(seed=1)

    # Pre-fix the original Ridge intercept gets ADDED at full value even after
    # one component drops. Compare to a manually-refitted Ridge on the
    # surviving 2 columns: this is the correct answer.
    test_y = np.array([10.0, 11.0, 9.0, 12.0])
    p0_test = test_y + 1.0
    p2_test = test_y + 2.0
    models[0].predict.return_value = p0_test
    models[1].predict.side_effect = RuntimeError("simulated component failure")
    models[2].predict.return_value = p2_test

    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite_ensemble"):
        preds = ens.predict("X_dummy")

    assert any("dropped out" in rec.message for rec in caplog.records), (
        f"expected a 'dropped out' refit-warning; got {[r.message for r in caplog.records]}"
    )

    # Compute the EXPECTED prediction: Ridge on (p1, p3) only, fit on the
    # same stashed training matrix.
    from sklearn.linear_model import Ridge

    train_preds = ens._linear_stack_train_preds
    train_y = ens._linear_stack_train_y
    refit = Ridge(alpha=1.0, fit_intercept=True)
    refit.fit(train_preds[:, [0, 2]], train_y)
    expected = refit.predict(np.column_stack([p0_test, p2_test]))
    np.testing.assert_allclose(preds, expected, rtol=1e-9, atol=1e-9)


def test_linear_stack_dropout_not_biased_by_original_intercept():
    """Sanity check on the bias direction.

    Build a stack where Ridge will land on a SIZEABLE intercept by
    biasing all three component predictions away from y. Then drop
    one component at predict time. Pre-fix predictions get the FULL
    original intercept added -- post-fix they get the refitted intercept
    appropriate for the surviving subset.
    """
    rng = np.random.default_rng(42)
    n = 600
    y_train = rng.normal(loc=100.0, scale=5.0, size=n)
    # Components ALL underestimate y by ~20 -- Ridge picks intercept ~+20.
    p1 = y_train - 20.0 + rng.normal(scale=0.5, size=n)
    p2 = y_train - 20.0 + rng.normal(scale=0.5, size=n)
    p3 = y_train - 20.0 + rng.normal(scale=0.5, size=n)
    component_preds = np.column_stack([p1, p2, p3])

    models = [MagicMock(name=f"c{i}") for i in range(3)]
    names = [f"c{i}" for i in range(3)]
    ens = CompositeCrossTargetEnsemble.from_linear_stack(
        component_models=models,
        component_names=names,
        component_predictions=component_preds,
        y_train=y_train,
        ridge_alpha=1.0,
    )

    # All three give shifted predictions at predict time; drop component 1.
    test_y = np.array([100.0, 105.0, 95.0])
    models[0].predict.return_value = test_y - 20.0
    models[1].predict.side_effect = ValueError("dropout")
    models[2].predict.return_value = test_y - 20.0

    preds = ens.predict("X_dummy")
    # Post-fix predictions should be close to the actual test_y (Ridge
    # refit on 2 components still picks the right intercept) -- well
    # within +/- 3 of test_y. Pre-fix the bias would push them off by an
    # extra Ridge-coefficient-weighting amount tied to the 3-component
    # intercept.
    assert preds.shape == test_y.shape
    # Sanity range: residuals should be small (the components' noise is
    # 0.5 stdev and Ridge averages two of them).
    assert np.max(np.abs(preds - test_y)) < 5.0, (
        f"preds {preds} too far from truth {test_y}; intercept-bias regression?"
    )
