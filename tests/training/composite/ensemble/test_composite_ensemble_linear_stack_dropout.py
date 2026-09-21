"""Regression: linear_stack predict on component dropout is a deterministic no-refit combine that keeps the prediction's level.

When a component returns None / raises at predict time, predict rebuilds the full design with that component's own OOF column
mean and applies the ORIGINAL Ridge coefficients plus the original intercept -- no solver refit. Keeping only the surviving
columns instead dropped that component's whole contribution, so a stack whose weights sum to ~1 served a fraction of the
forecast. Filling the column keeps predict a pure deterministic function of the inputs (no batch-order dependence) and lets the
multi-GB train-matrix stash stay out of every pickle.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np

from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble


def _build_stack_with_three_components(seed: int = 0):
    """Builds a linear-stack ensemble from 3 mock components whose predictions are y_train plus a per-component bias and noise."""
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
    """When every stacked component predicts successfully, no dropout warning is logged."""
    ens, models, _y_train = _build_stack_with_three_components(seed=0)
    # All components produce the same training-time mean pattern.
    test_y = np.array([10.0, 11.0, 9.0])
    for i, m in enumerate(models):
        m.predict.return_value = test_y + (i - 1) * 0.1  # tiny shift
    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.ensemble"):
        preds = ens.predict("X_dummy")
    assert preds.shape == (3,)
    assert all("dropped out" not in rec.message for rec in caplog.records)


def test_linear_stack_one_component_dropped_no_refit_warns(caplog):
    """When one component returns None at predict, predict must:
    (a) emit a 'dropped out' warning, and
    (b) apply the ORIGINAL coefficients to a design whose missing column is that component's OOF mean -- no solver refit,
        fully deterministic, and without the level shift that dropping the column outright caused.
    """
    ens, models, _y_train = _build_stack_with_three_components(seed=1)

    test_y = np.array([10.0, 11.0, 9.0, 12.0])
    p0_test = test_y + 1.0
    p2_test = test_y + 2.0
    models[0].predict.return_value = p0_test
    models[1].predict.side_effect = RuntimeError("simulated component failure")
    models[2].predict.return_value = p2_test

    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.ensemble"):
        preds = ens.predict("X_dummy")

    assert any("dropped out" in rec.message for rec in caplog.records), f"expected a 'dropped out' warning; got {[r.message for r in caplog.records]}"

    full_w = np.asarray(ens.weights, dtype=np.float64)
    intercept = float(ens._linear_stack_intercept)
    # From the stashed OOF design, so the expectation is independent of where the deployed code keeps its column means.
    col_mean = float(np.asarray(ens._linear_stack_train_preds, dtype=np.float64)[:, 1].mean())
    expected = full_w[0] * p0_test + full_w[1] * col_mean + full_w[2] * p2_test + intercept
    np.testing.assert_allclose(preds, expected, rtol=1e-9, atol=1e-9)

    # The level survives the dropout: predictions stay around the target, instead of shrinking by the dropped weight.
    assert np.all(np.abs(preds - test_y) < 2.0), f"dropout moved the prediction off the target level: {preds} vs {test_y}"

    # Determinism: repeated predict gives identical output.
    np.testing.assert_array_equal(preds, ens.predict("X_dummy"))


def test_linear_stack_dropout_is_deterministic_combine():
    """No-refit dropout is a deterministic combine over the mean-filled design.

    The deployed policy keeps the original Ridge coefficients and the original intercept, standing the dropped component's
    OOF mean in for its column. It is NOT a fresh refit, so it does not perfectly reconstruct y when a load-bearing component
    drops -- but it is a pure deterministic function of the inputs (the property that matters for batched serving) and it
    does not shrink the prediction by the dropped component's weight.
    """
    rng = np.random.default_rng(42)
    n = 600
    y_train = rng.normal(loc=100.0, scale=5.0, size=n)
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

    test_y = np.array([100.0, 105.0, 95.0])
    models[0].predict.return_value = test_y - 20.0
    models[1].predict.side_effect = ValueError("dropout")
    models[2].predict.return_value = test_y - 20.0

    preds = ens.predict("X_dummy")
    assert preds.shape == test_y.shape

    full_w = np.asarray(ens.weights, dtype=np.float64)
    intercept = float(ens._linear_stack_intercept)
    # From the stashed OOF design, so the expectation is independent of where the deployed code keeps its column means.
    col_mean = float(np.asarray(ens._linear_stack_train_preds, dtype=np.float64)[:, 1].mean())
    expected = full_w[0] * (test_y - 20.0) + full_w[1] * col_mean + full_w[2] * (test_y - 20.0) + intercept
    np.testing.assert_allclose(preds, expected, rtol=1e-9, atol=1e-9)
    # Three near-identical components: losing one must not move the served level by ~a third.
    assert np.all(np.abs(preds - test_y) < 5.0), f"dropout shifted the level: {preds} vs {test_y}"
    np.testing.assert_array_equal(preds, ens.predict("X_dummy"))
