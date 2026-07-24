"""Regression tests for a `store_params_in_object()` default-postfix mismatch discovered while
running the wave-9 baseline-debt regression suite: `store_params_in_object`'s default
``postfix="_param_"`` sets e.g. ``self.random_state_param_`` while every reader in these three
classes reads the bare ``self.random_state`` -- so the constructor parameter was silently
invisible to the rest of the class (195 downstream test failures in the neural suite alone, all
tracing back to ``PytorchLightningEstimator.random_state``). Fixed by passing ``postfix=""`` at
each call site, matching the precedent already established in ``AggregatingValidationCallback``
and ``mlframe.calibration.post``.
"""

from __future__ import annotations


def test_pytorch_lightning_estimator_random_state_is_a_direct_attribute():
    """`PytorchLightningRegressor(random_state=...)` must expose `self.random_state` directly
    (not `self.random_state_param_`) since `_base_fit.py` reads it unprefixed."""
    from mlframe.training.neural.base import PytorchLightningRegressor

    est = PytorchLightningRegressor(
        model_class=object,
        model_params={},
        network_params={},
        datamodule_class=object,
        datamodule_params={},
        trainer_params={},
        random_state=42,
    )
    assert est.random_state == 42
    assert not hasattr(est, "random_state_param_")
    assert est.get_params()["random_state"] == 42


def test_early_stopping_wrapper_random_state_is_a_direct_attribute():
    """`EarlyStoppingWrapper`'s constructor params must be direct attributes:
    `_resolve_scoring`/`fit`/etc. read `self.patience`/`self.random_state` unprefixed."""
    from mlframe.estimators.early_stopping import EarlyStoppingWrapper
    from sklearn.linear_model import LinearRegression

    est = EarlyStoppingWrapper(base_model=LinearRegression(), patience=3, random_state=7)
    assert est.patience == 3
    assert est.random_state == 7
    assert not hasattr(est, "patience_param_")


def test_ice_metric_params_are_direct_attributes():
    """`ICE`'s constructor params must be direct attributes: `is_max_optimal()` reads
    `self.higher_is_better` unprefixed."""
    from mlframe.metrics._ice_metric import ICE

    metric = ICE(metric=lambda y_true, y_pred: 0.0, higher_is_better=True, max_arr_size=100)
    assert metric.higher_is_better is True
    assert metric.max_arr_size == 100
    assert not hasattr(metric, "higher_is_better_param_")
    assert metric.is_max_optimal() is True
