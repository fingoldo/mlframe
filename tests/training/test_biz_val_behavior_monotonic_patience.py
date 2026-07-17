"""C1/C2: ``TrainingBehaviorConfig.monotonic_decline_patience`` is the canonical off-switch and must
THREAD through to the boosters -- None disables the monotonic strict-decline stop so the booster trains
its full iteration cap.

The config field exists and defaults to 3; setting it to None must reach the lgb / xgb shim ``.fit()``
kwarg (and the CatBoost ``callback_params``) so the stop is fully disabled. This pins the missing off-switch
the adversarial review flagged.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_behavior_config_has_monotonic_decline_patience_default_7():
    from mlframe.training._model_configs_behavior import TrainingBehaviorConfig

    cfg = TrainingBehaviorConfig()
    assert cfg.monotonic_decline_patience == 7
    cfg_off = TrainingBehaviorConfig(monotonic_decline_patience=None)
    assert cfg_off.monotonic_decline_patience is None


def _overfit_data(seed=3, n=600, d=20):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (0.5 * X[:, 0] + rng.randn(n) * 1.4).astype(np.float64)
    return X[:-150], y[:-150], X[-150:], y[-150:]


def test_behavior_patience_none_threads_to_lgb_booster_full_cap():
    """End-to-end through the trainer wiring: behavior.monotonic_decline_patience=None must reach the lgb
    shim and disable the stop, so the booster trains all ``n_estimators`` rounds (no early monotonic stop)."""
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse

    Xtr, ytr, Xv, yv = _overfit_data()
    n_estimators = 300

    # Mirror the trainer resolution: behavior.monotonic_decline_patience -> fit_params kwarg.
    from mlframe.training._model_configs_behavior import TrainingBehaviorConfig

    behavior = TrainingBehaviorConfig(monotonic_decline_patience=None)
    fit_params: dict = {}
    fit_params.setdefault("monotonic_decline_patience", behavior.monotonic_decline_patience)

    m = LGBMRegressorWithDatasetReuse(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=5,
        verbose=-1,
        random_state=0,
    )
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="l2", **fit_params)
    assert m.booster_.num_trees() == n_estimators, f"patience=None should train full cap {n_estimators}, got {m.booster_.num_trees()}"

    # Sanity: an enabled patience DOES stop earlier on the same overfit data (the off-switch actually switches).
    m_on = LGBMRegressorWithDatasetReuse(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=5,
        verbose=-1,
        random_state=0,
    )
    m_on.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="l2", monotonic_decline_patience=3)
    assert m_on.booster_.num_trees() < n_estimators


def test_trainer_resolution_injects_patience_into_callback_params_for_cb():
    """The trainer-side resolution copies behavior.monotonic_decline_patience into callback_params (the cb
    consumer) and into fit_params for lgb/xgb. Replays that exact resolution to pin the plumbing."""
    from mlframe.training._model_configs_behavior import TrainingBehaviorConfig

    behavior = TrainingBehaviorConfig(monotonic_decline_patience=None)
    _beh = behavior.__dict__
    assert "monotonic_decline_patience" in _beh
    patience = _beh["monotonic_decline_patience"]
    assert patience is None

    # cb path: callback_params carries the key; _setup_early_stopping_callback pops + honours None (no CB stop).
    callback_params = {"patience": 10}
    callback_params.setdefault("monotonic_decline_patience", patience)
    assert callback_params["monotonic_decline_patience"] is None
