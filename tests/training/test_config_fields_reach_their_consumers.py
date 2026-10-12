"""Regression tests for config values that were accepted but never reached, or crashed, their consumer.

* ``TrainingConfig.metamodel_func`` was dropped: model construction reads the wrapper only from ``TrainingBehaviorConfig.metamodel_func``.
* ``ModelHyperparamsConfig.def_regr_metric="Huber"`` is mapped for LGB / XGB but reached CatBoost bare, which rejects it without a delta.
* ``FeatureTypesConfig.datetime_methods`` documented ``ordinal_day`` as the portable day-of-year name, yet the pandas branch raised on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.configs import TrainingBehaviorConfig
from mlframe.training.core._setup_helpers import behavior_with_training_config_metamodel


def _wrap(model):
    """Stand-in metamodel wrapper."""
    return model


def _other(model):
    """A second, distinct wrapper."""
    return model


@pytest.mark.parametrize("behavior", [None, TrainingBehaviorConfig(), {}, {"metamodel_func": None}])
def test_training_config_metamodel_func_reaches_behavior_without_a_wrapper(behavior):
    """With no behavior-level wrapper, the carrier's wrapper becomes the effective one, whatever shape the behavior config has."""
    out = behavior_with_training_config_metamodel(behavior, _wrap)
    got = out["metamodel_func"] if isinstance(out, dict) else out.metamodel_func
    assert got is _wrap


@pytest.mark.parametrize("behavior", [TrainingBehaviorConfig(metamodel_func=_other), {"metamodel_func": _other}])
def test_explicit_behavior_metamodel_func_wins(behavior):
    """An explicitly set behavior-level wrapper is never overridden by the carrier's."""
    out = behavior_with_training_config_metamodel(behavior, _wrap)
    got = out["metamodel_func"] if isinstance(out, dict) else out.metamodel_func
    assert got is _other


def test_no_training_config_wrapper_leaves_behavior_untouched():
    """``metamodel_func=None`` on the carrier is a no-op (same object back)."""
    behavior = TrainingBehaviorConfig()
    assert behavior_with_training_config_metamodel(behavior, None) is behavior


def test_huber_def_regr_metric_gives_catboost_a_fittable_eval_metric():
    """``def_regr_metric="Huber"`` reaches CatBoost as ``Huber:delta=1.0`` and the regressor fits with it."""
    from catboost import CatBoostRegressor

    from mlframe.training._helpers_training_configs import get_training_configs

    cfg = get_training_configs(def_regr_metric="Huber", iterations=5, has_gpu=False)
    params = cfg.CB_REGR
    assert params["eval_metric"] == "Huber:delta=1.0"
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = X[:, 0] + rng.standard_normal(80) * 0.1
    CatBoostRegressor(iterations=5, verbose=0, eval_metric=params["eval_metric"], allow_writing_files=False).fit(X, y, eval_set=(X, y))


@pytest.mark.parametrize("method", ["ordinal_day", "dayofyear", "day_of_year"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_every_day_of_year_spelling_resolves_on_both_backends(method, backend):
    """The pandas-native, polars-native and logical day-of-year names all produce the same values on either frame type."""
    from mlframe.feature_engineering.basic import create_date_features

    ts = pd.date_range("2026-12-30", periods=4, freq="D")
    df = pd.DataFrame({"ts": ts}) if backend == "pandas" else pl.DataFrame({"ts": ts.to_pydatetime().tolist()})
    out = create_date_features(df, cols=["ts"], delete_original_cols=True, methods={method: np.int16})
    assert np.asarray(out[f"ts_{method}"]).tolist() == [364, 365, 1, 2]
