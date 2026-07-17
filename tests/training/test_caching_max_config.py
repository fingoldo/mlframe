"""Regression tests for ``TrainingBehaviorConfig.pre_pipeline_cache_max`` (P1).

Pre-fix the cache cap was a hard-coded ``_PRE_PIPELINE_CACHE_MAX = 2``
constant in ``_pipeline_helpers.py``. Post-fix it lives on the public
``TrainingBehaviorConfig`` so long-running services with bigger model
rosters can resize without monkey-patching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from mlframe.training.pipeline._pipeline_helpers import (
    _PRE_PIPELINE_CACHE,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_set,
)
from mlframe.training.configs import TrainingBehaviorConfig


def test_training_behavior_has_pre_pipeline_cache_max_field():
    cfg = TrainingBehaviorConfig()
    assert hasattr(cfg, "pre_pipeline_cache_max")
    assert cfg.pre_pipeline_cache_max == 4


def test_training_behavior_pre_pipeline_cache_max_is_overridable():
    cfg = TrainingBehaviorConfig(pre_pipeline_cache_max=8)
    assert cfg.pre_pipeline_cache_max == 8


def test_cache_max_override_propagates_to_pre_pipeline_cache():
    """Pass the config value through to ``_pre_pipeline_cache_set``."""
    _pre_pipeline_cache_clear()
    cfg = TrainingBehaviorConfig(pre_pipeline_cache_max=5)
    pipe = Pipeline(steps=[("imp", SimpleImputer())])
    rng = np.random.default_rng(7)
    for i in range(10):
        X = pd.DataFrame(rng.normal(size=(4, 2)), columns=["a", "b"])
        y = pd.Series(rng.normal(size=4), name=f"t{i}")
        _pre_pipeline_cache_set(
            X,
            None,
            pipe,
            f"t_{i}",
            f"v_{i}",
            train_target=y.to_numpy(),
            target_name=f"t{i}",
            cache_max=cfg.pre_pipeline_cache_max,
        )
    assert len(_PRE_PIPELINE_CACHE) == 5
