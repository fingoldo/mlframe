"""Regression tests for cached pre_pipeline state transfer (TC33).

A whole-transfer failure (cached fitted object exposes no ``__dict__``) must NOT be
swallowed at debug level: it leaves ``pre_pipeline`` with no fitted state, so the
caller's predict-time ``transform`` raises NotFittedError. The failure must be
surfaced at WARNING so the operator knows the state transfer was incomplete.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.pipeline import _pipeline_helpers as H
from mlframe.training.pipeline._pipeline_cache import (
    _PRE_PIPELINE_CACHE,
    _PRE_PIPELINE_CACHE_LOCK,
    _pre_pipeline_cache_key,
    _pre_pipeline_cache_clear,
)


class _SlotsOnlyFitted:
    """Stand-in cached fitted object with NO ``__dict__`` (``__slots__``-only).

    Iterating ``self.__dict__.items()`` in the transfer loop raises AttributeError,
    exercising the whole-transfer-failure branch.
    """

    __slots__ = ("dummy",)

    def __init__(self):
        self.dummy = 1


def _build_inputs():
    train_df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    val_df = pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0)})
    pre_pipeline = Pipeline([("sc", StandardScaler())])  # unfitted
    target = np.arange(10)
    return train_df, val_df, pre_pipeline, target


def test_state_transfer_failure_warns_not_silent(caplog):
    _pre_pipeline_cache_clear()
    train_df, val_df, pre_pipeline, target = _build_inputs()

    key = _pre_pipeline_cache_key(
        train_df, val_df, pre_pipeline, train_target=target, target_name="t"
    )
    # 3-tuple cache entry whose fitted object cannot be iterated for state transfer.
    with _PRE_PIPELINE_CACHE_LOCK:
        _PRE_PIPELINE_CACHE[key] = (train_df, val_df, _SlotsOnlyFitted())

    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
            H._apply_pre_pipeline_transforms(
                model=object(),
                pre_pipeline=pre_pipeline,
                train_df=train_df,
                val_df=val_df,
                train_target=target,
                skip_pre_pipeline_transform=False,
                skip_preprocessing=False,
                use_cache=True,
                model_file_name="m",
                verbose=0,
                target_name="t",
            )
    finally:
        _pre_pipeline_cache_clear()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("state transfer FAILED" in r.getMessage() for r in warnings), (
        "incomplete cache state transfer must surface at WARNING, not be hidden at debug"
    )
