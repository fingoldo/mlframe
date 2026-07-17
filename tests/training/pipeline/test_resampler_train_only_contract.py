"""Lock: the pre_pipeline slot is row-PRESERVING; a row-changing imblearn
sampler is rejected (not silently run), and val is always transform-only.

History: this test previously asserted that a row-DROPPING sampler "runs on
train only" and pinned the resulting 99-row ``out_train``. That codified a bug:
``_apply_pre_pipeline_transforms`` returns only ``(train_df, val_df)`` and never
resamples ``train_target`` / ``sample_weight``, so a row-changing train resampler
silently misaligns X and y at model fit. The driver now RAISES on any train
row-count change. This test pins the corrected contract:

1. a row-changing sampler in the pre_pipeline slot raises a clear ValueError, and
2. a genuine transform-only imblearn pre_pipeline leaves BOTH train and val row
   counts intact (val is never resampled -- no ES-detector / holdout leak).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

imblearn = pytest.importorskip("imblearn")
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

from mlframe.training.pipeline._pipeline_helpers import (
    _apply_pre_pipeline_transforms,
)


# Module-level call recorder so the FunctionSampler callables stay picklable/clonable.
_RESAMPLE_ROW_COUNTS: list[int] = []


def _drop_first_row(X, y):
    """A sampler that changes row count -- must be rejected by the driver."""
    _RESAMPLE_ROW_COUNTS.append(len(X))
    return X[1:], y[1:]


def _identity_sampler(X, y):
    """A transform-only sampler that preserves row count (no resampling)."""
    _RESAMPLE_ROW_COUNTS.append(len(X))
    return X, y


def _make_dfs():
    """Make dfs."""
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    val_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y_train = (rng.normal(size=100) > 0).astype(int)
    return train_df, val_df, y_train


def test_row_changing_resampler_in_pre_pipeline_is_rejected():
    """A row-dropping sampler in the FS slot raises (was: silent X/y desync)."""
    _RESAMPLE_ROW_COUNTS.clear()
    train_df, val_df, y_train = _make_dfs()
    pre_pipeline = ImbPipeline(
        [
            ("res", FunctionSampler(func=_drop_first_row, validate=False)),
            ("passthrough", FunctionTransformer(validate=False)),
        ]
    )
    with pytest.raises(ValueError, match=r"changed the train row count"):
        _apply_pre_pipeline_transforms(
            model=LinearRegression(),
            pre_pipeline=pre_pipeline,
            train_df=train_df,
            val_df=val_df,
            train_target=y_train,
            skip_pre_pipeline_transform=False,
            skip_preprocessing=False,
            use_cache=False,
            model_file_name="m",
            verbose=0,
        )


def test_transform_only_pre_pipeline_preserves_train_and_val_rows():
    """A row-preserving imblearn pre_pipeline keeps both train and val intact; val never resampled."""
    _RESAMPLE_ROW_COUNTS.clear()
    train_df, val_df, y_train = _make_dfs()
    pre_pipeline = ImbPipeline(
        [
            ("res", FunctionSampler(func=_identity_sampler, validate=False)),
            ("passthrough", FunctionTransformer(validate=False)),
        ]
    )
    out_train, out_val = _apply_pre_pipeline_transforms(
        model=LinearRegression(),
        pre_pipeline=pre_pipeline,
        train_df=train_df,
        val_df=val_df,
        train_target=y_train,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        use_cache=False,
        model_file_name="m",
        verbose=0,
    )
    # Sampler fired exactly once -- on the train fit -- and saw the train row count.
    assert _RESAMPLE_ROW_COUNTS == [100], f"sampler must run once on train (100 rows); saw {_RESAMPLE_ROW_COUNTS}"
    assert len(out_train) == 100, f"train rows must be preserved; got {len(out_train)}"
    # Val passed through transform untouched: row count preserved, NOT resampled.
    assert len(out_val) == 40, f"val must be transform-only (40 rows preserved); got {len(out_val)}"
