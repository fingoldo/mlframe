"""Lock: imbalance resampling is fit/applied on TRAIN rows only; val is
transform-only (cardinal val=ES-detector rule). A regression here -- a future
caller fitting/resampling the imblearn sampler on val (the ES detector) -- would
leak the holdout distribution into selection and inflate the honest estimate.

The contract is documented at the builder entry (``_build_pre_pipelines``); this
test pins the live behaviour in ``_apply_pre_pipeline_transforms``: the imblearn
Pipeline resamples under ``fit_transform`` (train) and is pass-through under
``transform`` (val), so val row-count is NEVER changed by the sampler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

imblearn = pytest.importorskip("imblearn")
from imblearn import FunctionSampler  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from mlframe.training.pipeline._pipeline_helpers import (  # noqa: E402
    _apply_pre_pipeline_transforms,
)


# Module-level call recorder so the FunctionSampler callable stays picklable/clonable.
_RESAMPLE_ROW_COUNTS: list[int] = []


def _drop_first_row(X, y):
    """A sampler that visibly changes row count when (and only when) it runs.

    imblearn only invokes the sampler under fit/fit_resample, never under
    transform, so recording here proves whether val was ever resampled.
    """
    _RESAMPLE_ROW_COUNTS.append(len(X))
    return X[1:], y[1:]


def test_resampler_runs_on_train_only_not_on_val():
    _RESAMPLE_ROW_COUNTS.clear()
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    val_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y_train = (rng.normal(size=100) > 0).astype(int)

    # A transform-only imblearn pre_pipeline (sampler + passthrough), mirroring the
    # mlframe pre_pipeline shape (selectors/preprocessing, NOT a final predictor).
    pre_pipeline = ImbPipeline([
        ("res", FunctionSampler(func=_drop_first_row, validate=False)),
        ("passthrough", FunctionTransformer(validate=False)),
    ])

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
    assert _RESAMPLE_ROW_COUNTS == [100], (
        f"resampler must run once on train (100 rows); saw {_RESAMPLE_ROW_COUNTS}"
    )
    # Val passed through transform untouched: row count preserved, NOT resampled.
    assert len(out_val) == 40, (
        f"val must be transform-only (40 rows preserved); got {len(out_val)}"
    )
