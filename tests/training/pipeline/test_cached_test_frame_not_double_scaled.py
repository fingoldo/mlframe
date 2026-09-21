"""A cached test frame already transformed by the pipeline must not be transformed again on a later target.

On a pipeline-cache hit the test frame is the cached pipeline OUTPUT, and ``skip_pre_pipeline_transform`` is set. The
override that still transforms a RAW-looking frame judged rawness by column count, which a column-count-preserving
pipeline (a scaler) cannot fail: its output has its input's width. So every target after the first in a suite scored
its test split on doubly-scaled features - a linear composite target recorded test RMSE 0.522 where the deployed model
scores 0.281. The frame now carries the mark of the pipeline that transformed it, and a marked frame is left alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.pipeline._pipeline_helpers import _prepare_test_split


def _setup(seed: int = 0):
    """A fitted imputer+scaler pipeline, a model on its output, and a raw test frame."""
    rng = np.random.default_rng(seed)
    cols = list("abcd")
    x_train = pd.DataFrame(rng.normal(loc=10.0, scale=3.0, size=(200, 4)), columns=cols)
    y = x_train @ np.array([1.0, -0.5, 0.2, 0.0]) + rng.normal(size=200)
    pp = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler())]).set_output(transform="pandas")
    xt = pp.fit_transform(x_train)
    model = LinearRegression().fit(xt, y)
    x_test = pd.DataFrame(rng.normal(loc=10.0, scale=3.0, size=(50, 4)), columns=cols)
    return pp, model, x_test


def _prepare(pp, model, frame, *, skip: bool):
    """Run the test-split preparation the trainer runs."""
    out, _, _ = _prepare_test_split(
        df=frame, test_df=frame, test_idx=np.arange(len(frame)), test_target=np.zeros(len(frame)), target=np.zeros(len(frame)),
        real_drop_columns=[], model=model, pre_pipeline=pp, skip_pre_pipeline_transform=skip,
    )
    return out


def test_a_cached_transformed_frame_is_not_transformed_again():
    """The first target transforms the frame; the next target, on a cache hit, must get exactly the same values."""
    pp, model, x_test = _setup()
    first = _prepare(pp, model, x_test, skip=False)
    again = _prepare(pp, model, first, skip=True)
    np.testing.assert_array_equal(np.asarray(again), np.asarray(first))
    np.testing.assert_allclose(np.asarray(first), np.asarray(pp.transform(x_test)))


def test_a_raw_frame_under_the_skip_flag_is_still_transformed():
    """The override's reason to exist: a raw frame reaching the skip path (e.g. carrying NaN) must still be transformed."""
    pp, model, x_test = _setup()
    raw = x_test.copy()
    raw.iloc[0, 0] = np.nan
    out = _prepare(pp, model, raw, skip=True)
    assert not out.isna().any().any(), "a raw frame under the skip flag must still go through the imputer"
    np.testing.assert_allclose(np.asarray(out), np.asarray(pp.transform(raw)))


def test_a_frame_marked_by_another_pipeline_is_not_trusted():
    """The mark names one pipeline; a frame transformed by a different one is judged afresh."""
    pp, model, x_test = _setup()
    other_pp, _, _ = _setup(seed=1)
    transformed_by_other = _prepare(other_pp, model, x_test, skip=False)
    out = _prepare(pp, model, transformed_by_other, skip=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(pp.transform(transformed_by_other)))
