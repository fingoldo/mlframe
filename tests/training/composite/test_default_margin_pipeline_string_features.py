"""The default composite base-margin model fits on a frame with string features and missing numeric cells.

A plain median imputer raised "could not convert string to float" on the string column, failing whole fuzz combos. The default pipeline
now imputes numeric columns by median and encodes the rest, while a purely numeric input keeps the original pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from mlframe.training.composite.classification import _default_margin_pipeline


def test_default_margin_pipeline_fits_string_features_with_missing_numerics():
    """A frame with a string column and NaN numerics fits and predicts probabilities."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan] * 5, "cat": ["A", "B", "A", "C", "B", "A"] * 5})
    y = rng.integers(0, 2, size=len(X))

    pipe = _default_margin_pipeline(X).fit(X, y)

    assert pipe.predict_proba(X).shape == (len(X), 2)


def test_default_margin_pipeline_numeric_input_keeps_median_imputer():
    """A numeric-only input keeps the plain median-imputer first step."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 2.0, 2.0]})

    pipe = _default_margin_pipeline(X)

    assert isinstance(pipe.steps[0][1], SimpleImputer)
