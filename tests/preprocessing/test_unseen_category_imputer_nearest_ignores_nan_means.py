"""Nearest-mean imputation skips categories without a finite mean and sends NaN query values to the mode."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer


def _fit():
    """Train: 'a' mean 1, 'b' mean 10, 'c' only NaN values (no mean); 'a' is the mode."""
    train = pd.DataFrame({"cat": ["a"] * 6 + ["b"] * 3 + ["c"] * 3, "v": [1.0] * 6 + [10.0] * 3 + [np.nan] * 3})
    return UnseenCategoryImputer(["cat"], similarity_mode="nearest", value_column="v").fit(train)


def test_large_query_maps_to_the_nearest_real_category_not_the_nan_mean_one():
    """Query 100 is nearest to 'b' (10); it used to land on 'c', whose NaN mean sorted last."""
    out = _fit().transform(pd.DataFrame({"cat": ["unseen"], "v": [100.0]}))
    assert out["cat"].iloc[0] == "b"


def test_nan_query_falls_back_to_the_mode():
    """A NaN value has no nearest category: the mode ('a') is used instead of an arbitrary NaN-mean category."""
    out = _fit().transform(pd.DataFrame({"cat": ["unseen"], "v": [np.nan]}))
    assert out["cat"].iloc[0] == "a"
