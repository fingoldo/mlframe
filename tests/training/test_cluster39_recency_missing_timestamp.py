"""Audit 2026-11-20 finding 39-F6: a missing timestamp gets the floor recency weight, so no training weight is NaN."""

import numpy as np
import pandas as pd

from mlframe.training.extractors._extractors_dtype_helpers import get_sample_weights_by_recency


def test_missing_timestamp_gets_the_floor_weight_and_no_weight_is_nan():
    """A NaT row gets min_weight; the other rows keep finite, recency-ordered weights."""
    ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01", None, "2025-01-01"]))
    weights = np.asarray(get_sample_weights_by_recency(ts, min_weight=0.5, weight_drop_per_year=0.5), dtype=np.float64)
    assert np.isfinite(weights).all()
    assert weights[2] == 0.5
    assert weights[3] > weights[0]
