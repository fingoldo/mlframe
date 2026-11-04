"""GroupTimeSeriesSplit reads groups by position, so a pandas Series with any index gives the same folds as the equivalent array."""
import numpy as np
import pandas as pd

from mlframe.models.selection import GroupTimeSeriesSplit


def test_series_with_a_non_default_index_matches_the_array():
    """groups[idx] was a label lookup: a Series indexed 100..107 raised KeyError: 0, and a shuffled index silently regrouped rows."""
    X = np.zeros((8, 1))
    values = list("aabbccdd")
    arr_folds = [(tr.tolist(), te.tolist()) for tr, te in GroupTimeSeriesSplit(n_splits=2).split(X, groups=np.array(values))]
    ser_folds = [(tr.tolist(), te.tolist()) for tr, te in GroupTimeSeriesSplit(n_splits=2).split(X, groups=pd.Series(values, index=range(100, 108)))]
    assert ser_folds == arr_folds
