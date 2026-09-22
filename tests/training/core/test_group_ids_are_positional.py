"""group_ids from an extractor are indexed positionally by the split indices, whatever the extractor returned."""

import numpy as np
import pandas as pd

from mlframe.training.core._phase_helpers import _positional_group_ids


def test_a_series_with_a_shuffled_index_is_indexed_by_position():
    groups = pd.Series([10, 20, 30, 40], index=[3, 2, 1, 0], name="match_id")
    val_idx = np.array([0, 1])
    assert list(groups[val_idx]) == [40, 30], "the raw Series does a label lookup - the hazard being guarded"
    assert list(_positional_group_ids(groups)[val_idx]) == [10, 20]


def test_none_and_arrays_pass_through():
    assert _positional_group_ids(None) is None
    arr = np.array([1, 1, 2])
    assert np.array_equal(_positional_group_ids(arr), arr)
