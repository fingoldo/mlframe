"""Cluster 27 regression: trainset aging without timestamps is refused (audit 2026-11-20, finding 27-F12).

With no timestamps the kept fraction used to be a positional slice of a shuffled split, which is a random subsample rather than the most recent
rows. The row-based split now refuses the aging limit in that case. Without the limit the split is unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.training._splitting_rowbased import _split_row_based_without_timestamps


def _call(aging):
    """Run the no-timestamp row-based split on 200 rows with the given aging limit."""
    df = pd.DataFrame({"a": np.arange(200.0)})
    return _split_row_based_without_timestamps(
        df, None, None, None, 0.2, 0.1, 0, True, None, True, aging,
    )


def test_aging_without_timestamps_is_refused():
    """A set trainset_aging_limit with no timestamps raises instead of slicing by position (27-F12)."""
    with pytest.raises(ValueError, match="requires timestamps"):
        _call(0.5)


def test_no_aging_limit_still_splits():
    """Control: without an aging limit the split still returns train and test indices (27-F12)."""
    out = _call(None)
    test_idx, train_idx = out[1], out[3]
    assert len(train_idx) > 0 and len(test_idx) > 0
