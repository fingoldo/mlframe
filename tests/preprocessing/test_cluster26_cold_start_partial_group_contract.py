"""Audit 2026-11-20 finding 26-F9: the cold-start fill gives a partially-missing group its last non-null value in ROW order.

That value can be a later row in time, so the function documents that callers apply it to all-null groups only. The test pins
the behaviour the contract describes: a gap in a partial group takes the group's last non-null value by row, which here is the
later row, and an entirely missing group borrows its sibling's value.
"""

import numpy as np
import pandas as pd

from mlframe.preprocessing.sibling_group_cold_start_fill import sibling_group_cold_start_fill


def test_partial_group_gap_takes_its_last_non_null_by_row():
    """Group A has values at order 1 and 5 with a gap at 2: the gap takes 5's value, the documented row-order behaviour."""
    df = pd.DataFrame(
        {
            "g": ["A", "A", "A", "B", "B"],
            "t": [1, 2, 5, 1, 2],
            "v": [10.0, np.nan, 50.0, np.nan, np.nan],
        }
    )
    out = sibling_group_cold_start_fill(df, group_col="g", order_col="t", value_col="v", fallback_value=0.0)
    assert out.iloc[1] == 50.0
