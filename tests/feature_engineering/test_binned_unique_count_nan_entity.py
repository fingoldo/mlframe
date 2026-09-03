"""Regression test: binned_unique_count must not crash when entity_col contains NaN/missing labels.

Pre-fix, pd.factorize's -1 sentinel for a NaN entity label flowed unmasked into the combined
(entity_code, bin_code) key whenever that row's value_col observation was itself finite, producing
a negative key and then a negative "unique_entity_code" fed straight into np.bincount, which
raises ValueError on any negative input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.binned_unique_count import binned_unique_count


def test_nan_entity_label_does_not_crash():
    """A NaN entity_col label alongside a valid value_col observation must not raise ValueError."""
    df = pd.DataFrame(
        {
            "entity": ["a", "a", "b", "b", None, np.nan],
            "value": [1.0, 2.0, 10.0, 11.0, 5.0, 6.0],
        }
    )
    out = binned_unique_count(df, entity_col="entity", value_col="value", n_bins=4)  # pre-fix: ValueError from np.bincount
    assert set(out["entity"].dropna()) == {"a", "b"}
    # Real entities still get a sane (>=1) count; nothing about their result is corrupted by the NaN rows.
    assert (out.loc[out["entity"].isin(["a", "b"]), "binned_unique_value"] >= 1).all()
