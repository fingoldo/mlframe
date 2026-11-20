"""Audit 2026-11-20 finding 26-F4: the expanding mean and std divide by the OBSERVED count, so a NaN in an entity's history does not shrink them."""

import numpy as np
import pandas as pd

from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift


def test_expanding_statistics_skip_nan_history_rows():
    """History [10, 20, NaN, 30]: the expanding mean and std run over the OBSERVED values (20 and 10), not divided by the 4 rows including the NaN.

    The synthetic row at 30 is standardized to (30 - 20) / 10 = 1.0. Counting the NaN in the divisor gave a different mean and std.
    """
    df = pd.DataFrame({"e": ["A"] * 5, "t": [1, 2, 3, 4, 5], "x": [10.0, 20.0, np.nan, 30.0, 40.0]})
    out = augment_temporal_drift(df, entity_col="e", time_col="t", feature_cols=["x"], min_history=2, n_drop_options=(1,))
    synth = out[out["_temporal_drift_augmented"]]
    expected = (30.0 - 20.0) / 10.0
    assert len(synth) == 1
    np.testing.assert_allclose(synth["x"].to_numpy(), [expected], rtol=1e-9)
