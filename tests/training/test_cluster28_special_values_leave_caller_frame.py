"""Audit 2026-11-20 finding 28-F21: normalising inf on a pandas frame does not change the frame the caller still holds.

The inf to NaN replacement was assigned into the caller's frame. The test replaces inf in a pandas frame and checks the caller's values
keep their infinities while the returned frame has NaN.
"""

import numpy as np
import pandas as pd

from mlframe.training.preprocessing import _process_special_values


def test_inf_replacement_does_not_touch_the_callers_frame():
    """The original frame keeps its inf; the returned frame has NaN where the inf was."""
    df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
    out = _process_special_values(df, verbose=0)
    assert np.isinf(df["x"].to_numpy()[1])
    assert np.isnan(out["x"].to_numpy()[1])
