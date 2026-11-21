"""Audit 2026-11-20 findings 26-F28 and 26-F11.

26-F28: a non-finite time or value is dropped for its entity before the GP fit, so one NaN reading does not fail the whole call.
26-F11: float16 input reaches the outlier kernels as float32, not as float16 that numba cannot type.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.outliers import _as_floating
from mlframe.signal import gp_smoothing as gp


def test_one_nan_value_does_not_fail_the_whole_panel():
    """A NaN value in one entity: the call returns, that entity is smoothed from its finite points, and every entity has a row."""
    pytest.importorskip("sklearn")
    df = pd.DataFrame(
        {
            "e": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "v": [0.1, np.nan, 0.3, 0.5, 0.4, 1.0, 1.1, 0.9, 1.0, 1.2],
        }
    )
    out = gp.compute_gp_smoothed_features(df, entity_col="e", time_col="t", value_col="v", query_times=[0.0, 2.0, 4.0])
    assert out.height == 2
    assert np.isfinite(out["gp_mean_t1"].to_numpy()).all()


def test_float16_is_widened_to_float32_for_the_kernels():
    """Half precision comes back as float32; float32 and float64 pass through; integers become float64."""
    assert _as_floating(np.array([1.0, 2.0], dtype=np.float16)).dtype == np.float32
    assert _as_floating(np.array([1.0], dtype=np.float32)).dtype == np.float32
    assert _as_floating(np.array([1, 2])).dtype == np.float64
