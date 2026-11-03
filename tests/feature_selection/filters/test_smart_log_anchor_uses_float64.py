"""The frozen smart_log shift is computed from the float64 operand replay will shift, not from a float32 rounding of it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fe_step._step_log_anchor import smart_log_anchor


def test_anchor_equals_the_float64_minimum_based_shift():
    """Column minimum -123456.79: float32 rounds it to -123456.7890625, off by 8e-4, far above the 1e-5 margin."""
    x = np.array([-123456.79, 0.5, 3.0], dtype=np.float64)
    anchor = smart_log_anchor("a", None, pd.DataFrame({"a": x}), {})
    assert anchor == 1e-5 - float(x.min())
    assert float(x.min() + anchor) > 0.0
