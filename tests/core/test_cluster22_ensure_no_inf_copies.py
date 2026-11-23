"""Cluster 22 regression: ensure_no_infinity_pd does not mutate the caller's DataFrame (audit 2026-11-20, finding 22-F39)."""

import numpy as np
import pandas as pd

from mlframe.core.helpers import ensure_no_infinity_pd


def test_caller_frame_keeps_its_infinities():
    """The returned frame has the infinity replaced; the input frame still holds it (22-F39)."""
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    out = ensure_no_infinity_pd(df, verbose=0)
    assert np.isinf(df["a"].iloc[1])
    assert np.isfinite(out["a"]).all()
