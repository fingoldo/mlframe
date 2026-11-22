"""Audit 2026-11-20 finding 26-F6: the power-transform search falls back to identity when every candidate fails the guard.

When identity is not among the candidates and the target-retention guard rejects all of them, the search used to take min() of an
empty set. The docstring's identity fallback is now evaluated and kept.
"""

import numpy as np
import pandas as pd

from mlframe.preprocessing.gaussian_power_transform_search import gaussian_power_transform_search


def test_all_candidates_rejected_falls_back_to_identity():
    """boxcox alone, with a retention factor no candidate can meet: the column is kept as identity."""
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=200)
    y = (x + rng.normal(scale=0.1, size=200)).astype(float)
    df = pd.DataFrame({"x": x})
    result = gaussian_power_transform_search(df, candidate_transforms=("boxcox",), y=y, require_target_correlation_retention=10.0)
    assert result["x"]["best_transform"] == "identity"
