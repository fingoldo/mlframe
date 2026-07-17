"""Regression test for tie-heavy qcut binning in create_fairness_subgroups (EDGE2).

Pre-fix ``pd.qcut(feature_vals, q=cont_nbins)`` on a zero-inflated / tie-heavy numeric feature produced
non-unique quantile edges and raised the opaque ``ValueError: Bin edges must be unique``. The fix passes
``duplicates="drop"`` so such features collapse to fewer bins instead of crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.metrics._fairness_metrics import create_fairness_subgroups


def test_tie_heavy_numeric_feature_does_not_raise():
    """Tie heavy numeric feature does not raise."""
    n = 3000
    # Zero-inflated: ~80% zeros, the rest a small spread -> repeated quantile edges at q=3.
    rng = np.random.default_rng(0)
    vals = np.where(rng.random(n) < 0.8, 0.0, rng.integers(1, 4, n).astype(float))
    df = pd.DataFrame({"income": vals})

    # Must not raise "Bin edges must be unique".
    subgroups = create_fairness_subgroups(df, features=["income"], cont_nbins=3, min_pop_cat_thresh=10)
    assert isinstance(subgroups, dict)
