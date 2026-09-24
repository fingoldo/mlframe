"""A column that shares fewer than three finite rows with y carries no leak evidence and is kept.

The leak-corr test re-checks every column with a non-finite cell on the rows where both it and y are finite. With fewer than
three such rows the re-check was skipped and the column kept the score of the mean-imputed pass: an almost constant column
whose two informative rows sit on finite y can score near 1 and be dropped as a leak of y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery._filter import _filter_features
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class _Disc:
    """The attribute surface ``_filter_features`` reads off the discovery instance."""

    def __init__(self):
        self.config = CompositeTargetDiscoveryConfig(enabled=True, random_state=0)
        self._target_col = "y"
        self._patterns_compiled = []
        self._filter_drops = []


def test_two_jointly_finite_rows_are_no_evidence_of_a_leak():
    n = 400
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    x = np.full(n, np.nan)
    x[:60] = rng.normal(size=60)  # 60 finite rows pass the finite-row gate
    y[2:60] = np.nan  # but only rows 0 and 1 are finite in both
    y[0], y[1] = 1e6, -1e6  # the two joint rows dominate both columns' spread on the finite-y rows
    x[0], x[1] = 50.0 + y[0], 50.0 + y[1]
    df = pd.DataFrame({"y": y, "sparse": x, "f": rng.normal(size=n)})
    disc = _Disc()
    kept = _filter_features(disc, df, ["sparse", "f"], y, np.arange(n))
    assert "sparse" in kept, disc._filter_drops
