"""The cross-sectional aggregate-stat argument accepts only the documented reducers, never an arbitrary numpy attribute.

``compute_cross_sectional_neighbor_features`` resolved ``agg_stats`` with ``getattr(np, stat)``, so a caller-supplied name could run any numpy
callable. (``windowed_edge_aggregate_diff`` already validates ``agg`` against an allow-list on HEAD.) ``np.unwrap`` is harmless and runs without error on these inputs, so a
name outside the allow-list is accepted by the old code and refused by the new one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.cross_sectional_neighbors import compute_cross_sectional_neighbor_features


def test_cross_sectional_refuses_a_non_reducer_stat():
    """An ``agg_stats`` entry outside the allow-list raises ValueError before any feature is computed."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"snap": np.repeat(np.arange(2), 30), "f0": rng.normal(size=60), "f1": rng.normal(size=60), "f2": rng.normal(size=60)})

    with pytest.raises(ValueError, match="unsupported agg_stats"):
        compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=5, agg_stats=("unwrap",))
