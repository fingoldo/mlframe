"""Cluster 13 regression: the BorutaShap auto routing probe never reads the 30 percent holdout (audit 2026-11-20, finding 13-F4)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_probe_rows_are_exactly_the_non_holdout_rows():
    """The rows handed to the probe equal the train part of the split the holdout mode uses, and the held-out rows are absent."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": rng.normal(size=n)})
    y = pd.Series((rng.uniform(size=n) > 0.5).astype(int))
    bs = BorutaShap(classification=True, random_state=3, importance_measure="auto")
    px, py = bs._probe_rows_outside_holdout(X, y)
    tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=3, stratify=bs.stratify)
    np.testing.assert_array_equal(px["a"].to_numpy(), tr.astype(float))
    assert len(px) == len(py) == len(tr) and not set(px["a"].astype(int)) & set(te)
