"""Cluster 11 regression: the oracle arm keeps exactly the declared relevant names that are present (audit 2026-11-20, finding 11-F15).

The membership test against the names set was hoisted out of the comprehension. The output must be unchanged: declared names that
are in the frame are selected, absent ones are not, and order follows the declaration.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.fs_hybrid._arms_baselines import OracleInformativeArm


def test_oracle_keeps_only_present_declared_names():
    """Declared names absent from X are dropped; present ones are selected in declaration order (11-F15)."""
    X = pd.DataFrame(np.zeros((10, 4)), columns=["a", "b", "c", "d"])
    y = np.zeros(10)
    out = OracleInformativeArm(relevant=["c", "missing", "a"])._compute(X, y)
    assert out["provenance"]["n_present"] == 2
    assert out["provenance"]["n_declared"] == 3
    assert out["support"].tolist() == [True, False, True, False]
