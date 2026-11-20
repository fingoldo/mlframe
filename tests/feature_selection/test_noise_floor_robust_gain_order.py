"""The robust noise-floor bench ranks by gain on the rows it is given and returns every column (audit 2026-11-20, finding 12-F22)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.fs_hybrid.round4_noise_floor_robust import _gain_order


def test_gain_order_returns_every_column_with_the_signal_first():
    """The informative column must lead the ordering and no column may be dropped."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=["n0", "n1", "sig", "n2", "n3"])
    y = (X["sig"] + 0.2 * rng.standard_normal(n) > 0).astype(int)
    order = _gain_order(X, y)
    assert sorted(order) == sorted(X.columns)
    assert order[0] == "sig"
