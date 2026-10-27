"""The routing-cell enumeration returns the same cells, in the same order, with one thread and with several."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_routing_fe import _enumerate_routing_cells


def test_threaded_enumeration_equals_serial():
    """Names, metadata and values of every cell are identical for n_jobs=1 and n_jobs=4, including a column with NaN that is mean-filled."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(800, 7)), columns=[f"x{i}" for i in range(7)])
    X.loc[::17, "x2"] = np.nan
    args = (X, list(X.columns), ("identity", "log1p_abs", "tanh"), ("hermite", "legendre"), (1, 2, 3))
    serial = _enumerate_routing_cells(*args, n_jobs=1)
    threaded = _enumerate_routing_cells(*args, n_jobs=4)
    assert serial[0] == threaded[0] and serial[2] == threaded[2]
    assert len(serial[0]) > 20
    for a, b in zip(serial[1], threaded[1]):
        np.testing.assert_array_equal(a, b)
