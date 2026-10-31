"""The gradient seeder's pool matrix built from a frame equals the whole-frame matrix indexed by the pool."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


def test_pool_matrix_from_frame_equals_full_matrix_slice():
    """A mixed-dtype frame (float, int, bool columns): the subset conversion gives the same values and dtype as converting everything."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 9, size=n), "c": rng.normal(size=n).astype(np.float32), "d": rng.random(n) > 0.5, "e": rng.normal(size=n)})
    pool = [4, 0, 2, 1]
    full = np.asarray(X, dtype=_crit_np_dtype())[:, pool]
    subset = np.asarray(X.iloc[:, list(pool)], dtype=_crit_np_dtype())
    assert subset.dtype == full.dtype and np.array_equal(subset, full)
