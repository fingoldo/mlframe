"""float32 and float64 X select the same features on a well-separated synthetic.

The only existing float32 selection-parity test casts y; X is where discretisation, MI and pairwise correlation consume the dtype, and the
njit/CUDA backends route float32 separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(dtype, seed=0, n=1500):
    """Two strong signal columns far above noise, so the selection is decided by a wide margin, cast to ``dtype``."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 6))
    y = ((2.0 * base[:, 0] + 1.5 * base[:, 1]) > 0).astype(np.int64)
    return pd.DataFrame(base.astype(dtype), columns=[f"x{i}" for i in range(6)]), y


def _fit(X, y):
    """A deterministic light fit without FE."""
    MRMR._FIT_CACHE.clear()
    return MRMR(random_seed=3, n_jobs=1, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)


def test_float32_X_matches_float64_X_support():
    """Identical support; gains agree to float32 precision."""
    X64, y = _frame(np.float64)
    X32, _ = _frame(np.float32)
    m64 = _fit(X64, y)
    m32 = _fit(X32, y)
    assert np.array_equal(np.asarray(m64.support_), np.asarray(m32.support_))
    assert {"x0", "x1"} <= set(map(str, m64.get_feature_names_out()))
    assert np.allclose(np.asarray(m32.mrmr_gains_), np.asarray(m64.mrmr_gains_), rtol=1e-5, atol=1e-7)
