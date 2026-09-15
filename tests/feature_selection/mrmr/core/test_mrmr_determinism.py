"""With an explicit random_seed, MRMR selection does not depend on the global numpy / random state, and two fresh fits are byte-identical.

pytest-randomly reseeds the global RNGs per test; a code path that falls back to an unseeded generator would pass under ``-p no:randomly``
and flake only in CI. Gains are compared byte-exactly: an unseeded subsample perturbs them far above any float-reorder tolerance.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(seed=0, n=600):
    """Two signal columns, an interaction and noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    y = ((X["x0"] + 0.7 * X["x1"] + 0.5 * X["x2"] * X["x3"]) > 0).astype(np.int64).to_numpy()
    return X, y


def _fit(X, y):
    """A fresh fit with an explicit seed and the fit cache cleared, so nothing is replayed."""
    MRMR._FIT_CACHE.clear()
    return MRMR(random_seed=42, n_jobs=1, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2, skip_retraining_on_same_content=False).fit(X, y)


def test_selection_independent_of_global_numpy_seed():
    """Three different global seeds give the same support and byte-identical gains."""
    X, y = _frame()
    supports, gains, names = [], [], []
    for gseed in (0, 1, 999_983):
        np.random.seed(gseed)
        random.seed(gseed)
        m = _fit(X, y)
        supports.append(np.asarray(m.support_).copy())
        gains.append(np.asarray(m.mrmr_gains_).copy())
        names.append(list(map(str, m.get_feature_names_out())))
    assert all(np.array_equal(supports[0], s) for s in supports[1:])
    assert all(n == names[0] for n in names[1:])
    assert all(np.array_equal(gains[0], g) for g in gains[1:]), "gains differ across global seeds: an unseeded RNG is in the fit path"


def test_two_fits_same_random_state_are_byte_identical():
    """Two fresh estimators with the same seed agree byte-for-byte without relying on the fit cache."""
    X, y = _frame(seed=5)
    g1 = np.asarray(_fit(X, y).mrmr_gains_)
    g2 = np.asarray(_fit(X, y).mrmr_gains_)
    assert np.array_equal(g1, g2)
