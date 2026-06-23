"""Identity pin for the conditional-permutation strata-array hoist.

The optimization materialises the per-stratum index arrays ONCE before the
permutation loop (instead of rebuilding each ``arr`` from a Python list via
``np.asarray`` on every permutation). It is bit-identical by construction:
``rng.permutation`` sees the SAME int64 arrays in the SAME dict-iteration order,
so the RNG draw sequence -> the null distribution -> (observed, p_value) are all
unchanged.

This test reconstructs the pre-optimization reference inline (the per-permutation
``np.asarray`` rebuild) and asserts the live function reproduces it EXACTLY across
several shapes / seeds / stratum counts, including a case with singleton strata.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._conditional_permutation import (
    conditional_permutation_test,
)
from mlframe.feature_selection.filters._cmi_perm_stop import _cmi_plugin_njit


def _reference(x, y, z, nbx, nby, nbz, n_permutations, seed):
    """Pre-optimization algorithm: rebuild each stratum array per permutation."""
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x).astype(np.int64).ravel()
    y = np.asarray(y).astype(np.int64).ravel()
    z = np.asarray(z).astype(np.int64).ravel()

    def stat(_x, _y, _z):
        return float(_cmi_plugin_njit(_x, _y, _z, int(nbx), int(nby), int(nbz)))

    observed = float(stat(x, y, z))
    if n_permutations <= 0:
        return observed, 1.0
    strata: dict[int, list] = {}
    for idx, zv in enumerate(z):
        strata.setdefault(int(zv), []).append(idx)
    null_dist = np.empty(int(n_permutations), dtype=np.float64)
    for p in range(int(n_permutations)):
        x_perm = x.copy()
        for zv, idx_list in strata.items():
            if len(idx_list) <= 1:
                continue
            arr = np.asarray(idx_list, dtype=np.int64)
            shuffled = rng.permutation(arr)
            x_perm[arr] = x[shuffled]
        null_dist[p] = float(stat(x_perm, y, z))
    n_exceed = int(np.count_nonzero(null_dist >= observed))
    p_value = (1.0 + n_exceed) / (int(n_permutations) + 1.0)
    return observed, p_value


@pytest.mark.parametrize(
    "n,nbx,nby,nbz,B,seed",
    [
        (2000, 6, 6, 8, 100, 3),
        (5000, 4, 8, 10, 150, 11),
        (1500, 8, 4, 20, 80, 0),
        # Singleton-heavy: nbz close to n forces many size-1 strata (the skip branch).
        (300, 4, 4, 250, 60, 5),
    ],
)
def test_cpt_strata_hoist_bit_identical(n, nbx, nby, nbz, B, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nbx, n)
    y = rng.integers(0, nby, n)
    z = rng.integers(0, nbz, n)

    obs_new, p_new = conditional_permutation_test(
        x, y, z, nbx, nby, nbz, n_permutations=B, seed=seed
    )
    obs_ref, p_ref = _reference(x, y, z, nbx, nby, nbz, B, seed)

    assert obs_new == obs_ref, "observed statistic diverged"
    assert p_new == p_ref, "permutation p-value diverged (RNG order changed)"
