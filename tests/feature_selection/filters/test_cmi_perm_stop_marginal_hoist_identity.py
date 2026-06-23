"""Pins the CMI permutation-stop marginal-hoist equivalence (bench wave: rejected 2.5x lead).

The rejected fused permutation loop (see
``filters/_benchmarks/bench_cmi_perm_stop_fused_loop.py``) hoists the ``Pz`` / ``Pyz``
marginals out of the per-permutation loop because they are INVARIANT under within-stratum
permutation of X (only X moves; Y and Z are fixed). This test pins that invariance + the
arithmetic equivalence to the production ``_cmi_plugin_njit`` so a future revisit (e.g. a
bit-exact in-njit RNG that would unlock the 2.5x bit-identically) builds on proven-exact math.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._cmi_perm_stop import (
    _cmi_plugin_njit,
    cmi_permutation_stop,
)


def _make(n, K_x, K_y, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, K_x, n).astype(np.int64)
    y = rng.integers(0, K_y, n).astype(np.int64)
    s1 = rng.integers(0, 5, n).astype(np.int64)
    s2 = rng.integers(0, 4, n).astype(np.int64)
    z = (s1 * 4 + s2).astype(np.int64)
    return x, y, z, 20


def _cmi_hoisted(x, y, z, K_x, K_y, K_z):
    """Reference CMI computed with Pz/Pyz/Pxz folded the way the rejected fused kernel does."""
    n = x.shape[0]
    joint = np.zeros((K_x, K_y, K_z), dtype=np.float64)
    Pz = np.zeros(K_z, dtype=np.float64)
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)
    for i in range(n):
        joint[x[i], y[i], z[i]] += 1.0
        Pz[z[i]] += 1.0
        Pyz[y[i], z[i]] += 1.0
    n_f = float(n)
    cmi = 0.0
    for k in range(K_z):
        if Pz[k] <= 0.0:
            continue
        for i in range(K_x):
            pxz = joint[i, :, k].sum()
            if pxz <= 0.0:
                continue
            for j in range(K_y):
                v = joint[i, j, k]
                if v <= 0.0 or Pyz[j, k] <= 0.0:
                    continue
                p_xyz = v / n_f
                cmi += p_xyz * np.log((p_xyz * (Pz[k] / n_f)) / ((pxz / n_f) * (Pyz[j, k] / n_f)))
    return max(0.0, cmi)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n", [2000, 8000])
def test_marginal_hoist_matches_plugin_cmi(seed, n):
    x, y, z, K_z = _make(n, 8, 8, seed)
    prod = _cmi_plugin_njit(x, y, z, 8, 8, K_z)
    hoist = _cmi_hoisted(x, y, z, 8, 8, K_z)
    assert abs(prod - hoist) <= 1e-9, f"hoist diverged: {prod} vs {hoist}"


def test_marginals_invariant_under_within_stratum_permutation():
    """Pz and Pyz must NOT change when X is permuted within Z-strata -- the premise of the hoist."""
    x, y, z, K_z = _make(4000, 6, 6, 3)
    rng = np.random.default_rng(99)
    K_y = 6
    Pz0 = np.bincount(z, minlength=K_z).astype(np.float64)
    Pyz0 = np.zeros((K_y, K_z))
    for i in range(x.shape[0]):
        Pyz0[y[i], z[i]] += 1.0
    x_perm = x.copy()
    for zv in np.unique(z):
        idx = np.flatnonzero(z == zv)
        if idx.size > 1:
            x_perm[idx] = x[rng.permutation(idx)]
    Pz1 = np.bincount(z, minlength=K_z).astype(np.float64)
    Pyz1 = np.zeros((K_y, K_z))
    for i in range(x_perm.shape[0]):
        Pyz1[y[i], z[i]] += 1.0
    assert np.array_equal(Pz0, Pz1)
    assert np.array_equal(Pyz0, Pyz1)
    assert not np.array_equal(x, x_perm)


def test_cmi_permutation_stop_runs_and_is_deterministic():
    x, y, z_a, _ = _make(3000, 6, 6, 5)
    sel = [(z_a % 5).astype(np.int64), (z_a % 4).astype(np.int64)]
    r1 = cmi_permutation_stop(x, y, sel, 6, 6, [5, 4], n_permutations=50, seed=11)
    r2 = cmi_permutation_stop(x, y, sel, 6, 6, [5, 4], n_permutations=50, seed=11)
    assert r1 == r2
    is_sig, obs, pval = r1
    assert isinstance(is_sig, bool)
    assert obs >= 0.0
    assert 0.0 < pval <= 1.0
