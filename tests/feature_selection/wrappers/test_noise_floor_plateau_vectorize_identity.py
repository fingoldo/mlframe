"""Regression: the vectorized ``noise_floor_plateau`` inner-j-loop must stay BIT-IDENTICAL to the prior scalar
double-loop implementation (the optimization replaced ~G scalar ``np.percentile`` calls per i with one column-wise
percentile per i). Pins n_star, idx, remaining_gain, remaining_env exactly across many random curve shapes.

This test FAILS on the pre-optimization code only if the optimization changes numerics (it does not). Its real job is
to lock the equivalence so a future "simplify" cannot silently regress the tie-breaking / edge handling.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest


def _load_nf():
    """Load nf."""
    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(here, "..", "..", "..", "src", "mlframe", "feature_selection", "wrappers", "_noise_floor.py")
    spec = importlib.util.spec_from_file_location("_noise_floor_test", mod_path)
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)
    return nf


def _old_plateau(n_grid, real_curve, perm_curves, pct=95.0):
    """Verbatim pre-optimization scalar double-loop reference."""
    n_grid = list(n_grid)
    real_curve = np.asarray(real_curve, dtype=float)
    perm_curves = np.atleast_2d(np.asarray(perm_curves, dtype=float))
    G = len(n_grid)
    remaining_gain = np.full(G, -np.inf)
    remaining_env = np.zeros(G)
    star_idx = G - 1
    found = False
    for i in range(G):
        best_excess, best_rg, best_env = -np.inf, -np.inf, 0.0
        for j in range(i + 1, G):
            rg = real_curve[j] - real_curve[i]
            env = float(np.percentile(perm_curves[:, j] - perm_curves[:, i], pct))
            if (rg - env) > best_excess:
                best_excess, best_rg, best_env = rg - env, rg, env
        remaining_gain[i] = best_rg if i < G - 1 else 0.0
        remaining_env[i] = best_env
        if i < G - 1 and best_excess <= 0 and not found:
            star_idx = i
            found = True
    return n_grid[star_idx], star_idx, remaining_gain, remaining_env


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("n_perm", [1, 3, 5, 50])
def test_vectorized_plateau_bit_identical_to_scalar(seed, n_perm):
    """Vectorized plateau bit identical to scalar."""
    nf = _load_nf()
    rng = np.random.default_rng(seed)
    G = int(rng.integers(2, 20))
    n_grid = sorted(rng.choice(np.arange(1, 600), size=G, replace=False).tolist())
    real = (0.5 + 0.4 * rng.random(G)).astype(float)
    # Mix climbing + flat + noisy curves so tie-breaking and plateau edges are exercised.
    if seed % 3 == 0:
        real = np.sort(real)  # monotone climb
    elif seed % 3 == 1:
        real[G // 2 :] = real[G // 2]  # flat tail (plateau onset)
    perm = 0.5 + 0.03 * rng.standard_normal((n_perm, G))
    for pct in (90.0, 95.0, 99.0):
        old = _old_plateau(n_grid, real, perm, pct=pct)
        new = nf.noise_floor_plateau(n_grid, real, perm, pct=pct)
        assert old[0] == new[0], f"n_star mismatch seed={seed} pct={pct}: {old[0]} vs {new[0]}"
        assert old[1] == new[1], f"idx mismatch seed={seed} pct={pct}"
        assert np.array_equal(old[2], new[2]), f"remaining_gain mismatch seed={seed} pct={pct}"
        assert np.array_equal(old[3], new[3]), f"remaining_env mismatch seed={seed} pct={pct}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly", "-p", "no:cacheprovider"]))
