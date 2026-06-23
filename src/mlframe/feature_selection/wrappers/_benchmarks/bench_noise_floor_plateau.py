"""Perf bench for ``noise_floor_plateau`` (wrappers/_noise_floor.py).

THE HOT PATH: the plateau rule is an O(G^2) double loop whose inner body calls
``np.percentile(perm_curves[:, j] - perm_curves[:, i], pct)`` once PER (i, j)
pair -- i.e. ~G*(G-1)/2 separate ``np.percentile`` dispatches, each paying the
full Python-level arg-parsing + a partial-sort over ``n_perm`` draws on a tiny
length-``n_perm`` vector. For the default grid (G=19) at n_perm=50 that is 171
``np.percentile`` calls per cut, dominated by per-call dispatch overhead.

THE OPTIMIZATION: for a fixed ``i`` the envelope for ALL larger ``j`` is one
vectorized percentile over axis 0:

    np.percentile(perm_curves[:, i+1:] - perm_curves[:, i:i+1], pct, axis=0)

This collapses the inner G-loop of scalar percentile calls into ONE call per
``i`` (G-1 total instead of ~G^2/2), with identical numerics (same data, same
``pct``, same default linear interpolation -- ``np.percentile`` along axis 0 is
the column-wise application of the scalar version).

IDENTITY: bit-identical by construction (same draws, same percentile method).
Run:
    PYTHONPATH=src python src/mlframe/feature_selection/wrappers/_benchmarks/bench_noise_floor_plateau.py
"""
from __future__ import annotations

import time

import numpy as np


def _old_plateau(n_grid, real_curve, perm_curves, pct=95.0):
    """Verbatim prior implementation (scalar np.percentile per (i, j))."""
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


def _bench(fn, args, reps=300):
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best / reps * 1e6  # us per call


def main():
    # Import the module directly (not via the wrappers package __init__, which pulls heavy GPU/cupy modules).
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(here, "..", "_noise_floor.py")
    spec = importlib.util.spec_from_file_location("_noise_floor_bench", mod_path)
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)
    noise_floor_plateau, _default_grid = nf.noise_floor_plateau, nf._default_grid

    rng = np.random.default_rng(0)
    for p, n_perm in [(500, 50), (200, 50), (43, 3)]:
        n_grid = _default_grid(p)
        G = len(n_grid)
        real = np.sort(0.5 + 0.3 * rng.random(G))
        perm = 0.5 + 0.02 * rng.standard_normal((n_perm, G))

        old = noise_floor_plateau  # placeholder; replaced below
        # identity check
        o = _old_plateau(n_grid, real, perm)
        n = noise_floor_plateau(n_grid, real, perm)
        assert o[0] == n[0] and o[1] == n[1]
        assert np.array_equal(o[2], n[2]) and np.allclose(o[3], n[3], rtol=0, atol=0), "identity broken"

        t_old = _bench(_old_plateau, (n_grid, real, perm))
        t_new = _bench(noise_floor_plateau, (n_grid, real, perm))
        print(f"p={p:4d} G={G:2d} n_perm={n_perm:3d}: OLD {t_old:8.1f} us  NEW {t_new:8.1f} us  "
              f"speedup {t_old / t_new:5.2f}x  (n_star={n[0]} identical)")


if __name__ == "__main__":
    main()
