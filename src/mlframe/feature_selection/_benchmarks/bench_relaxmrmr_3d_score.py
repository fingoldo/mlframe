"""Bench: per-candidate timing of ``relax_mrmr_score`` (RelaxMRMR 3-D-MI, Vinh 2016) on realistic shapes.

The score costs ``O(|S|^2)`` 3-D plug-in MIs per candidate (one CMI + one joint-CMI + one unconditional joint-MI per
selected pair, plus ``|S|`` marginal MIs). This records the warmed, best-of-N wall time so the corrected path's cost is
tracked. The corrected interaction term adds one ``_mi_x_pair_njit`` call per selected pair vs the pre-fix version, so
the pair loop does roughly one extra O(n) histogram pass per pair -- this bench quantifies that.

Run:

    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection._benchmarks.bench_relaxmrmr_3d_score
    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection._benchmarks.bench_relaxmrmr_3d_score --quick
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score


def _make_case(rng, n, n_selected, K):
    """A redundant cluster: latent drives y; candidate + selected set are noisy copies (worst case for the pair loop)."""
    latent = rng.integers(0, K, n).astype(np.int64)
    y = latent.copy()

    def noisy(p=0.12):
        out = latent.copy()
        flip = rng.random(n) < p
        out[flip] = rng.integers(0, K, int(flip.sum()))
        return out.astype(np.int64)

    x = noisy()
    selected = [noisy() for _ in range(n_selected)]
    return x, selected, y, K, [K] * n_selected, K


def _best_of(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    K = 8
    repeats = 5 if args.quick else 15
    shapes = [(2000, 5), (10000, 10)] if args.quick else [(2000, 5), (10000, 10), (50000, 15)]

    # Warm numba (compiles every njit kernel on the path) outside the timed region.
    xw, selw, yw, Kx, Ks, Ky = _make_case(rng, 1000, 4, K)
    for a in (0.0, 1.0):
        relax_mrmr_score(xw, selw, yw, Kx, Ks, Ky, alpha=a)

    results = []
    for n, n_sel in shapes:
        x, sel, y, Kx, Ks, Ky = _make_case(rng, n, n_sel, K)
        t0 = _best_of(lambda: relax_mrmr_score(x, sel, y, Kx, Ks, Ky, alpha=0.0), repeats)
        t1 = _best_of(lambda: relax_mrmr_score(x, sel, y, Kx, Ks, Ky, alpha=1.0), repeats)
        row = {"n": n, "n_selected": n_sel, "ms_alpha0": round(t0 * 1e3, 3), "ms_alpha1": round(t1 * 1e3, 3)}
        results.append(row)
        print(f"n={n:>6} |S|={n_sel:>2}  alpha=0: {t0 * 1e3:8.3f} ms   alpha=1: {t1 * 1e3:8.3f} ms")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "relaxmrmr_3d_score.json")
    with open(out_path, "w") as f:
        json.dump({"repeats": repeats, "K": K, "results": results}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
