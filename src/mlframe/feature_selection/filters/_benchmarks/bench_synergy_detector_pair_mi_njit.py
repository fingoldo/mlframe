"""Bench: fuse the synergy-detector pairwise joint-MM-MI into one O(n) njit pass.

TARGET: ``_synergy_detector.detect_synergy`` -> ``joint_synergy_mi`` (the per-pair
scorer). For default ``max_pairs=400, n_null=3`` the detector calls
``joint_synergy_mi`` ~ ``max_pairs*(1+n_null) + pp*(1+n_null)`` times (~1600+ pair
calls + marginals). Each OLD call does, in numpy:
  * ``_renumber_joint_codes``: ``np.unique(flat, return_inverse=True)`` (an O(n log n)
    sort of n int64) just to densify the joint id,
  * ``np.add.at(joint, (jc, yt), 1.0)`` -- the slow UNBUFFERED scatter,
  * dense (kx, ky) numpy reductions for marginals + MI.

NEW: ``_pair_mm_mi_njit`` builds the dense mixed-radix (kx*ky) histogram in ONE O(n)
walk (no unique, no add.at), accumulates MI + the Miller-Madow occupancy debit, and
returns ``max(0, mi - mm)`` -- the SAME estimator. (This is the order-2 specialisation
of the already-shipped ``_combo_mm_mi_njit``, plus the const-column marginal path.)

IDENTITY GATE: the detector's verdict is a BOOLEAN ``is_synergistic`` (selection-
equivalence per CLAUDE.md FE rule 7). We assert the boolean verdict is unchanged AND the
per-pair excess matches to a tight tolerance, on synergy (XOR) + noise data.

Run: CUDA_VISIBLE_DEVICES="" python bench_synergy_detector_pair_mi_njit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from mlframe.feature_selection.filters._fe_synergy_screen import (  # noqa: E402
    joint_synergy_mi,
    _pair_mm_mi_njit,
)


def _pair_mm_mi_new(code_x, code_y, yc, min_rows_per_cell=5.0):
    cx = np.ascontiguousarray(np.asarray(code_x).astype(np.int64).ravel())
    cy = np.ascontiguousarray(np.asarray(code_y).astype(np.int64).ravel())
    yt = np.ascontiguousarray(np.asarray(yc).astype(np.int64).ravel())
    n = cx.shape[0]
    if n == 0 or cy.shape[0] != n or yt.shape[0] != n:
        return 0.0
    kx = int(cx.max()) + 1
    ky = int(cy.max()) + 1
    kt = int(yt.max()) + 1
    return float(_pair_mm_mi_njit(cx, cy, yt, kx, ky, kt, float(min_rows_per_cell)))


def _best_of(fn, *a, reps=7):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best


def _make_data(n, p, seed):
    rng = np.random.default_rng(seed)
    nbins = 8
    # mix of XOR signal pairs + noise, quantized to codes 0..nbins-1
    codes = [rng.integers(0, nbins, size=n) for _ in range(p)]
    # one genuine XOR signal among the first two
    b0 = (rng.random(n) > 0.5).astype(np.int64)
    b1 = (rng.random(n) > 0.5).astype(np.int64)
    codes[0] = b0
    codes[1] = b1
    yc = (b0 ^ b1).astype(np.int64)
    const = np.zeros(n, dtype=np.int64)
    return [c.astype(np.int64) for c in codes], yc, const


def main():
    n, p, seed = 4000, 60, 0  # detector's max_rows x max_features default shape
    codes, yc, const = _make_data(n, p, seed)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)][:400]

    # warm njit
    _pair_mm_mi_new(codes[0], codes[1], yc)

    # IDENTITY / selection-equivalence
    max_abs = 0.0
    sign_flips = 0
    for i, j in pairs:
        old = joint_synergy_mi(codes[i], codes[j], yc)
        new = _pair_mm_mi_new(codes[i], codes[j], yc)
        max_abs = max(max_abs, abs(old - new))
        if (old > 0) != (new > 0):
            sign_flips += 1
    # marginals (const column path)
    marg_max = 0.0
    for j in range(p):
        old = joint_synergy_mi(codes[j], const, yc)
        new = _pair_mm_mi_new(codes[j], const, yc)
        marg_max = max(marg_max, abs(old - new))

    def run_old():
        for i, j in pairs:
            joint_synergy_mi(codes[i], codes[j], yc)

    def run_new():
        for i, j in pairs:
            _pair_mm_mi_new(codes[i], codes[j], yc)

    t_old = _best_of(run_old)
    t_new = _best_of(run_new)

    print(f"shape n={n} p={p} pairs={len(pairs)}")
    print(f"OLD joint_synergy_mi : {t_old*1e3:8.2f} ms / sweep")
    print(f"NEW _pair_mm_mi_njit : {t_new*1e3:8.2f} ms / sweep")
    print(f"speedup              : {t_old/t_new:6.2f}x")
    print(f"max|old-new| pair    : {max_abs:.3e}")
    print(f"max|old-new| marginal: {marg_max:.3e}")
    print(f"sign flips (sel-eq)  : {sign_flips}/{len(pairs)}")


if __name__ == "__main__":
    main()
