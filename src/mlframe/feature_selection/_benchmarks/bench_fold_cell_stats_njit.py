"""Paired interleaved best-of-N: np.add.at on masked gathers vs the fused njit pass, per-fold target-encode loop (isolated + OOF end-to-end)."""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters._cat_pair_fe import _kfold_target_encode_codes
from mlframe.feature_selection.filters._fold_cell_stats import fold_cell_sum_cnt


def _old_oof(codes, y, n_folds=5, smoothing=10.0, random_state=0):
    """Reference implementation using masked gathers and np.add.at."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(codes)
    gm = float(y_arr.mean())
    n_cells = int(codes.max()) + 1
    perm = np.random.default_rng(random_state).permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n) % n_folds
    oof = np.full(n, gm)
    c64 = codes.astype(np.int64)
    for f in range(n_folds):
        tm = fold_ids != f
        cs = np.zeros(n_cells)
        cc = np.zeros(n_cells)
        np.add.at(cs, c64[tm], y_arr[tm])
        np.add.at(cc, c64[tm], 1.0)
        nz = cc > 0
        raw = np.where(nz, cs / np.maximum(cc, 1.0), gm)
        sh = (cc * raw + smoothing * gm) / (cc + smoothing)
        oof[~tm] = np.where(nz, sh, gm)[c64[~tm]]
    return oof


def _best(fn, reps):
    """Best perf_counter over reps."""
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    """Run multi-size sweep and print speedups plus identity."""
    fold_cell_sum_cnt(np.zeros(4, np.int64), np.zeros(4), np.zeros(4, np.int64), 0, 2)
    for n, k in [(20_000, 50), (200_000, 50), (2_000_000, 50), (2_000_000, 5000)]:
        rng = np.random.default_rng(0)
        codes = rng.integers(0, k, n)
        y = rng.normal(size=n)
        assert np.array_equal(_old_oof(codes, y), _kfold_target_encode_codes(codes, y)[0])
        told = tnew = 1e9
        for _ in range(3):
            told = min(told, _best(lambda: _old_oof(codes, y), 3))
            tnew = min(tnew, _best(lambda: _kfold_target_encode_codes(codes, y), 3))
        print(f"n={n} cells={k} old={told*1e3:.1f}ms new={tnew*1e3:.1f}ms speedup={told/tnew:.2f}x bit-identical")


if __name__ == "__main__":
    main()
