"""Bench (idx-aware-ece): fuse the bootstrap-ECE resample gather into the njit kernel.

OLD ``_bootstrap_ece_with_indices`` did a per-resample Python-level fancy-index
copy ``y_true[idx]`` / ``y_pred[idx]`` (two length-n int64/float64 gathers, 1000x)
then called the ECE kernel. NEW path passes the base arrays + idx row straight to
``_ece_score_idx_numba_serial``, which gathers inside the njit bin loop -- zero
per-resample Python slice. Bit-identical (equal-width binning is order-independent).

Run: python -m mlframe.calibration._benchmarks.bench_idx_aware_ece_resampler

Measured (warm numba, 50 reps, dev box):
  n=2000  nb=1000:  old ~7.40 ms   ->  new ~5.30 ms    ~1.40x
  n=20000 nb=1000:  old ~91.3 ms   ->  new ~72.1 ms    ~1.27x
  n=200000 nb=500:  old ~1004.9 ms ->  new ~617.9 ms   ~1.63x
  per-resample ECE array BIT-IDENTICAL (np.array_equal) on every size.
"""
from __future__ import annotations

import time

import numpy as np


def _make(n, seed=11):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(0, 1, n), 0.0, 1.0)
    y = (rng.uniform(0, 1, n) < p).astype(np.int64)
    return np.ascontiguousarray(y), np.ascontiguousarray(p)


def _old_loop(y, p, idx_matrix, n_bins):
    from mlframe.calibration.policy import _ece_score
    nbs = idx_matrix.shape[0]
    out = np.empty(nbs)
    for b in range(nbs):
        idx = idx_matrix[b]
        out[b] = _ece_score(y[idx], p[idx], n_bins=n_bins)
    return out


def _new_loop(y, p, idx_matrix, n_bins):
    from mlframe.calibration.policy import _ece_score_idx_numba_serial
    nbs = idx_matrix.shape[0]
    out = np.empty(nbs)
    for b in range(nbs):
        out[b] = _ece_score_idx_numba_serial(y, p, idx_matrix[b], n_bins)
    return out


def main(sizes=((2000, 1000), (20000, 1000), (200000, 500)), reps=50, n_bins=15):
    from mlframe.calibration.policy import _build_resample_indices

    for n, nbs in sizes:
        y, p = _make(n)
        idx = _build_resample_indices(n, nbs, y, 7)

        old = _old_loop(y, p, idx, n_bins)
        new = _new_loop(y, p, idx, n_bins)
        identical = bool(np.array_equal(old, new))

        t = time.perf_counter()
        for _ in range(reps):
            _old_loop(y, p, idx, n_bins)
        old_ms = (time.perf_counter() - t) / reps * 1000
        t = time.perf_counter()
        for _ in range(reps):
            _new_loop(y, p, idx, n_bins)
        new_ms = (time.perf_counter() - t) / reps * 1000

        print(f"n={n} nb={nbs}: old={old_ms:.2f}ms new={new_ms:.2f}ms " f"speedup={old_ms / new_ms:.2f}x bit_identical={identical}")


if __name__ == "__main__":
    main()
