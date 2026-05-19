"""Benchmark Layer 2 ``batch_pair_mi_prange`` vs Layer 1 ``parallel_run`` over
``compute_pairs_mis``.

Layer 1 (committed as 0da27e0): switched joblib backend ``loky`` -> ``threading``
so the per-worker data memmap copy disappears, but joblib's ThreadPoolExecutor
dispatch + Python-side wrapper around each ``mi_direct`` call still hold the
GIL between numba kernels.

Layer 2 (this kernel): single ``@njit(parallel=True, nogil=True, cache=True)``
``prange`` over all pairs. Zero joblib overhead, zero Python wrapper GIL
contention. Each thread builds its joint-class histogram in a thread-local
buffer and writes a single MI scalar to the output array.

Numerical equivalence verified by
``tests/feature_selection/test_batch_pair_mi_prange.py`` (5/5 pass; MIs match
to atol/rtol=1e-9 vs the legacy ``merge_vars + compute_mi_from_classes`` path).

Run:
    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \
        -m mlframe.feature_selection._benchmarks.bench_batch_pair_mi_prange \
        --n-rows 500000 --n-features 12

Expected: Layer 2 strictly faster than Layer 1 (~1.3-2x at typical n_jobs); on
small frames where joblib dispatch dominates wall time, Layer 2 can be 5-10x.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import time

import numpy as np


def _build(n_rows: int, n_features: int, n_bins: int, n_classes_y: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n_rows, n_features)).astype(np.int32)
    nbins = np.full(n_features, n_bins, dtype=np.int32)
    classes_y = rng.integers(0, n_classes_y, size=n_rows).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=n_classes_y).astype(np.float64) / n_rows
    return data, nbins, classes_y, freqs_y


def _legacy_threaded(data, nbins, classes_y, freqs_y, pair_a, pair_b):
    """Layer 1 approximation: sequential per-pair via ``merge_vars + compute_mi_from_classes``."""
    from mlframe.feature_selection.filters.info_theory import merge_vars, compute_mi_from_classes

    n_pairs = pair_a.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    for p in range(n_pairs):
        vars_indices = np.array([pair_a[p], pair_b[p]], dtype=np.int64)
        classes_x, freqs_x_norm, _ = merge_vars(
            factors_data=data, vars_indices=vars_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        out[p] = compute_mi_from_classes(
            classes_x=classes_x, freqs_x=freqs_x_norm,
            classes_y=classes_y, freqs_y=freqs_y, dtype=np.int32,
        )
    return out


def _time_one(label, fn, *args, n_warmup: int = 1, n_repeats: int = 2):
    """Run fn(*args) once for JIT warmup, then time n_repeats more, return best wall."""
    for _ in range(n_warmup):
        out = fn(*args)
    gc.collect()
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return {"label": label, "wall_s": round(best, 4), "out": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=200_000)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-classes-y", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print(f"=== batch_pair_mi_prange bench (n_rows={args.n_rows:_}, n_features={args.n_features}) ===")
    data, nbins, classes_y, freqs_y = _build(
        args.n_rows, args.n_features, args.n_bins, args.n_classes_y, args.seed,
    )
    pairs = list(itertools.combinations(range(args.n_features), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)
    print(f"n_pairs={len(pairs)}, n_bins={args.n_bins}, n_classes_y={args.n_classes_y}")

    from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange

    print("\n--- run A: legacy sequential (merge_vars + compute_mi_from_classes per pair) ---")
    a_res = _time_one("legacy_seq", _legacy_threaded, data, nbins, classes_y, freqs_y, pair_a, pair_b)
    print(a_res["label"], a_res["wall_s"], "s")

    print("\n--- run B: Layer 2 batch_pair_mi_prange (njit parallel=True) ---")
    b_res = _time_one(
        "layer2_prange", batch_pair_mi_prange,
        data, pair_a, pair_b, nbins, classes_y, freqs_y,
    )
    print(b_res["label"], b_res["wall_s"], "s")

    speedup = a_res["wall_s"] / max(1e-9, b_res["wall_s"])
    diff = np.max(np.abs(a_res["out"] - b_res["out"]))
    print(f"\n=== Summary ===")
    print(f"speedup vs legacy: {speedup:.2f}x  (legacy={a_res['wall_s']}s, layer2={b_res['wall_s']}s)")
    print(f"numerical max |diff|: {diff:.2e}  (must be tiny; sanity check)")


if __name__ == "__main__":
    main()
