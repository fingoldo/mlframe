"""Benchmark batch_pair_mi backends: legacy_seq vs njit prange vs numba.cuda vs cupy.

Four backends:

* ``legacy_seq``  -- pre-Layer-1 reference: sequential merge_vars + compute_mi_from_classes
                      per pair. Stand-in for the joblib(loky) cost ceiling.
* ``layer2_prange``-- Layer 2 ``batch_pair_mi_prange`` (``@njit(parallel=True, nogil=True)``).
* ``cuda``  -- ``batch_pair_mi_cuda``: one CUDA block per pair, shared-memory
                      joint histogram (sized for 48 KB / block budget on cc 6.x).
* ``cupy``  -- ``batch_pair_mi_cupy``: vectorised ``cupy.bincount`` per pair.

The bench drives a single (n_rows, n_features) point per invocation; the
dispatcher crossover thresholds in :mod:`mlframe.feature_selection.filters.batch_pair_mi_gpu`
were tuned from sweeping this script across (200k, 1M, 5M) x (8, 28, 78 pairs).

Numerical equivalence is sanity-checked between every backend pair (max |diff|
must be <1e-9). Backends that are unavailable (no CUDA / no CuPy) are silently
skipped.

Run:
    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \\
        -m mlframe.feature_selection._benchmarks.bench_batch_pair_mi_prange \\
        --n-rows 500000 --n-features 12

Expected ordering at typical mlframe fuzz axis (200k rows, 28 pairs):
    legacy_seq        :  baseline (slowest)
    layer2_prange     :  ~10-15x faster than legacy_seq
    cuda              :  ~20-50x faster than legacy_seq when n_rows >= 200k
    cupy              :  competitive with cuda at n_pairs >= 200 (large fan-out)
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


def _legacy_seq(data, nbins, classes_y, freqs_y, pair_a, pair_b):
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
    parser.add_argument(
        "--skip-legacy", action="store_true",
        help="Skip legacy_seq baseline (it's O(n_pairs * n_rows) Python; >1M rows takes minutes)",
    )
    args = parser.parse_args()

    print(f"=== batch_pair_mi bench (n_rows={args.n_rows:_}, n_features={args.n_features}) ===")
    data, nbins, classes_y, freqs_y = _build(
        args.n_rows, args.n_features, args.n_bins, args.n_classes_y, args.seed,
    )
    pairs = list(itertools.combinations(range(args.n_features), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)
    print(f"n_pairs={len(pairs)}, n_bins={args.n_bins}, n_classes_y={args.n_classes_y}")

    from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
        _CUDA_AVAIL,
        _CUPY_AVAIL,
        batch_pair_mi_cuda,
        batch_pair_mi_cupy,
        batch_pair_mi_njit_prange,
    )

    results = []

    if not args.skip_legacy:
        print("\n--- run A: legacy_seq (merge_vars + compute_mi_from_classes per pair) ---")
        a_res = _time_one("legacy_seq", _legacy_seq, data, nbins, classes_y, freqs_y, pair_a, pair_b)
        print(f"  {a_res['label']:14s} {a_res['wall_s']}s")
        results.append(a_res)
    else:
        print("\n--- skipping legacy_seq (--skip-legacy) ---")

    print("\n--- run B: layer2_prange (njit parallel=True, nogil=True) ---")
    b_res = _time_one(
        "layer2_prange", batch_pair_mi_njit_prange,
        data, pair_a, pair_b, nbins, classes_y, freqs_y,
    )
    print(f"  {b_res['label']:14s} {b_res['wall_s']}s")
    results.append(b_res)

    if _CUDA_AVAIL:
        print("\n--- run C: cuda (numba.cuda one-block-per-pair, shared-memory hist) ---")
        try:
            c_res = _time_one(
                "cuda", batch_pair_mi_cuda,
                data, pair_a, pair_b, nbins, classes_y, freqs_y,
            )
            print(f"  {c_res['label']:14s} {c_res['wall_s']}s")
            results.append(c_res)
        except Exception as e:
            print(f"  cuda skipped: {type(e).__name__}: {e}")
    else:
        print("\n--- skipping cuda (numba.cuda unavailable) ---")

    if _CUPY_AVAIL:
        print("\n--- run D: cupy (vectorised cupy.bincount per pair) ---")
        try:
            d_res = _time_one(
                "cupy", batch_pair_mi_cupy,
                data, pair_a, pair_b, nbins, classes_y, freqs_y,
            )
            print(f"  {d_res['label']:14s} {d_res['wall_s']}s")
            results.append(d_res)
        except Exception as e:
            print(f"  cupy skipped: {type(e).__name__}: {e}")
    else:
        print("\n--- skipping cupy (cupy unavailable) ---")

    # Sanity check: every backend agrees with layer2_prange (CPU reference).
    ref_out = b_res["out"]
    print("\n=== Numerical sanity (vs layer2_prange) ===")
    for r in results:
        if r["label"] == "layer2_prange":
            continue
        diff = float(np.max(np.abs(r["out"] - ref_out)))
        ok = "OK" if diff < 1e-9 else "FAIL"
        print(f"  {r['label']:14s}  max|diff| = {diff:.2e}  [{ok}]")

    # Speedup vs first available baseline (legacy_seq if present, else layer2_prange).
    baseline = results[0]
    print(f"\n=== Speedups (baseline = {baseline['label']}) ===")
    for r in results:
        if r is baseline:
            print(f"  {r['label']:14s}  {r['wall_s']:>8.4f}s   1.00x   (baseline)")
        else:
            spd = baseline["wall_s"] / max(1e-9, r["wall_s"])
            print(f"  {r['label']:14s}  {r['wall_s']:>8.4f}s   {spd:>5.2f}x")


if __name__ == "__main__":
    main()
