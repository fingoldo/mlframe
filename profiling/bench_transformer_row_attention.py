"""Bench harness for ``compute_row_attention`` Mode A + Mode B at multiple sizes.

Reports a table per (n_train, d, n_heads, head_dim, k) configuration:
- Mode A wall clock (5-fold OOF; dominated by hnswlib build x5)
- Mode B build wall clock (one-shot key_bank construction)
- Mode B query wall clock per 1000 queries (steady-state after build)
- GPU stage 4 vs CPU stage 4 speedup ratio (when GPU available)

Methodology mirrors ``bench_transformer_rff.py``: 3 warmup discarded, 7 measured, median + IQR, sync GPU before timestamps.

Use the printed numbers to (a) calibrate the row-attention default thresholds in ``row_attention.py``, (b) populate the wall-clock table in README.

For hotspot attribution inside the numba kernels, run ``py-spy record -o profile.svg --native -- D:/ProgramData/anaconda3/python.exe profiling/bench_transformer_row_attention.py``
- cProfile is blind to njit kernel bodies (numba coverage rule).
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import (
    attend,
    build_key_bank,
    compute_row_attention,
)


def _gpu_sync():
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def _time_one(fn, *, warmup: int, measured: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(measured):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = statistics.median(times)
    q1, q3 = times[len(times) // 4], times[(3 * len(times)) // 4]
    return median, q3 - q1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="10000,100000", help="n_train values; 1M is too slow for default bench, pass explicitly if you want it.")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n_query_batches", type=int, default=3, help="Number of Mode-B query batches after build, to amortise H2D.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=3)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print(f"# Row-attention bench (d={args.d}, n_heads={args.n_heads}, head_dim={args.head_dim}, k={args.k}, warmup={args.warmup}, measured={args.measured})\n")
    print("| N | mode A oof (s) | mode B build (s) | mode B query per 1k (s) | gpu_stage4 speedup |")
    print("|---|---|---|---|---|")

    rng = np.random.default_rng(args.seed)
    for n in sizes:
        X = rng.standard_normal((n, args.d)).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        splitter = KFold(n_splits=5, shuffle=True, random_state=args.seed)

        def mode_a():
            compute_row_attention(
                X, y, None, splitter, seed=args.seed,
                n_heads=args.n_heads, head_dim=args.head_dim, k=args.k,
                gpu_stage4=False, dedupe_threshold=None,
            )
        oof_med, _ = _time_one(mode_a, warmup=args.warmup, measured=args.measured)

        def mode_b_build():
            build_key_bank(X_train=X, y_train=y, seed=args.seed, n_heads=args.n_heads, head_dim=args.head_dim)
        build_med, _ = _time_one(mode_b_build, warmup=args.warmup, measured=args.measured)

        # Reuse a single bank across query timings.
        bank = build_key_bank(X_train=X, y_train=y, seed=args.seed, n_heads=args.n_heads, head_dim=args.head_dim)
        query_n = 1000
        X_query = rng.standard_normal((query_n, args.d)).astype(np.float32)
        def mode_b_query():
            attend(bank=bank, X_query=X_query, k=args.k)
        q_med, _ = _time_one(mode_b_query, warmup=args.warmup, measured=args.measured)

        # GPU stage 4 speedup (single query batch).
        try:
            from mlframe.feature_engineering.transformer._utils import is_gpu_available
            if is_gpu_available():
                def query_cpu():
                    attend(bank=bank, X_query=X_query, k=args.k, stage4_callable=None)
                def query_gpu():
                    from mlframe.feature_engineering.transformer._kernels_cupy import row_attention_stage4_cupy
                    _gpu_sync()
                    attend(bank=bank, X_query=X_query, k=args.k, stage4_callable=row_attention_stage4_cupy)
                    _gpu_sync()
                cpu_q, _ = _time_one(query_cpu, warmup=args.warmup, measured=args.measured)
                gpu_q, _ = _time_one(query_gpu, warmup=args.warmup, measured=args.measured)
                speedup_str = f"{cpu_q / max(gpu_q, 1e-9):.2f}x"
            else:
                speedup_str = "N/A (no GPU)"
        except Exception as exc:
            speedup_str = f"err ({type(exc).__name__})"

        print(f"| {n} | {oof_med:.2f} | {build_med:.2f} | {q_med:.3f} | {speedup_str} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
