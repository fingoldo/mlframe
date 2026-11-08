"""A/B of ``MLFRAME_PERM_NULL_TRUSTED_CHUNKS`` on ``conditional_perm_null_gpu``: the range guard of the batched CMI once per null (trusted chunks) against once per chunk of permutations.

Usage: ``python bench_perm_null_trusted_chunks.py [rows ...]`` (default 100000 400000 1000000). The support ``z`` is near-continuous (about n / 4 distinct values, the redundancy gate's shape at full n), so the
dense joint is wide and the permutations go in many small chunks, which is where a per-chunk guard repeats. Alternating A/B in one process, device synchronised around every call, medians of 5.
"""

import os
import statistics
import sys
import time

import numpy as np


def _inputs(n: int, seed: int = 0):
    """Candidate codes (10 bins), a binary target and a near-continuous dense support with its stratum order and rank."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 10, n).astype(np.int64)
    y = (rng.random(n) < 0.5).astype(np.int64)
    z = rng.integers(0, max(2, n // 4), n).astype(np.int64)
    order = np.argsort(z, kind="stable")
    _, z_rank = np.unique(z[order], return_inverse=True)
    return x, y, z, order, z_rank.astype(np.float64)  # Intentional float64: the stratum rank of the kernel contract


def main(rows: list) -> None:
    """Print the median time of the null with the per-chunk guard (0) and with the guard once per null (1)."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_cmi_perm_null_gpu import conditional_perm_null_gpu

    for n in rows:
        x, y, z, order, z_rank = _inputs(n)
        times = {"0": [], "1": []}

        def one(flag: str) -> float:
            """One timed call under the given setting, device synchronised."""
            os.environ["MLFRAME_PERM_NULL_TRUSTED_CHUNKS"] = flag
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            res = conditional_perm_null_gpu(x, y, z, order=order, z_rank=z_rank, n_permutations=25, quantile=0.99, seed=1, salt=0)
            cp.cuda.Device().synchronize()
            return time.perf_counter() - t0 if res is not None else float("nan")

        one("1"), one("0")  # warm-up: kernels, resident caches
        for _ in range(5):
            for flag in ("0", "1"):
                times[flag].append(one(flag))
        a, b = statistics.median(times["0"]), statistics.median(times["1"])
        print(f"rows={n:>8}  per-chunk guard {a * 1e3:8.1f} ms   guard once {b * 1e3:8.1f} ms   speedup x{a / b:.3f}   spread {min(times['0']) * 1e3:.0f}-{max(times['0']) * 1e3:.0f} / {min(times['1']) * 1e3:.0f}-{max(times['1']) * 1e3:.0f} ms")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100_000, 400_000, 1_000_000])
