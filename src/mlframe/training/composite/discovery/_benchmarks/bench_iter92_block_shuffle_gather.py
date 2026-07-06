"""iter92 A/B bench: fused njit block-shuffle gather vs the legacy numpy broadcast+mask+gather.

The auto-base permutation-MI null filter (``_auto_base.py``) block-shuffles each screened column's
bin-codes (int64) or values (float32) ``auto_base_null_perms`` times per column to build the null MI
distribution. The legacy ``_block_shuffle`` fast path built a ``(n_blocks, block_len)`` int64 index
template (broadcast), ravelled it, boolean-masked ``idx < m``, then fancy-indexed ``arr[idx]`` --
O(n_blocks*block_len) temp + mask every call. ``block_shuffle_gather`` fuses index-build + gather
into one njit pass with no temp/mask, bit-identical (same element order for the same ``perm`` draw).

At the 1M-driven discovery the MI screen subsamples to ~100k; the null block length is
``sqrt(n_screen)`` so the shuffled arrays are ~20k-100k long. Measured per-call (warm, best-of):
2.6-3.0x faster, byte-identical output on int64 + float32.

Run:
    MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python -m mlframe.training.composite.discovery._benchmarks.bench_iter92_block_shuffle_gather [m]
"""
from __future__ import annotations

import sys

sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402
import numba  # noqa: F401,E402

from timeit import default_timer as timer  # noqa: E402

import numpy as np  # noqa: E402

from mlframe.training.composite.discovery._collinear_numba import block_shuffle_gather  # noqa: E402


def _legacy(arr, perm, block_len):
    m = arr.size
    idx = (perm[:, None] * block_len + np.arange(block_len)[None, :]).ravel()
    idx = idx[idx < m]
    return arr[idx]


def _bench_one(m: int) -> None:
    block_len = max(2, int(np.sqrt(m)))
    n_blocks = (m + block_len - 1) // block_len
    for dt in (np.int64, np.float32):
        arr = np.random.default_rng(0).integers(0, 50, m).astype(dt)
        # Identity across seeds (incl. trailing short block).
        for s in range(8):
            perm = np.random.default_rng(s).permutation(n_blocks)
            assert np.array_equal(_legacy(arr, perm, block_len), block_shuffle_gather(arr, perm, block_len))  # nosec B101 - internal invariant check in src/mlframe/training/composite/discovery/_benchmarks, not reachable with untrusted input
        rng = np.random.default_rng(1)
        for _ in range(3):
            block_shuffle_gather(arr, rng.permutation(n_blocks), block_len)
        N = 3000
        rng = np.random.default_rng(2)
        perms = [rng.permutation(n_blocks) for _ in range(N)]
        s = timer()
        for p in perms:
            _legacy(arr, p, block_len)
        old = (timer() - s) / N * 1e6
        s = timer()
        for p in perms:
            block_shuffle_gather(arr, p, block_len)
        new = (timer() - s) / N * 1e6
        print(f"m={m:>7} {dt.__name__:>8}: IDENTITY OK | OLD {old:6.1f}us  NEW {new:6.1f}us  speedup {old/new:.2f}x")


def main(argv):
    ms = [int(argv[1])] if len(argv) > 1 else [20000, 100000]
    for m in ms:
        _bench_one(m)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
