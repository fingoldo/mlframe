"""iter79 bench: fused njit packer for the K<=64 bitmap-Jaccard fast path.

``_pack_for_bitmap`` is the dominant cost of the multilabel-metric block at n=200k
(cProfile: 0.396s / 0.564s = 70% across 60 calls, 2 per ``jaccard_score_multilabel``).
The old numpy path allocates a full ``(N, 64)`` zero buffer, copies ``arr`` into the
first ``K`` columns, and runs ``np.packbits`` over all 64 columns even when K<64.

The fused njit kernel packs (N, K) uint8 -> (N,) uint64 in one pass, computing each
label's final bit index directly (``(j>>3)*8 + (7-(j&7))`` -- np.packbits big-endian
within a byte + the LE uint64 view reversing byte order). Bit-identical to the numpy
reference; the parallel twin auto-selects above ``_PARALLEL_MULTILABEL_THRESHOLD``.

Run:  python -m mlframe.metrics._benchmarks.bench_multilabel_pack_bitmap_iter79

Measured (n=200k, Win32 store py3.14, best-of-50 isolated):
    K=16: old 4.452ms  njit 0.898ms  njit_par 0.409ms  (par 10.9x)
    K=32: old 4.633ms  njit 1.305ms  njit_par 0.548ms  (par 8.5x)
    K=64: old 1.646ms  njit 2.303ms  njit_par 0.795ms  (par 2.1x; seq njit LOSES at K=64
          because the old path skips the zero-buffer there, hence par is the default)
End-to-end 3-metric block (jaccard+hamming+subset, 30 iters): 0.564s -> 0.154s (2.7x);
jaccard_score_multilabel cumtime 0.459s -> 0.049s (9.4x). RESOLVED, bit-identical.
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.metrics._multilabel_metrics import (
    _pack_for_bitmap_numpy,
    _pack_for_bitmap_kernel_seq,
    _pack_for_bitmap_kernel_par,
)


def _best(fn, arr, reps: int = 50) -> float:
    fn(arr)  # warm
    best = 1e9
    for _ in range(reps):
        s = time.perf_counter()
        fn(arr)
        best = min(best, time.perf_counter() - s)
    return best * 1000.0


def main() -> None:
    rng = np.random.default_rng(0)
    n = 200_000
    for k in (16, 32, 64):
        arr = (rng.random((n, k)) < 0.2).astype(np.uint8)
        ref = _pack_for_bitmap_numpy(arr)
        seq = _pack_for_bitmap_kernel_seq(arr)
        par = _pack_for_bitmap_kernel_par(arr)
        assert np.array_equal(ref, seq) and np.array_equal(ref, par), f"K={k} not bit-identical"  # nosec B101 - internal invariant check in src/mlframe/metrics/_benchmarks, not reachable with untrusted input
        t_old = _best(_pack_for_bitmap_numpy, arr)
        t_seq = _best(_pack_for_bitmap_kernel_seq, arr)
        t_par = _best(_pack_for_bitmap_kernel_par, arr)
        print(f"K={k}: old {t_old:.3f}ms  njit {t_seq:.3f}ms  njit_par {t_par:.3f}ms  (par {t_old / t_par:.1f}x)")


if __name__ == "__main__":
    main()
