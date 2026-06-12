"""Microbench: per-suite cache for near_collinear_keep_mask on a shared featureset.

near_collinear_keep_mask_fast is already njit + KTC-dispatched (at the compute
floor for a single call). But the per-base keep-mask is a pure function of the
(base-dropped) feature matrix + threshold. When several targets share the SAME
feature matrix per base (same df / train_idx / sample_idx / surviving features),
discovery recomputes the identical mask once per target. A content-signature
cache returns the stored mask for targets 2..N.

CAVEAT (measured here): in discovery the surviving feature subset comes from a
per-target y-correlation LEAKAGE filter (_filter_features), so the matrix is
target-DEPENDENT in general -- the cache only hits when two targets happen to
produce the byte-identical matrix for a base. This bench measures BOTH the
kernel cost (to size the win) AND the cache key cost (to confirm the key is far
cheaper than a recompute, so a miss never regresses).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite.discovery._benchmarks.bench_near_collinear_cache
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from mlframe.training.composite.discovery._collinear_numba import (
    near_collinear_keep_mask_fast,
)
from mlframe.training.composite.discovery._eval_stats import (
    _near_collinear_keep_mask_numpy,
)

_THR = 0.99


def _make_matrix(n: int, b: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, b // 3 + 1))
    cols = []
    for j in range(b):
        src = base[:, j % base.shape[1]]
        # Mix some near-duplicates (high corr) with independent columns.
        if j % 4 == 0:
            cols.append(src + 1e-3 * rng.standard_normal(n))
        else:
            cols.append(rng.standard_normal(n))
    return np.ascontiguousarray(np.stack(cols, axis=1))


def _time(fn, iters: int) -> float:
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    print(f"{'n':>8} {'B':>5} {'kernel_ms':>10} {'keycost_ms':>11} {'key/kernel':>11}")
    for n, b in ((4000, 30), (20000, 60), (80000, 120)):
        m = _make_matrix(n, b)
        # Warm JIT.
        near_collinear_keep_mask_fast(m, corr_threshold=_THR, reference_fn=_near_collinear_keep_mask_numpy)
        kernel_ms = _time(
            lambda: near_collinear_keep_mask_fast(
                m, corr_threshold=_THR, reference_fn=_near_collinear_keep_mask_numpy,
            ),
            iters=20,
        )
        # Collision-safe content key: shape + dtype + threshold + blake2b of the
        # full contiguous buffer. A miss pays this once; the kernel would have
        # run anyway, so the only net cost on a miss is the hash.
        mc = np.ascontiguousarray(m, dtype=np.float64)

        def _key():
            h = hashlib.blake2b(digest_size=16)
            h.update(mc.tobytes())
            return (mc.shape, mc.dtype.str, _THR, h.digest())
        key_ms = _time(_key, iters=50)
        print(f"{n:>8} {b:>5} {kernel_ms:>10.3f} {key_ms:>11.4f} {key_ms / kernel_ms:>10.4f}x")
    print()
    # Per-suite multiplier: N targets that DO share the matrix for a base.
    m = _make_matrix(20000, 60)
    near_collinear_keep_mask_fast(m, corr_threshold=_THR, reference_fn=_near_collinear_keep_mask_numpy)
    one = _time(
        lambda: near_collinear_keep_mask_fast(
            m, corr_threshold=_THR, reference_fn=_near_collinear_keep_mask_numpy,
        ),
        iters=20,
    )
    print(f"single-call kernel @ 20k x 60: {one:.3f} ms")
    print(f"recompute for 5 shared-matrix targets: {5 * one:.3f} ms (cache -> ~1x compute + 4 key lookups)")


if __name__ == "__main__":
    main()
