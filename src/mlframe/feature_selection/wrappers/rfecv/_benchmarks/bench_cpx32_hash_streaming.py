"""CPX32: streamed row-chunk X-content hashing vs whole-frame ``.tobytes()``.

The RFECV skip-retrain signature fingerprints X via blake2b. The legacy pandas numeric path did
``h.update(np.ascontiguousarray(X[cols].to_numpy()).tobytes())`` -- the ``.tobytes()`` allocates a
SECOND full-frame buffer, doubling peak RAM on top of the array ``to_numpy()`` already built. On a
100+ GB frame that extra copy OOMs the host. The fix streams ``h.update`` over row-chunks so peak
EXTRA RAM is one ~64 MB chunk, not the whole frame -- bit-identically (contiguous row slices'
``.tobytes()`` concatenate to the full buffer's bytes).

Run: CUDA_VISIBLE_DEVICES="" python bench_cpx32_hash_streaming.py
"""
from __future__ import annotations

import hashlib
import time
import tracemalloc

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers.rfecv._fit_init import _stream_hash_array


def _old_hash(arr: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _new_hash(arr: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    _stream_hash_array(h, arr)
    return h.hexdigest()


def _best_of(fn, arr, n=3):
    best = float("inf")
    digest = None
    for _ in range(n):
        t0 = time.perf_counter()
        digest = fn(arr)
        best = min(best, time.perf_counter() - t0)
    return best, digest


def _peak_mb(fn, arr):
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(arr)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def main():
    rows, cols = 2_000_000, 50
    print(f"shape={rows}x{cols} float64 (~{rows*cols*8/1e6:.0f} MB)")
    arr = np.ascontiguousarray(pd.DataFrame(np.random.rand(rows, cols)).to_numpy())

    t_old, d_old = _best_of(_old_hash, arr)
    t_new, d_new = _best_of(_new_hash, arr)
    p_old = _peak_mb(_old_hash, arr)
    p_new = _peak_mb(_new_hash, arr)

    print(f"OLD  wall={t_old*1e3:8.2f} ms  peak_extra={p_old:8.1f} MB  digest={d_old}")
    print(f"NEW  wall={t_new*1e3:8.2f} ms  peak_extra={p_new:8.1f} MB  digest={d_new}")
    print(f"identical={d_old == d_new}  peak_ratio_new/old={p_new/p_old:.4f}")


if __name__ == "__main__":
    main()
