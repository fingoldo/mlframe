"""Benchmark torch.from_numpy(np_array) writable vs .copy() on mlframe MLP shapes.

The warning "The given NumPy array is not writeable" fires when the underlying numpy
buffer is read-only (pandas 2.x with PyArrow-backed Series + zero-copy views, or any
ndarray with flags.writeable=False). Two strategies to silence it:

- Direct from_numpy on the read-only buffer (warning still fires, but harmless).
- ``.copy()`` first (warning silenced, but pays N*itemsize memcpy).

Bench result on this machine (2026-05-21):

  shape=(10000, 50) (2.0 MB):
    writable baseline    :    0.002 ms/iter
    non-writable direct  :    3.757 ms/iter  (fires warning)
    non-writable .copy() :    0.700 ms/iter  (285.85x baseline)
    smart conditional    :    0.014 ms/iter  (5.55x baseline)
  shape=(100000, 50) (20.0 MB):
    writable baseline    :    0.005 ms/iter
    non-writable direct  :    0.005 ms/iter  (fires warning)
    non-writable .copy() :    7.004 ms/iter  (1554.80x baseline)
    smart conditional    :    0.121 ms/iter  (26.91x baseline)
  shape=(1000000, 50) (200.0 MB):
    writable baseline    :    0.003 ms/iter
    non-writable direct  :    0.009 ms/iter  (fires warning)
    non-writable .copy() :   66.528 ms/iter  (21741.29x baseline)
    smart conditional    :    4.180 ms/iter  (1366.16x baseline)

Conclusion: ``.copy()`` to silence the warning is 1500x-21000x slower than the direct
path on production shapes. The correct fix is to suppress the specific UserWarning at
the call site (see neural/base.py:to_tensor_any). This benchmark is kept so the
trade-off can be re-measured on different hardware / numpy / torch versions.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


def bench(shape, n_iter: int = 20) -> None:
    rng = np.random.default_rng(42)
    arr_writable = rng.normal(size=shape).astype(np.float32)
    arr_nonwritable = arr_writable.copy()
    arr_nonwritable.setflags(write=False)

    # Strategy A: from_numpy directly (warning fires on non-writable, no copy)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        t = torch.from_numpy(arr_nonwritable)
        _ = t.float()
    t_direct = (time.perf_counter() - t0) / n_iter

    # Strategy B: .copy() first (no warning, but pays the memcpy)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        t = torch.from_numpy(arr_nonwritable.copy())
        _ = t.float()
    t_copy = (time.perf_counter() - t0) / n_iter

    # Strategy C: np.ascontiguousarray (only copies if needed)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        a = arr_nonwritable
        if not a.flags.writeable:
            a = np.ascontiguousarray(a)
        t = torch.from_numpy(a)
        _ = t.float()
    t_smart = (time.perf_counter() - t0) / n_iter

    # Strategy D: writable input baseline
    t0 = time.perf_counter()
    for _ in range(n_iter):
        t = torch.from_numpy(arr_writable)
        _ = t.float()
    t_writable = (time.perf_counter() - t0) / n_iter

    nbytes_mb = arr_writable.nbytes / 1e6
    print(f"shape={shape} ({nbytes_mb:.1f} MB):")
    print(f"  writable baseline    : {t_writable*1000:8.3f} ms/iter")
    print(f"  non-writable direct  : {t_direct*1000:8.3f} ms/iter  (fires warning)")
    print(f"  non-writable .copy() : {t_copy*1000:8.3f} ms/iter  ({t_copy/t_writable:.2f}x baseline)")
    print(f"  smart conditional    : {t_smart*1000:8.3f} ms/iter  ({t_smart/t_writable:.2f}x baseline)")
    print()


if __name__ == "__main__":
    for shape in [(10_000, 50), (100_000, 50), (1_000_000, 50)]:
        bench(shape, n_iter=20 if shape[0] < 500_000 else 5)
