"""Bench: np.nanmedian vs np.nanquantile(q=0.5) for compute_member_quality_gate.

iter119 (2026-05-21) discovery: ``np.nanquantile(arr, 0.5, axis=0)`` falls back
to ``apply_along_axis`` which iterates the non-axis dimensions in Python, so
on a (K, N) ensemble payload it makes N 1-D calls. ``np.nanmedian(arr, axis=0)``
uses numpy's dedicated C reduction.

Bench at the c0085 fuzz shape (K=3 members, N=200_000):

    nanquantile q=0.5 : 13_500 ms
    nanmedian         :     49 ms     (~275x)

3-D multilabel (K=3, N=200_000, C=4):

    nanquantile q=0.5 : 54_000 ms
    nanmedian         :    250 ms     (~215x)

Outputs differ by 2.22e-16 (machine epsilon) -- both ignore NaN by
definition, just nanmedian dispatches through the fast partition+sort path
without the per-element apply_along_axis overhead.

Run: ``python profiling/bench_member_gate_nanmedian.py``
"""

import time
import numpy as np

rng = np.random.default_rng(0)
K, N = 3, 200_000
arr = rng.standard_normal((K, N))
# Sprinkle NaNs
nan_mask = rng.random(arr.shape) < 0.01
arr[nan_mask] = np.nan

# Warmup
_ = np.nanquantile(arr, 0.5, axis=0)
_ = np.nanmedian(arr, axis=0)

for _ in range(3):
    t = time.perf_counter()
    out_q = np.nanquantile(arr, 0.5, axis=0)
    print(f'nanquantile q=0.5 : {(time.perf_counter()-t)*1000:.1f}ms')

for _ in range(3):
    t = time.perf_counter()
    out_m = np.nanmedian(arr, axis=0)
    print(f'nanmedian         : {(time.perf_counter()-t)*1000:.1f}ms')

print(f'identical: {np.array_equal(out_q, out_m, equal_nan=True)}')
print(f'max abs diff: {np.nanmax(np.abs(out_q - out_m)):.2e}')

# Also test (K, N, C) shape like ensemble multilabel
arr3 = rng.standard_normal((K, N, 4))
arr3[rng.random(arr3.shape) < 0.01] = np.nan

print()
for _ in range(3):
    t = time.perf_counter()
    o_q = np.nanquantile(arr3, 0.5, axis=0)
    print(f'3D nanquantile : {(time.perf_counter()-t)*1000:.1f}ms')

for _ in range(3):
    t = time.perf_counter()
    o_m = np.nanmedian(arr3, axis=0)
    print(f'3D nanmedian   : {(time.perf_counter()-t)*1000:.1f}ms')

print(f'3D identical: {np.array_equal(o_q, o_m, equal_nan=True)}')
