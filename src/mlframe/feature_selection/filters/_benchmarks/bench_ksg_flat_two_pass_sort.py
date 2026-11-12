"""Bench of a rejected attempt: sort the block of a KSG batch as TWO flat stable passes instead of one sort per column.

The device MI of the KSG batch (``_ksg_cd_gpu.mi_cd_columns_device``) sorts each of its columns with a Python loop of cub radix sorts (``device_argsort``), about 1200 kernel launches per call at 100 columns. The attempt:
sort all columns at once as a flat array, in two stable passes (the low 32 bits of the order-preserving image of each double, then the high 32 bits together with the column number), which removes the launches. It is correct (the order
equals numpy's stable argsort, duplicates included) but measured on a GTX 1050 Ti with 100 columns: 107 ms -> 90 ms at 100000 rows and 32 ms -> 28 ms at 30000 rows (14 to 16 percent), because the sort is bound by the device's
memory traffic over about 10 million 64-bit keys, not by launches. Not adopted: the gain does not pay for the extra arrays and the dependence on cupy's sort being stable.

Run: ``PYTHONPATH=src python src/mlframe/feature_selection/filters/_benchmarks/bench_ksg_flat_two_pass_sort.py``
"""

import time
import numpy as np, cupy as cp
from mlframe.feature_selection.filters._device_sort_safe import device_argsort

def sync(): cp.cuda.Stream.null.synchronize()
def best(f, r=5):
    f(); sync(); ts = []
    for _ in range(r):
        sync(); t0 = time.perf_counter(); out = f(); sync(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1000, out

def flat_order(x):
    b, n = x.shape
    bits = x.view(cp.uint64)
    ordered = bits ^ cp.where((bits >> cp.uint64(63)) == 1, cp.uint64(0xFFFFFFFFFFFFFFFF), cp.uint64(0x8000000000000000))
    lo = (ordered & cp.uint64(0xFFFFFFFF)).astype(cp.uint32).ravel()
    p1 = cp.argsort(lo)
    hi_col = ((cp.arange(b, dtype=cp.uint64)[:, None] << cp.uint64(32)) | (ordered >> cp.uint64(32))).ravel()[p1]
    p2 = cp.argsort(hi_col)
    return (p1[p2] % n).reshape(b, n)

# 1) stability / correctness on data with heavy duplicates and signed values
rng = np.random.default_rng(0)
x = rng.integers(-5, 5, size=(7, 5000)).astype(np.float64) * 0.25
want = np.stack([np.argsort(row, kind="stable") for row in x])
got = cp.asnumpy(flat_order(cp.asarray(x)))
print("CHECK stable-equal on duplicates:", np.array_equal(want, got), "| sorted values equal:", np.array_equal(np.take_along_axis(x, got, 1), np.take_along_axis(x, want, 1)))
xr = rng.standard_normal((9, 4000)); got2 = cp.asnumpy(flat_order(cp.asarray(xr)))
print("CHECK random values: order identical to numpy:", np.array_equal(got2, np.stack([np.argsort(r, kind='stable') for r in xr])))

for n in (30_000, 100_000):
    xd = cp.asarray(rng.standard_normal((100, n)))
    t_loop, o1 = best(lambda: device_argsort(xd, axis=1))
    t_flat, o2 = best(lambda: flat_order(xd))
    same = bool((cp.take_along_axis(xd, o1, 1) == cp.take_along_axis(xd, o2, 1)).all())
    print(f"TIME n={n:6d} b=100: per-column loop={t_loop:6.1f} ms | flat two-pass={t_flat:6.1f} ms | same sorted values: {same}")
