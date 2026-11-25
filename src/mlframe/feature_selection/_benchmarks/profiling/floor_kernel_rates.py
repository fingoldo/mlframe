"""Throughput of the two order-2 floor kernels (device one-launch kernel, CPU permutation-batched njit) on the same pairs at the 100000-row shape."""
import sys
import time

import numpy as np

from mlframe.feature_selection.filters import _pair_mi_perm_cuda as pc
from mlframe.feature_selection.filters.info_theory import batch_pair_mi_perm_batched
from mlframe.feature_selection.filters.info_theory.shared import joint_cardinality_cap

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
bins = int(sys.argv[2]) if len(sys.argv) > 2 else 12
pairs_n = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
K = 25
rng = np.random.default_rng(0)
cols = 200
data = rng.integers(0, bins, (n, cols)).astype(np.int8)
nb = np.full(cols, bins, dtype=np.int64)
y = (rng.random(n) < 0.5).astype(np.int8)
fy = np.bincount(y, minlength=2) / n
y_perms = np.stack([rng.permutation(y) for _ in range(K)])
pa = rng.integers(0, cols, pairs_n).astype(np.int64)
pb = ((pa + 1 + rng.integers(0, cols - 1, pairs_n)) % cols).astype(np.int64)
cap = joint_cardinality_cap()
print(f"n={n} bins={bins} pairs={pairs_n} K={K}")
out_g = pc.pair_mi_perm_matrix_cuda(data, pa, pb, nb, y_perms, fy, cap)
t0 = time.perf_counter()
out_g = pc.pair_mi_perm_matrix_cuda(data, pa, pb, nb, y_perms, fy, cap)
tg = time.perf_counter() - t0
batch_pair_mi_perm_batched(data, pa[:50], pb[:50], nb, y_perms, fy, cap)
t0 = time.perf_counter()
out_c = batch_pair_mi_perm_batched(data, pa, pb, nb, y_perms, fy, cap)
tc = time.perf_counter() - t0
print(f"device {tg:.2f}s ({pairs_n / tg:8.0f} pairs/s)   cpu {tc:.2f}s ({pairs_n / tc:8.0f} pairs/s)   cpu/device rate {tg / tc:.2f}   max |diff| {float(np.abs(np.asarray(out_g) - np.asarray(out_c)).max()):.2e}")
