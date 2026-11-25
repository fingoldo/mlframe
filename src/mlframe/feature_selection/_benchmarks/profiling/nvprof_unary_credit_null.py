"""Run under ``nvprof --profile-from-start off`` (add ``--metrics ...`` for counters): one warm null batch of the logit credit at its real shape (10000 rows, 13 candidates + the zero row, 50 draws): draw the classes, the fused kernel, the copy back; cupy profiler window for nvprof."""
import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _unary_credit_logit_cuda as kc

rng = np.random.default_rng(0)
n, K, D = 10000, 13, 50
z = rng.standard_normal((K + 1, n)).astype(np.float32); z[-1] = 0
offset = (0.5 * rng.standard_normal(n)).astype(np.float32)
p = (1 / (1 + np.exp(-offset))).astype(np.float32)
od, zd, pd = cp.asarray(offset), cp.asarray(z), cp.asarray(p)
gen = cp.random.default_rng(1)


def batch():
    y = kc.draw_classes(cp, gen, pd, D)
    ll = kc.loglik_device(cp, od, zd, y, 5)
    lrt = 2.0 * (ll[:-1] - ll[-1][None, :])
    return cp.asnumpy(lrt[1:] - lrt[0][None, :])


for _ in range(3):
    batch()
cp.cuda.Device().synchronize()
cp.cuda.profiler.start()
t0 = time.perf_counter()
for _ in range(10):
    out = batch()
print(f"batch wall {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms, out {out.shape}")
cp.cuda.profiler.stop()
