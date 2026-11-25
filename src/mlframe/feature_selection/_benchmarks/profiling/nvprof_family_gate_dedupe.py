"""Run under ``nvprof --profile-from-start off`` (add ``--metrics ...`` for counters): one warm device family gate call on a dumped pair family (``python nvprof_family_gate_dedupe.py family.npz`` with arrays ``codes`` (rows, columns) and ``y``), bracketed by the cupy profiler window."""
import sys
import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _family_maxt_gate_device as dev

if "--no-dedupe" in sys.argv:  # the gate on every column, for the comparison
    dev.DEDUPE_MIN_COLUMNS = 10**9
d = np.load(sys.argv[1])
codes, y = d["codes"], d["y"].astype(np.int64)
n, k = codes.shape
gd = cp.asarray(codes)


class B:
    def __init__(self, a):
        self.dev = a
        self.shape = a.shape


batches = [B(gd[:, i : i + 1]) for i in range(k)]
for _ in range(2):
    dev.family_pass_batches_device(batches, y, 0.01, 9999, 0)
cp.cuda.Device().synchronize()
cp.cuda.profiler.start()
t0 = time.perf_counter()
out = dev.family_pass_batches_device(batches, y, 0.01, 9999, 0)
assert out is not None, "the device route declined the family"
cp.cuda.Device().synchronize()
print(f"wall {time.perf_counter() - t0:.3f}s passed={sum(out)}/{len(out)}")
cp.cuda.profiler.stop()
