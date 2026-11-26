"""The binary-target pair null kernel in its logarithm form and its table form, timed against each other and bracketed for nvprof (``--profile-from-start off``, one warm launch inside the profiler window).

``python nvprof_pair_mi_perm_bin.py [n] [bins] [pairs] [mode]`` with ``mode`` one of ``time`` (interleaved rounds of both kernels, min and median, GPU temperature and clocks before and after), ``log`` or ``lut``
(the profiler window around one warm launch of that kernel only: a metrics run replays every launch, so one kernel per run). The kernel and the launch are the production ones
(``_pair_mi_perm_cuda_bin.launch_pairs_binary``; the form is picked by ``MLFRAME_PAIR_PERM_LUT``, which this script sets itself per call).
"""

import os
import subprocess
import sys
import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _pair_mi_perm_cuda_bin as B

n = int(sys.argv[1]) if len(sys.argv) > 1 else 55296
bins = int(sys.argv[2]) if len(sys.argv) > 2 else 48
pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 600
mode = sys.argv[4] if len(sys.argv) > 4 else "time"
K, cols = 25, 400

rng = np.random.default_rng(0)
ii, jj = np.triu_indices(cols, 1)
pick = rng.choice(ii.size, pairs, replace=False)
pa, pb = ii[pick].astype(np.int64), jj[pick].astype(np.int64)
data = rng.integers(0, bins, size=(n, cols), dtype=np.int8)
data[:, 1] = (data[:, 0] + rng.integers(0, 2, n)) % bins  # one real association, so the MI values are not all null
nb = np.full(cols, bins, dtype=np.int64)
y = (rng.random(n) < 0.5).astype(np.int64)
fy = np.bincount(y, minlength=2) / n
y_perms = np.stack([rng.permutation(y) for _ in range(K)])


def call(lut: bool) -> np.ndarray:
    """One production launch of the chosen form, synchronised."""
    os.environ["MLFRAME_PAIR_PERM_LUT"] = "1" if lut else "0"
    out = B.launch_pairs_binary(cp, data, pa, pb, nb, y_perms, fy, 1 << 16)
    cp.cuda.Stream.null.synchronize()
    return out


def gpu_state() -> str:
    """Temperature, SM clock against its maximum, throttle-reason mask and load of the card, as nvidia-smi reports them now."""
    q = "temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,utilization.gpu"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    return out


if mode in ("log", "lut"):
    lut = mode == "lut"
    call(lut)
    call(lut)
    cp.cuda.profiler.start()
    call(lut)
    cp.cuda.profiler.stop()
    print(f"profiled one warm launch of the {mode} kernel: n={n} bins={bins} pairs={pairs} K={K}")
else:
    a, b = call(False), call(True)
    print(f"n={n} bins={bins} pairs={pairs} K={K} kb={B.perms_per_pass(bins * bins, K)}; max |log - table| = {np.abs(a - b).max():.2e} (values up to {np.abs(a).max():.3e})")
    print("gpu before:", gpu_state())
    walls: dict = {False: [], True: []}
    for _ in range(9):
        for lut in (False, True):
            t = time.perf_counter()
            call(lut)
            walls[lut].append(time.perf_counter() - t)
    print("gpu after: ", gpu_state())
    for lut, label in ((False, "logarithm"), (True, "table")):
        w = np.array(walls[lut])
        print(f"{label:9s}: min {w.min() * 1e3:8.1f} ms, median {np.median(w) * 1e3:8.1f} ms ({w.min() / pairs * 1e6:7.1f} us per pair at the min)")
    print(f"ratio of minima {min(walls[False]) / min(walls[True]):.2f}x, of medians {np.median(walls[False]) / np.median(walls[True]):.2f}x")
