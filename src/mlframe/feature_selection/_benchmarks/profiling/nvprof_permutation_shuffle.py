"""Run under ``nvprof --profile-from-start off`` (add ``--metrics ...`` for counters), or alone for the timing and GIL comparison: the sort-free Feistel target shuffles against the thrust argsort of random keys.

``python nvprof_permutation_shuffle.py [nperm] [n]``: times both generators (device idle after the call), measures how much of its free pace a counter thread keeps while each runs
(94 percent while a thread sleeps; near zero while a call holds the GIL), and brackets one warm Feistel call in the cupy profiler window.
"""

import sys
import threading
import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _permutation_null_resident as resident
from mlframe.feature_selection.filters import _permutation_shuffle_cuda as shuf

nperm = int(sys.argv[1]) if len(sys.argv) > 1 else 50
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
rng = np.random.default_rng(0)
y = rng.integers(0, 10, n).astype(np.int32)
d_y = cp.asarray(y)

ticks = {"n": 0}
stop = threading.Event()


def spin() -> None:
    """Count while the GIL is available to this thread."""
    while not stop.is_set():
        ticks["n"] += 1


def wall(fn, reps: int = 7) -> float:
    """Median wall of ``fn`` until the device is idle, with no other thread running (a spinning thread makes every GIL hand-off of the call cost up to a switch interval)."""
    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - s)
    return float(np.median(ts))


generators = {
    f"argsort of float64 keys ({nperm}, {n})": lambda: resident.gen_target_shuffles_cupy_sorted(y, nperm, np.int32, 1),
    f"Feistel shuffles ({nperm}, {n})": lambda: shuf.shuffled_rows(cp, d_y, nperm, 1),
}
walls = {label: wall(fn) for label, fn in generators.items()}
threading.Thread(target=spin, daemon=True).start()
time.sleep(0.3)
t0, n0 = time.perf_counter(), ticks["n"]
time.sleep(0.5)
free = (ticks["n"] - n0) / (time.perf_counter() - t0)


def gil_share(fn, reps: int = 5) -> float:
    """Share of its free pace that the counter thread keeps while ``fn`` runs (about 94 percent during a sleep, near zero while a call holds the GIL)."""
    shares = []
    for _ in range(reps):
        c0, s = ticks["n"], time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        shares.append((ticks["n"] - c0) / (free * (time.perf_counter() - s)))
    return float(np.median(shares))


print(f"{'sleep 0.1 s (releases the GIL)':46s} counter thread at {100 * gil_share(lambda: time.sleep(0.1)):5.1f}% of its free pace")
for label, fn in generators.items():
    print(f"{label:46s} {walls[label] * 1e3:8.2f} ms   counter thread at {100 * gil_share(fn):5.1f}% of its free pace")
stop.set()
cp.cuda.Device().synchronize()
cp.cuda.profiler.start()
shuf.shuffled_rows(cp, d_y, nperm, 2)
cp.cuda.Device().synchronize()
cp.cuda.profiler.stop()
