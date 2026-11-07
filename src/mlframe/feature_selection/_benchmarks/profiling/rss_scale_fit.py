"""Peak resident memory of one MRMR fit at a given row count, with a timeline of the main thread's innermost mlframe function at each new peak.

Usage: rss_scale_fit.py ROWS [COLS] [all|default]; RSS_TICK_S=20 adds a timeline line of the main thread every 20 s. The frame is a float32 DataFrame (the project dtype), the target a noisy function of five columns. The sampler records the
process RSS every 0.2 s; a new peak is attributed to the main thread's innermost mlframe frame, which names the step that allocated.
"""
import os
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
import psutil

warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection.filters.mrmr import MRMR

rows = int(sys.argv[1])
cols = int(sys.argv[2]) if len(sys.argv) > 2 else 80
mode = sys.argv[3] if len(sys.argv) > 3 else "default"
if mode == "all":
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F

    kwargs = F.all_families_kwargs()
    kwargs["nbins_strategy"] = os.environ.get("BENCH_NBINS", "mdlp")
else:
    kwargs = {}

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.standard_normal((rows, cols), dtype=np.float32), columns=[f"x{i}" for i in range(cols)])
signal = X["x0"].to_numpy() * X["x1"].to_numpy() + 0.7 * X["x2"].to_numpy() - 0.5 * X["x3"].to_numpy() ** 2 + 0.3 * X["x4"].to_numpy()
y = pd.Series((signal + 0.5 * rng.standard_normal(rows, dtype=np.float32) > np.median(signal)).astype(np.int8), name="y")
frame_mb = X.memory_usage(deep=False).sum() / 1e6

import tracemalloc

TRACE = os.environ.get("RSS_TRACEMALLOC") == "1"
if TRACE:
    tracemalloc.start(6)
snapshots = []
proc = psutil.Process()
TICK = float(os.environ.get("RSS_TICK_S", "0"))  # seconds between timeline lines of the main thread's step (0: off)
ticks = []
MIN_AVAILABLE = float(os.environ.get("RSS_MIN_AVAILABLE_GB", "1.0")) * 1e9
base = proc.memory_info().rss
peak = [base]
events = []
stop = threading.Event()
main_id = threading.main_thread().ident


def innermost(frame):
    """The innermost mlframe frame of a stack as file:function:line."""
    while frame is not None:
        fn = frame.f_code.co_filename.replace("\\", "/")
        if "/mlframe/" in fn and "rss_scale_fit" not in fn:
            return f"{fn.split('/mlframe/')[-1]}:{frame.f_code.co_name}:{frame.f_lineno}"
        frame = frame.f_back
    return "(outside mlframe)"


def sampler():
    """Track the RSS peak and name the step at each new peak above the previous one by 50 MB."""
    t0 = time.perf_counter()
    last_logged = base
    while not stop.wait(0.2):
        rss = proc.memory_info().rss
        if psutil.virtual_memory().available < MIN_AVAILABLE:
            # the machine is about to page: report what is known and stop the process instead of freezing the laptop
            print(f"ABORTED at t={time.perf_counter() - t0:.1f}s: available memory under {MIN_AVAILABLE / 1e9:.1f} GB, peak RSS growth {(max(peak[0], rss) - base) / 1e6:.0f} MB = {(max(peak[0], rss) - base) / 1e6 / frame_mb:.1f} x the frame", flush=True)
            for when, mb, where in events:
                print(f"  t={when:7.1f}s +{mb:7.0f} MB  {where}", flush=True)
            print(f"  now: {innermost(sys._current_frames().get(main_id))}", flush=True)
            os._exit(3)
        if TICK and (not ticks or time.perf_counter() - t0 - ticks[-1][0] >= TICK):
            ticks.append((time.perf_counter() - t0, (rss - base) / 1e6, innermost(sys._current_frames().get(main_id))))
        if rss > peak[0]:
            peak[0] = rss
            if TRACE and rss - base > float(os.environ.get("RSS_SNAP_FIRST_MB", "600")) * 1e6 and (not snapshots or rss - snapshots[-1][0] > float(os.environ.get("RSS_SNAP_STEP_MB", "400")) * 1e6):
                snap = tracemalloc.take_snapshot()
                snapshots.append((rss, (rss - base) / 1e6, snap.statistics("traceback")[:14]))
            if rss - last_logged > 50e6:
                last_logged = rss
                events.append((time.perf_counter() - t0, (rss - base) / 1e6, innermost(sys._current_frames().get(main_id))))


th = threading.Thread(target=sampler, daemon=True)
th.start()
t = time.perf_counter()
status = "ok"
try:
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kwargs).fit(X, y)
except BaseException as exc:  # reported with the peak: a MemoryError at scale is the finding
    status = f"{type(exc).__name__}: {str(exc)[:120]}"
wall = time.perf_counter() - t
stop.set()
def _rss_growth_mb():
    """The process RSS growth over the start of the script, in MB."""
    return (proc.memory_info().rss - base) / 1e6


print(f"RSS growth right after the fit: {_rss_growth_mb():.0f} MB")
import gc

gc.collect()
print(f"after gc.collect(): {_rss_growth_mb():.0f} MB")
try:
    import cupy as _cp

    _cp.get_default_pinned_memory_pool().free_all_blocks()
    print(f"after the pinned pool is freed: {_rss_growth_mb():.0f} MB")
    _cp.get_default_memory_pool().free_all_blocks()
    print(f"after the device pool is freed: {_rss_growth_mb():.0f} MB")
except Exception as exc:  # no cupy
    print("pool release unavailable:", exc)
try:
    import cupy as _cp

    _pin = _cp.get_default_pinned_memory_pool()
    _dev = _cp.get_default_memory_pool()
    print(f"after the fit: pinned host pool {_pin.n_free_blocks()} free blocks; device pool used {_dev.used_bytes() / 1e6:.0f} MB, total {_dev.total_bytes() / 1e6:.0f} MB")
except Exception as exc:  # no cupy
    print("pool stats unavailable:", exc)
print(f"rows={rows} cols={cols} mode={mode} frame={frame_mb:.0f} MB status={status} wall={wall:.1f}s")
print(f"peak RSS growth {(peak[0] - base) / 1e6:.0f} MB = {(peak[0] - base) / 1e6 / frame_mb:.2f} x the frame; RSS before the fit {base / 1e6:.0f} MB")
for when, mb, where in events:
    print(f"  t={when:7.1f}s +{mb:7.0f} MB  {where}")
if ticks:
    print("--- timeline of the main thread")
    for when, mb, where in ticks:
        print(f"  tick t={when:7.1f}s +{mb:7.0f} MB  {where}")

for rss, mb, stats in snapshots[-2:]:
    print(f"--- tracemalloc at RSS growth {mb:.0f} MB: the largest live allocations")
    for st in stats:
        print(f"  {st.size / 1e6:8.0f} MB in {st.count} blocks")
        for line in st.traceback.format()[-4:]:
            print("     ", line.strip()[:150])
