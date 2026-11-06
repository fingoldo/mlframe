"""Wall of every map_on_edge_pool call in one warm GPU fit, with the caller and the time since the fit started."""
import os, sys, time, threading, warnings, functools, inspect
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters import _edge_pool

kw = F.all_families_kwargs()
kw["nbins_strategy"] = os.environ.get("BENCH_NBINS", "mdlp")
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))

rows = []
lock = threading.Lock()
t0 = [0.0]
real = _edge_pool.map_on_edge_pool


@functools.wraps(real)
def wrapped(fn, items, *a, **k):
    frame = inspect.stack()[1]
    start = time.perf_counter() - t0[0]
    t = time.perf_counter()
    try:
        return real(fn, items, *a, **k)
    finally:
        with lock:
            rows.append((start, time.perf_counter() - t, len(items), f"{os.path.basename(frame.filename)}:{frame.function}:{frame.lineno}"))


import mlframe.feature_selection.filters as pkg
import sys as _s
for name, mod in list(_s.modules.items()):
    if name.startswith("mlframe") and getattr(mod, "map_on_edge_pool", None) is real:
        setattr(mod, "map_on_edge_pool", wrapped)
data = F._make(100000, 100, 0)
t0[0] = time.perf_counter()
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
print("WALL", round(time.perf_counter() - t0[0], 1))
for r in sorted(rows):
    print(f"t={r[0]:6.1f}s wall={r[1]:6.2f}s items={r[2]:4d} {r[3]}")
