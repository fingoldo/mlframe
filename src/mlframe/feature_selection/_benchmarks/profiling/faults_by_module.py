import types
"""Inclusive wall seconds of every function of the given modules in one warm GPU fit (wrapper timing, all threads summed). Usage: module_walls.py n p mod1 mod2 ..."""
import os, sys, time, threading, warnings, importlib, functools, types
import psutil
_me = psutil.Process()
os.environ['MLFRAME_FE_STAGE_CONCURRENCY'] = '0'  # serial stages: the process-wide fault counter then belongs to the running function
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
kw = F.all_families_kwargs(); kw["nbins_strategy"] = os.environ.get("BENCH_NBINS", "mdlp")
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
walls = {}; calls = {}; stats = types.SimpleNamespace(faults={}); lock = threading.Lock()
base = "mlframe.feature_selection.filters."
for name in sys.argv[3:]:
    mod = importlib.import_module(base + name)
    for attr in [a for a in dir(mod) if not a.startswith("__")]:
        fn = getattr(mod, attr)
        if isinstance(fn, types.FunctionType) and fn.__module__ == mod.__name__:
            def wrap(fn=fn, key=f"{name}.{attr}"):
                @functools.wraps(fn)
                def w(*a, **k):
                    t = time.perf_counter(); f0 = _me.memory_info().num_page_faults
                    try: return fn(*a, **k)
                    finally:
                        import cupy as _cp
                        _cp.cuda.Stream.null.synchronize()  # the timer stops at completion, not at the launch of the last kernel
                        dt = time.perf_counter() - t
                        with lock: walls[key] = walls.get(key, 0.0) + dt; calls[key] = calls.get(key, 0) + 1; stats.faults[key] = stats.faults.get(key, 0) + _me.memory_info().num_page_faults - f0
                return w
            setattr(mod, attr, wrap())
data = F._make(int(sys.argv[1]), int(sys.argv[2]), 0)
t = time.perf_counter(); MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); print("WALL", round(time.perf_counter() - t, 1))
for k, v in sorted(walls.items(), key=lambda kv: -stats.faults.get(kv[0], 0))[:25]: print(f"{v:7.2f} {calls[k]:6d} {stats.faults.get(k,0)/1e3:8.0f}k faults  {k}")
