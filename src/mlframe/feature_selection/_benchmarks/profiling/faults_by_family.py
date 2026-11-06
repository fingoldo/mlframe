"""Wall seconds of every _stage_* family function of the mid_a / mid_b / early_b cascades in one warm 100k x 100 GPU fit (wrapper timing, no profiler)."""
import os, sys, time, threading, warnings, importlib, functools
import psutil
_me = psutil.Process()
os.environ['MLFRAME_FE_STAGE_CONCURRENCY'] = '0'  # serial stages: the process-wide fault counter then belongs to the running family
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
kw = F.all_families_kwargs(); kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
walls = {}; faults = {}; lock = threading.Lock()
pkg = "mlframe.feature_selection.filters._mrmr_fit_impl"
import pkgutil
mods = importlib.import_module(pkg)
for m in pkgutil.iter_modules(mods.__path__):
    if not m.name.startswith("_fe_stage_cascade"):
        continue
    mod = importlib.import_module(f"{pkg}.{m.name}")
    for name in [n for n in dir(mod) if n.startswith("_stage_") or n.startswith("_fe_stage_cascade")]:
        fn = getattr(mod, name)
        if not callable(fn) or getattr(fn, "__module__", "") != mod.__name__:
            continue
        def wrap(fn=fn, key=f"{m.name}.{name}"):
            @functools.wraps(fn)
            def w(*a, **k):
                t = time.perf_counter(); f0 = _me.memory_info().num_page_faults
                try:
                    return fn(*a, **k)
                finally:
                    with lock:
                        walls[key] = walls.get(key, 0.0) + time.perf_counter() - t
                        faults[key] = faults.get(key, 0) + _me.memory_info().num_page_faults - f0
            return w
        setattr(mod, name, wrap())
data = F._make(int(sys.argv[1]), int(sys.argv[2]), 0)
f_total0 = _me.memory_info().num_page_faults; t = time.perf_counter(); MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); print('TOTAL faults k', (_me.memory_info().num_page_faults - f_total0) // 1000); print("WALL", round(time.perf_counter() - t, 1))
for k, v in sorted(walls.items(), key=lambda kv: -kv[1]):
    print(f"{v:7.2f}  {faults.get(k,0)/1e3:9.0f}k faults  {k}")
