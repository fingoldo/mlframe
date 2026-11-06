"""Two consecutive 100000 x 100 every-family GPU fits in one process (the first warms the numba disk cache and the pools); prints both walls and the selection."""
import os, sys, time, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
import mlframe
print("TREE", os.path.dirname(mlframe.__file__), flush=True)
kw = {f: True for f in F._flag_names()}; kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
data = F._make(100000, 100, 0)
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    MRMR._FIT_CACHE.clear()
    t = time.perf_counter(); m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); w = time.perf_counter() - t
    print("FIT", i, round(w, 1), "sel", list(m.get_feature_names_out()), flush=True)
