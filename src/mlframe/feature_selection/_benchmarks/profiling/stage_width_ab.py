"""Same-process A/B of releasing an ended stage's cores to the pools of the stages still running.

Needs the experimental RELEASE_ENDED_STAGES flag of _edge_pool (not in the tree: the experiment was reverted, see stage_sample_100k_gpu_2026_11_05.md)."""
from typing import Any, cast
import os, sys, time, logging, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters import _edge_pool as E
kw: dict[str, Any] = {f: True for f in F._flag_names()}; kw["binning_scheme"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
class H(logging.Handler):
    def __init__(self): super().__init__(); self.lines=[]
    def emit(self, r):
        m=r.getMessage()
        if "stage walls" in m: self.lines.append(m)
h=H(); lg=logging.getLogger("mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency"); lg.setLevel(logging.DEBUG); lg.addHandler(h)
data = F._make(int(sys.argv[1]), int(sys.argv[2]), 0)
order = [True, False, True, False] if len(sys.argv) < 4 else [x == "1" for x in sys.argv[3].split(",")]
for flag in order:
    cast(Any, E).RELEASE_ENDED_STAGES = flag
    MRMR._FIT_CACHE.clear()
    h.lines.clear()
    t = time.perf_counter(); m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); w = time.perf_counter() - t
    print("RELEASE" if flag else "FIXED", round(w, 1), "sel", len(m.get_feature_names_out()), h.lines, flush=True)
