"""Same-process A/B of releasing an ended stage's cores to the pools of the stages still running (RELEASE_ENDED_STAGES)."""
import os, sys, time, logging, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
kw = {f: True for f in F._flag_names()}; kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
class H(logging.Handler):
    def __init__(s): super().__init__(); s.lines=[]
    def emit(s, r):
        m=r.getMessage()
        if "stage walls" in m: s.lines.append(m)
h=H(); lg=logging.getLogger("mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency"); lg.setLevel(logging.DEBUG); lg.addHandler(h)
data = F._make(int(sys.argv[1]), int(sys.argv[2]), 0)
order = [True, False, True, False, True, False]
for flag in order:
    os.environ["MLFRAME_FE_STAGE_CONCURRENCY"] = "1" if flag else "0"
    MRMR._FIT_CACHE.clear()
    h.lines.clear()
    t = time.perf_counter(); m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); w = time.perf_counter() - t
    print("CONCURRENT" if flag else "SERIAL", round(w, 1), "sel", len(m.get_feature_names_out()), h.lines, flush=True)
