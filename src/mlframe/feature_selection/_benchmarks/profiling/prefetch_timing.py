"""Timestamps (seconds from the start of the timed fit) of the surrogate-seeder prefetch: start, the worker's own begin / end, the wait in take_prefetched."""
import os, sys, time, threading, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters import _gbm_seeder_prefetch as P
import mlframe.feature_selection.filters._surrogate_interaction_seeder as S
kw = {f: True for f in F._flag_names()}; kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
data = F._make(100000, 100, 0)
T0 = [0.0]; ev = []
def stamp(name): ev.append((round(time.perf_counter() - T0[0], 1), name, threading.current_thread().name))
orig_fit = S.surrogate_gbm_interaction_seeds
def timed_fit(*a, **k):
    stamp("seeds BEGIN")
    try: return orig_fit(*a, **k)
    finally: stamp("seeds END")
S.surrogate_gbm_interaction_seeds = timed_fit
orig_pf, orig_tk = P.prefetch_surrogate_seeds, P.take_prefetched
def pf(*a, **k):
    stamp("prefetch call"); return orig_pf(*a, **k)
def tk(*a, **k):
    stamp("take_prefetched enter")
    try: return orig_tk(*a, **k)
    finally: stamp("take_prefetched return")
P.prefetch_surrogate_seeds = pf; P.take_prefetched = tk
import mlframe.feature_selection.filters._mrmr_fe_step._step_core as C
if hasattr(C, "prefetch_surrogate_seeds"): C.prefetch_surrogate_seeds = pf
import mlframe.feature_selection.filters._surrogate_interaction_seeder as S2
MRMR._FIT_CACHE.clear(); T0[0] = time.perf_counter()
t = time.perf_counter(); MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); print("WALL", round(time.perf_counter() - t, 1))
for e in ev: print(e)
