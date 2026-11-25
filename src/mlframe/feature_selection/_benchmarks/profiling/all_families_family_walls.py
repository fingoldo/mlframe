"""Per-family wall ledger of the all-families fit (``fe_family_wall``), longest 30 first, and the selected features.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.all_families_family_walls N P [cpu|gpu]``. The ledger is a snapshot copy, so the rows are the walls of the measured fit only. Laps run
until the next gate: a family whose gate is followed by a gate-less tail is billed that tail (a 13.7 s stage showed as a 65 s lap), so use the sums for relative comparisons and
``all_families_stage_sampler`` for the real wall of a stage. The protocol, history and target of the fit are in ``all_families_wall``.

Companions (all in this package, same protocol: a warm-up fit first, then the measured fit; run one at a time, nothing else heavy beside it):

- ``all_families_wall``            this file: the wall of the benchmark fit and the per-stage walls of the stage-concurrency scheduler.
- ``all_families_stage_sampler``   main-thread sampler: seconds per innermost FE / screen stage and per innermost frame, GPU busy and CPU load per stage (the occupancy table).
- ``all_families_idle_frames``     the same sampler aimed at the stretches where CPU < 25 % and GPU < 10 % (who waits), with the call chains and the pool memory by stage.
- ``all_families_family_walls``    per-family wall ledger (``fe_family_wall``) of the same fit, longest first, plus the selected features.
- ``all_families_fit_ledger``      the ledger the fitted estimator carries (``fe_family_wall_``, 30000 x 40), a quick sanity probe.
- ``all_families_selection_quality``  held-out AUC (LightGBM, logistic regression, all raw columns) of the selected features: judge a GPU / CPU selection difference by this, not by names.
- ``numba_signature_counts``       numba dispatchers with more than one compiled signature after two fits (recompiles inside a fit).
- ``env_sampled_fit``, ``stack_timeline``, ``utilization``, ``tick_utilization``, ``family_profile``, ``family_scale``: older siblings of the same fit (system sampler, 6 s buckets, idle share, one family at a time).
"""
from typing import Any
import os,sys,warnings,time
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
mode=sys.argv[3] if len(sys.argv)>3 else "gpu"
if mode=="gpu":
    for k,v in {"MLFRAME_FE_GPU_STRICT":"1","MLFRAME_CMI_GPU":"1","MLFRAME_FE_VRAM_F32":"1","MLFRAME_FE_GPU_DISCRETIZE":"1","MLFRAME_FE_GPU_BINNING":"1"}.items(): os.environ.setdefault(k,v)
else: os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from mlframe.feature_selection.filters import _fe_family_timing as FT
from mlframe.feature_selection._benchmarks.profiling.family_profile import _SnapshotLedger
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
n=int(sys.argv[1]); p=int(sys.argv[2])
kw: dict[str, Any]={f:True for f in F._flag_names()}; kw["binning_scheme"] = "mdlp"
MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*F._make(20000,30,1))
led=_SnapshotLedger(); FT._FE_FAMILY_WALL=led; FT.reset_fe_family_wall()
data=F._make(n,p,0)
t=time.perf_counter(); m=MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*data); wall=time.perf_counter()-t
print("WALL",round(wall,1),"mode",mode)
print("SELECTED",[str(s) for s in m.get_feature_names_out()])
w=led.last or {k:(v[0],int(v[1])) for k,v in led.items()}
tot=0
for k,(sec,_c) in sorted(w.items(),key=lambda kv:-kv[1][0])[:30]:
    print(f"{sec:7.2f}s  {k}")
