"""Main-thread sampler of the all-families 100000 x 100 fit (no profiler): seconds per innermost FE / screen stage frame and per innermost Python frame, and per stage the mean GPU busy % and CPU %.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.all_families_stage_sampler N P [cpu|gpu]``; env ``SS_NBINS`` (default mdlp) and ``SS_FOCUS=<stage>,...`` (leaf frames of those stages).
This is the occupancy table the work on the near-100 % CPU / GPU goal reads: the ``UTIL`` lines (mean, median, time below the thresholds) and the last table (stages over 1 s with their GPU and CPU load).
Replaces the older ``stage_sample*.py`` drafts. The protocol, history and target of the fit are in ``all_families_wall``.

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
import os,sys,threading,time,warnings,collections
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
mode=sys.argv[3] if len(sys.argv)>3 else "gpu"
if mode=="gpu":
    for k,v in {"MLFRAME_FE_GPU_STRICT":"1","MLFRAME_CMI_GPU":"1","MLFRAME_FE_VRAM_F32":"1","MLFRAME_FE_GPU_DISCRETIZE":"1","MLFRAME_FE_GPU_BINNING":"1"}.items(): os.environ.setdefault(k,v)
else: os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
n=int(sys.argv[1]); p=int(sys.argv[2])
kw=F.all_families_kwargs(); kw["binning_scheme"] = "mdlp"
MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*F._make(20000,30,1))
data=F._make(n,p,0)
main=threading.get_ident(); stop=threading.Event(); hist: Any=collections.Counter(); leaf: Any=collections.Counter(); both: Any=collections.Counter()
PRE=("_stage_","hybrid_","screen_predictors","_run_fe_step","check_prospective_fe_pairs","_fe_stage_cascade_","_gate_engineered_accuracy","_second_pass","_emit_pair","_score_one_pair","family_pass","compute_pair_mis","build_fe_operand_pool","apply_surrogate","propose_gradient","mrmr_fit_post","_fit_body","_finalise","populate_fe","_apply_pair_family_gate","_adaptive_relax","_materialise_and_fina","filter_inert","honest_","_cross_stage","_temporal","generate_","score_","fit_binned")
def stage(fr):
    best = None
    while fr is not None:
        nm=fr.f_code.co_name
        if nm.startswith(PRE) and best is None: best=nm
        fr=fr.f_back
    return best
def inner(fr):
    # leaf frame plus the nearest mlframe function above it
    g=fr
    while g is not None and 'filters' not in g.f_code.co_filename: g=g.f_back
    near=(g.f_code.co_name+':'+str(g.f_lineno)) if g is not None else '?'
    return (f"{fr.f_code.co_filename.split('Lib')[-1].split('filters')[-1][-26:]}:{fr.f_code.co_name}  <- {near}" if fr is not None else None)
import psutil, pynvml
pynvml.nvmlInit(); _gh=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
util: Any = []  # (t, cpu% all cores, [gpu busy %], [gpu mem used GB])
def hw_sampler():
    psutil.cpu_percent(interval=None)
    while not stop.is_set():
        stop.wait(0.2)
        _fr=sys._current_frames().get(main); util.append((time.perf_counter(), psutil.cpu_percent(interval=None), [pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in _gh], [pynvml.nvmlDeviceGetMemoryInfo(h).used/2**30 for h in _gh], stage(_fr) if _fr is not None else None))
threading.Thread(target=hw_sampler,daemon=True).start()
def sampler():
    while not stop.is_set():
        fr=sys._current_frames().get(main)
        st_=stage(fr); hist[st_]+=0.2; lf_=inner(fr); leaf[lf_]+=0.2; both[(st_,lf_)]+=0.2
        stop.wait(0.2)


threading.Thread(target=sampler, daemon=True).start()
t = time.perf_counter()
m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
wall = time.perf_counter() - t
stop.set()
print("WALL", round(wall, 1), mode)
print("--- seconds by innermost stage frame")
[print(f"{v:7.1f}  {k}") for k, v in hist.most_common(26)]
print("--- seconds by innermost python frame (where the main thread is waiting/working)")
[print(f"{v:7.1f}  {k}") for k, v in leaf.most_common(26)]

FOCUS = os.environ.get("SS_FOCUS", "").split(",")
for f in FOCUS:
    if not f: continue
    print("--- leaves in stage",f)
    # Slice before printing: the comprehension printed every matching leaf and only then cut to 12 (13-F35).
    for k, v in [item for item in both.most_common() if item[0][0] == f][:12]:
        print(f"{v:7.1f}  {k[1]}")

if util:
    import numpy as _np
    cpu=_np.array([u[1] for u in util]); gpu=_np.array([u[2] for u in util]); mem=_np.array([u[3] for u in util])
    print(f"UTIL samples={len(util)} cpu_mean={cpu.mean():.1f}% cpu_p50={_np.median(cpu):.1f}% cpu_time_below_50%={(cpu<50).mean()*100:.0f}%")
    for d in range(gpu.shape[1]):
        print(f"UTIL gpu{d} busy_mean={gpu[:,d].mean():.1f}% busy_p50={_np.median(gpu[:,d]):.0f}% time_below_10%={(gpu[:,d]<10).mean()*100:.0f}% mem_peak={mem[:,d].max():.2f}GB")

if util:
    import collections as _c
    per=_c.defaultdict(list)
    for u in util: per[u[4]].append((u[1],u[2][0]))
    print("--- per stage: seconds, mean GPU busy %, mean CPU % (stages over 1 s)")
    rows=sorted(((len(v)*0.2, sum(g for _,g in v)/len(v), sum(c for c,_ in v)/len(v), k) for k,v in per.items()), reverse=True)
    for sec,g,c,k in rows:
        if sec>=1.0: print(f"{sec:6.1f}s gpu {g:5.1f}% cpu {c:5.1f}%  {k}")
