"""The same sampler aimed at the idle stretches of the all-families fit: which stages and call chains the main thread is in while the mean CPU load is below 25 % AND the GPU below 10 %, the load per stage,
the chains of a target stage (``TARGET``, ``TARGET_STACK``) and the peak memory-pool use by stage.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.all_families_idle_frames N P [cpu|gpu]``; env ``TARGET`` (default ``_stage_binned_numeric_agg``) and ``TARGET_STACK``.
The protocol, history and target of the fit are in ``all_families_wall``.

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
import os,sys,threading,time,warnings,collections
import numpy as np
warnings.simplefilter("ignore")
mode=sys.argv[3] if len(sys.argv)>3 else "gpu"
if mode=="gpu":
    for k,v in {"MLFRAME_FE_GPU_STRICT":"1","MLFRAME_CMI_GPU":"1","MLFRAME_FE_VRAM_F32":"1","MLFRAME_FE_GPU_DISCRETIZE":"1","MLFRAME_FE_GPU_BINNING":"1"}.items(): os.environ.setdefault(k,v)
else: os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
n=int(sys.argv[1]); p=int(sys.argv[2])
kw=F.all_families_kwargs(); kw["nbins_strategy"]="mdlp"
MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*F._make(20000,30,1))
data=F._make(n,p,0)
main=threading.get_ident(); stop=threading.Event(); hist=collections.Counter(); leaf=collections.Counter()
PRE=("_stage_","hybrid_","screen_predictors","_run_fe_step","check_prospective_fe_pairs","_fe_stage_cascade_","_gate_engineered_accuracy","_second_pass","_emit_pair","_score_one_pair","family_pass","compute_pair_mis","build_fe_operand_pool","apply_surrogate","propose_gradient","mrmr_fit_post","_fit_body","_finalise","populate_fe","_apply_pair_family_gate","_adaptive_relax","_materialise_and_fina","filter_inert","honest_","_cross_stage","_temporal","generate_","score_","fit_binned")
def stage(fr):
    best=None
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
chain=collections.Counter()
def in_stack(fr):
    t=os.environ.get('TARGET_STACK')
    while t and fr is not None:
        if fr.f_code.co_name==t: return True
        fr=fr.f_back
    return False
def chain_of(fr):
    out=[]
    while fr is not None and len(out)<5:
        if 'filters' in fr.f_code.co_filename or 'mlframe' in fr.f_code.co_filename: out.append(fr.f_code.co_name+':'+str(fr.f_lineno))
        fr=fr.f_back
    return ' < '.join(out)
import subprocess, psutil
def gpu_pct():
    try:
        return float(subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5).stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")
cmem={}; mem={}; TARGET=os.environ.get('TARGET','_stage_binned_numeric_agg'); tchain=collections.Counter(); lowc=collections.Counter(); agg=collections.defaultdict(lambda:np.zeros(3)); idle_chain=collections.Counter(); idle_stage=collections.Counter(); nsamp=[0,0]
def sampler():
    psutil.cpu_percent(None,percpu=True)
    while not stop.is_set():
        stop.wait(0.1)
        fr=sys._current_frames().get(main)
        cpu=float(np.mean(psutil.cpu_percent(None,percpu=True))); g=gpu_pct()
        nsamp[0]+=1
        try:
            import cupy as _cp; _p=_cp.get_default_memory_pool(); _u=_p.used_bytes(); _t=_p.total_bytes(); st=stage(fr); mem[st]=max(mem.get(st,(0,0)),(_u,_t))
            if in_stack(fr): ck=chain_of(fr); cmem[ck]=max(cmem.get(ck,0),_u)
        except Exception: pass
        agg[stage(fr)]+=[1,cpu,g]
        if cpu<35: lowc[chain_of(fr)]+=1
        if stage(fr)==TARGET or in_stack(fr): tchain[chain_of(fr)]+=1
        if cpu<25 and g<10:
            nsamp[1]+=1; idle_stage[stage(fr)]+=1; idle_chain[chain_of(fr)]+=1
threading.Thread(target=sampler,daemon=True).start()
t=time.perf_counter(); m=MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*data); wall=time.perf_counter()-t; stop.set()
print("WALL",round(wall,1),mode, "samples",nsamp)
print("--- stages while both idle"); [print(f"{v:5d}  {k}") for k,v in idle_stage.most_common(14)]
print("--- chains while both idle"); [print(f"{v:5d}  {k}") for k,v in idle_chain.most_common(18)]

print("--- per stage: samples avgCPU%% avgGPU%%")
for k,v in sorted(agg.items(),key=lambda kv:-kv[1][0])[:22]: print(f"{int(v[0]):4d} cpu{v[1]/v[0]:5.0f} gpu{v[2]/v[0]:5.0f}  {k}")

print("--- target chains");[print(f"{v:4d} {k}") for k,v in tchain.most_common(30)]

print("--- low-cpu chains");[print(f"{v:4d} {k}") for k,v in lowc.most_common(25)]

print("--- peak pool used/total MB by stage")
for k,(u,t) in sorted(mem.items(),key=lambda kv:-kv[1][0])[:14]: print("%6.0f %6.0f  %s"%(u/1e6,t/1e6,k))

print("--- peak used MB by chain (TARGET_STACK)")
for k,u in sorted(cmem.items(),key=lambda kv:-kv[1])[:12]: print("%6.0f  %s"%(u/1e6,k[:170]))
