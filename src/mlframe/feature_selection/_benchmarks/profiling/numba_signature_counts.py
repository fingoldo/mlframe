"""Numba dispatchers of mlframe with more than one compiled signature after two all-families fits (20000 x 30, 30000 x 30): a recompile inside a measured fit shows up here.

A tuple argument is typed per length (every new arity recompiles); dtype and float32-versus-float64 variants are disk-cached and compile once per machine. Usage:
``python -m mlframe.feature_selection._benchmarks.profiling.numba_signature_counts``. The protocol, history and target of the benchmark fit are in ``all_families_wall``.

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
import os,gc,warnings; warnings.simplefilter("ignore")
for k,v in {"MLFRAME_FE_GPU_STRICT":"1","MLFRAME_CMI_GPU":"1","MLFRAME_FE_VRAM_F32":"1","MLFRAME_FE_GPU_DISCRETIZE":"1","MLFRAME_FE_GPU_BINNING":"1"}.items(): os.environ.setdefault(k,v)
from numba.core.registry import CPUDispatcher
from mlframe.feature_selection._benchmarks.profiling import family_scale as Fm
from mlframe.feature_selection.filters.mrmr import MRMR
kw=Fm.all_families_kwargs()
for seed,n in ((1,20000),(0,30000)):
    MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*Fm._make(n,30,seed))
rows=[]
for o in gc.get_objects():
    try:
        if type(o) is CPUDispatcher and len(o.signatures)>1 and "mlframe" in (getattr(o.py_func,"__module__","") or ""):
            rows.append((len(o.signatures),o.py_func.__module__+"."+o.py_func.__name__))
    except ReferenceError:
        pass
for r in sorted(set(rows),reverse=True)[:25]: print(r)
