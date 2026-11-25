"""The family-wall ledger the fitted estimator carries (``fe_family_wall_``), the budget wall keys and the FE report type, on a 30000 x 40 all-families fit: a quick check that the ledger is filled.

The protocol, history and target of the benchmark fit are in ``all_families_wall``.

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
from typing import Any, cast
import os,warnings
warnings.simplefilter("ignore")
for k,v in {"MLFRAME_FE_GPU_STRICT":"1","MLFRAME_CMI_GPU":"1","MLFRAME_FE_VRAM_F32":"1","MLFRAME_FE_GPU_DISCRETIZE":"1","MLFRAME_FE_GPU_BINNING":"1"}.items(): os.environ.setdefault(k,v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters._fe_family_timing import reset_fe_family_wall
kw=F.all_families_kwargs(); kw["binning_scheme"] = "mdlp"
reset_fe_family_wall()
sel=MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(*F._make(30000,40,0))
w=sel.fe_family_wall_
print("ledger keys:",len(w))
for k,r in w.items(): print("  %6.2fs x%-3d %s"%(r["wall_s"],r["calls"],k))
print("fe_family_budget_ wall keys:",sorted((getattr(sel,"fe_family_budget_",None) or {}).get("wall",{}).keys())[:5])
rep=sel.get_fe_report() if hasattr(sel,"get_fe_report") else None
print("report type:",type(rep).__name__, (list(rep.keys())[:12] if isinstance(rep,dict) else (list(cast(Any, rep).columns)[:12] if hasattr(rep,"columns") else "")))
