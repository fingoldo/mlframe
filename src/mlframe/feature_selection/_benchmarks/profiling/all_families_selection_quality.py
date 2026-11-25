"""Held-out quality of the features an all-families fit selects: test AUC of a LightGBM, of a logistic regression on them, and of a LightGBM on all raw columns, for the CPU or the GPU path.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.all_families_selection_quality cpu|gpu``. The GPU and CPU paths do not select the same names (the GPU keeps composite expressions and drops
raw columns); judge the difference by these AUCs (100000 x 100 train, 50000 test), which were equal within noise on 2026-11-03 (GPU 0.9020 / 0.9028 against CPU 0.9018 / 0.8958 for lgbm / logreg).
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
import os,sys,time,warnings; warnings.simplefilter("ignore")
mode=sys.argv[1]
if mode=="cpu": os.environ["CUDA_VISIBLE_DEVICES"]="-1"
else:
    for k,v in {"MLFRAME_FE_GPU_STRICT":"1","MLFRAME_CMI_GPU":"1","MLFRAME_FE_VRAM_F32":"1","MLFRAME_FE_GPU_DISCRETIZE":"1","MLFRAME_FE_GPU_BINNING":"1"}.items(): os.environ.setdefault(k,v)
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from mlframe.feature_selection._benchmarks.profiling import family_scale as Fm
from mlframe.feature_selection.filters.mrmr import MRMR
kw=Fm.all_families_kwargs(); kw["binning_scheme"] = "mdlp"
Xtr,ytr=Fm._make(100000,100,0); Xte,yte=Fm._make(50000,100,1)
t=time.perf_counter(); m=MRMR(verbose=0,random_seed=0,n_jobs=1,**kw).fit(Xtr,ytr); wall=time.perf_counter()-t
Ttr=m.transform(Xtr); Tte=m.transform(Xte)
cols=list(Ttr.columns) if hasattr(Ttr,'columns') else []
A=np.asarray(Ttr,dtype=np.float32); B=np.asarray(Tte,dtype=np.float32)
clf=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,verbose=-1,random_state=0).fit(A,ytr)
auc=roc_auc_score(yte,np.asarray(clf.predict_proba(B))[:,1])
raw=[c for c in Xtr.columns if c!='c0']
r=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,verbose=-1,random_state=0).fit(np.asarray(Xtr[raw],dtype=np.float32),ytr)
auc_raw=roc_auc_score(yte,np.asarray(r.predict_proba(np.asarray(Xte[raw],dtype=np.float32)))[:,1])
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=2000).fit(np.nan_to_num(A.astype(float)),ytr); auc_lr=roc_auc_score(yte,lr.predict_proba(np.nan_to_num(B.astype(float)))[:,1])
print("RES",mode,"wall",round(wall,1),"n_sel",Ttr.shape[1],"test AUC lgbm",round(auc,4),"logreg",round(auc_lr,4),"raw-all lgbm",round(auc_raw,4),cols,flush=True)
