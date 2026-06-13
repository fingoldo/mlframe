import sys, time
sys.modules['cupy'] = None
import numba, numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def aucs_with_ks(y_true, y_score, desc):
    n = len(desc)
    total_pos = 0.0
    for i in range(n):
        total_pos += y_true[desc[i]]
    total_neg = n - total_pos
    if total_pos == 0 or total_neg == 0:
        return np.nan, np.nan, np.nan
    last_fps=0; last_tps=0; tps=0; fps=0; roc=0.0
    prev_recall=0.0; pr=0.0; ks=0.0
    inv_pos=1.0/total_pos; inv_neg=1.0/total_neg
    for i in range(n):
        idx=desc[i]
        yt=y_true[idx]
        tps += yt
        fps += 1-yt
        if i==n-1 or y_score[desc[i+1]]!=y_score[idx]:
            roc += (fps-last_fps)*(last_tps+tps)
            last_fps=fps; last_tps=tps
            cp=tps/(tps+fps) if (tps+fps)>0 else 0.0
            cr=tps/total_pos
            pr += (cr-prev_recall)*cp
            prev_recall=cr
            d=tps*inv_pos - fps*inv_neg
            if d<0: d=-d
            if d>ks: ks=d
    denom=tps*fps*2
    roc = roc/denom if denom>0 else np.nan
    return roc, pr, ks

from mlframe.metrics._core_auc_brier import _argsort_desc_for_metrics, fast_numba_aucs
from mlframe.metrics.classification._classification_extras import ks_statistic

rng=np.random.default_rng(1)
for n in (1000, 50000, 1_000_000):
    yp=rng.beta(2,5,n)
    yt=(rng.random(n)<yp).astype(np.float64)
    desc=_argsort_desc_for_metrics(yp)
    roc,pr,ks=aucs_with_ks(yt,yp,np.ascontiguousarray(desc))
    roc0,pr0=fast_numba_aucs(yt,yp,desc)
    ks0=ks_statistic(yt.astype(np.int64),yp,desc_order=desc)
    print(f"n={n}: dROC={abs(roc-roc0):.2e} dPR={abs(pr-pr0):.2e} dKS={abs(ks-ks0):.2e}")

# tied/discrete check
yp=np.round(rng.beta(2,5,200000),2); yt=(rng.random(200000)<yp).astype(np.float64)
desc=_argsort_desc_for_metrics(yp)
roc,pr,ks=aucs_with_ks(yt,yp,np.ascontiguousarray(desc))
roc0,pr0=fast_numba_aucs(yt,yp,desc); ks0=ks_statistic(yt.astype(np.int64),yp,desc_order=desc)
print(f"tied: dROC={abs(roc-roc0):.2e} dPR={abs(pr-pr0):.2e} dKS={abs(ks-ks0):.2e}")

# timing at 1M
yp=rng.beta(2,5,1_000_000); yt=(rng.random(1_000_000)<yp).astype(np.float64)
desc=_argsort_desc_for_metrics(yp); descc=np.ascontiguousarray(desc)
def best(fn,k=7):
    ts=[]
    for _ in range(k):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000
aucs_with_ks(yt,yp,descc)
print("fused auc+ks :", round(best(lambda: aucs_with_ks(yt,yp,descc)),2))
print("auc only     :", round(best(lambda: fast_numba_aucs(yt,yp,desc)),2))
print("ks separate  :", round(best(lambda: ks_statistic(yt.astype(np.int64),yp,desc_order=desc)),2))
