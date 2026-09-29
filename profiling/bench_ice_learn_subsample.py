"""Per-call cost of ICE.evaluate on a 555k-row learn set: legacy sentinel skip vs fixed subsample vs full compute."""
import time, numpy as np
from mlframe.training._picklable_metrics import IntegralCalibrationError
from mlframe.metrics._ice_metric import ICE
ice=IntegralCalibrationError(method="multicrit",mae_weight=3,std_weight=2,brier_loss_weight=0.8,roc_auc_weight=1.5,pr_auc_weight=0.1,min_roc_auc=0.54,roc_auc_penalty=0.0,use_weighted_calibration=True,weight_by_class_npositives=False,nbins=10)
rng=np.random.default_rng(0); NL=555_000; NV=62_000
def mk(n):
    lg=rng.normal(size=n); return lg,(rng.random(n)<1/(1+np.exp(-lg))).astype(np.float64)
L=mk(NL); V=mk(NV)
res={}
for name,kw in [("legacy",dict(subsample_skipped_sets=False)),("sub25k",dict(learn_sample_size=25_000)),("sub50k",dict()),("sub100k",dict(learn_sample_size=100_000)),("full",dict(learn_sample_size=10**9))]:
    m=ICE(metric=ice,higher_is_better=False,skip_largest_set=True,**kw)
    m.evaluate((L[0],),L[1],None); m.evaluate((V[0],),V[1],None); m.evaluate((L[0],),L[1],None)
    ts=[]
    for _ in range(30):
        s=time.perf_counter(); v=m.evaluate((L[0],),L[1],None)[0]; ts.append(time.perf_counter()-s)
    res[name]=(np.median(ts)*1e3,min(ts)*1e3,v)
full=res["full"][2]
for k,(med,mn,v) in res.items(): print(f"{k:8s} median {med:7.2f} ms  min {mn:7.2f} ms  learn ICE {v:.5f}  |diff vs full| {abs(v-full):.5f}")
