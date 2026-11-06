import time, numpy as np
from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused as M
rng=np.random.default_rng(0); n,nf,nb=100_000,120,12
f=rng.integers(0,nb,(n,nf)).astype(np.int32); y=rng.integers(0,2,n).astype(np.int32)
nbins=np.full(nf,nb,dtype=np.int32); pa=rng.integers(0,nf,3000).astype(np.int64); pb=((pa+rng.integers(1,nf,3000))%nf).astype(np.int64)
def old():
    for col in np.unique(np.concatenate([pa,pb])):
        c=int(col); v=f[:,c]
        lo,hi=int(v.min()),int(v.max())
        if lo<0 or hi>=int(nbins[c]): raise ValueError
for fn,name in ((old,"old per-column"),(lambda: M.validate_pair_codes(f,pa,pb,nbins,y,2),"new one pass")):
    fn(); ts=[]
    for _ in range(5):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    print(f"{name}: median {np.median(ts)*1e3:.1f} ms")
bad=f.copy(); bad[5,int(pa[0])]=-1
try: M.validate_pair_codes(bad,pa,pb,nbins,y,2); print("NO RAISE")
except ValueError as e: print("raised:",str(e)[:80])
