"""Same-process A/B / cost probe for the pre-gathered draw ranks of the tied replicate MI kernel (see audits/2026-10-22/profiles/_bootstrap_rank_mi_tied_gpu.md). Run with PYTHONPATH on the worktree src."""
import time, numpy as np, cupy as cp
n=100000; m=80000; B=10
rng=np.random.default_rng(0)
for cols in (32,64,128):
    rk=cp.asarray(rng.integers(0,300,(cols,n)).astype(np.int16)); idx=cp.asarray(np.stack([rng.integers(0,n,m) for _ in range(B)]).astype(np.int32))
    yc=cp.asarray((rng.random(n)<0.3).astype(np.int32))
    out=cp.empty((cols,B,m),dtype=cp.int16)
    def f():
        for j in range(cols): cp.take(rk[j],idx,out=out[j])
    f(); cp.cuda.Stream.null.synchronize()
    t=time.perf_counter(); f(); cp.cuda.Stream.null.synchronize(); print(cols,"cols take loop",round(time.perf_counter()-t,3),"s", out.nbytes/1e6,"MB")
    t=time.perf_counter(); g=rk[:,idx]; cp.cuda.Stream.null.synchronize(); print(cols,"cols fancy",round(time.perf_counter()-t,3)); del g
