"""Same-process A/B / cost probe for the pre-gathered draw ranks of the tied replicate MI kernel (see audits/2026-10-22/profiles/_bootstrap_rank_mi_tied_gpu.md). Run with PYTHONPATH on the worktree src."""
import time, numpy as np, cupy as cp, os
from mlframe.feature_selection.filters import _bootstrap_rank_mi_tied_gpu as T, _bootstrap_rank_mi_njit as H
from mlframe.feature_selection.filters._kernel_types import render
from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache
rng=np.random.default_rng(0); n=100000; k=128; sc=float(os.environ.get("SC","40"))
arr=np.round(rng.normal(size=(n,k))*sc).astype(np.float32)
y=(rng.random(n)<0.3).astype(np.int64)
reps=[rng.integers(0,n,int(0.8*n)).astype(np.int64) for _ in range(10)]
prep=H._resample_tables(reps,y,n,10,True)
rk,nlev,okb=H.dense_ranks_f32(np.ascontiguousarray(arr).view(np.uint32))
nl=np.asarray(nlev).astype(np.int32)
if not ((nl<=T.MAX_LEVELS).all()):
    raise AssertionError('(nl<=T.MAX_LEVELS).all()')
rk16=cp.asarray(np.ascontiguousarray(rk.T if rk.shape[0]==n else rk).astype(np.int16))
print("rk",rk16.shape)
idx=cp.asarray(prep["idx"].astype(np.int32)); binpos=cp.asarray(prep["binpos"].astype(np.int16)); ycode=cp.asarray(prep["y_codes"].astype(np.int32)); lens=cp.asarray(prep["lens"].astype(np.int32))
B,m_max=idx.shape; nb=10; ncls=int(prep["n_classes"]); nl_d=cp.asarray(nl)
src=T._SRC
new=src.replace("const int* __restrict__ ycode,","const int* __restrict__ ycode, const short* __restrict__ lv, const int* __restrict__ yg,")
new=new.replace("    const int* ib = idx + (long long)b * m_max;","    const int* ib = idx + (long long)b * m_max;\n    const short* lj = lv + ((long long)j * B + b) * m_max;\n    const int* ygb = yg + (long long)b * m_max;")
new=new.replace("atomicAdd(&mycnt[rj[ib[t]]], 1)","atomicAdd(&mycnt[lj[t]], 1)").replace("atomicAdd(&start[rj[ib[t]]], 1)","atomicAdd(&start[lj[t]], 1)")
new=new.replace("r = ib[t];\n            const int L = rj[r];","r = t;\n            const int L = lj[t];").replace("atomicAdd(&myhist[bp[start[L]] * ncls + ycode[r]], 1)","atomicAdd(&myhist[bp[start[L]] * ncls + ygb[t]], 1)")
new=new.replace("if (t < t1) { r = ib[t]; q = sid[rj[r]]; }","if (t < t1) { r = t; q = sid[lj[t]]; }").replace("myhist[bp[base + __popc(same & lt)] * ncls + ycode[r]]","myhist[bp[base + __popc(same & lt)] * ncls + ygb[r]]")
if not ("ycode[r]" not in new and "rj[ib" not in new):
    raise AssertionError('"ycode[r]" not in new and "rj[ib" not in new')
mods={"base":RawModuleCache(render(src)),"pregather":RawModuleCache(render(new))}
width=T._block_width(mods["base"].get(cp).get_function("tied_replicate_mi"),T._shared_bytes(int(nl.max()),8,nb,ncls),m_max)
shared=T._shared_bytes(int(nl.max()),width//32,nb,ncls)
yg=ycode[idx]   # (B, m_max) column-independent
def run(name):
    kern=mods[name].get(cp).get_function("tied_replicate_mi")
    res=cp.zeros((B,k),dtype=cp.float64)
    cp.cuda.Stream.null.synchronize(); t=time.perf_counter()
    if name=="base":
        kern((k*B,),(width,),(rk16,idx,binpos,ycode,lens,nl_d,np.int64(n),np.int32(m_max),np.int32(B),np.int32(nb),np.int32(ncls),res),shared_mem=shared)
    else:
        lv=cp.empty((k,B,m_max),dtype=cp.int16)
        for j in range(k): cp.take(rk16[j],idx,out=lv[j])
        kern((k*B,),(width,),(rk16,idx,binpos,ycode,lv,yg,lens,nl_d,np.int64(n),np.int32(m_max),np.int32(B),np.int32(nb),np.int32(ncls),res),shared_mem=shared)
    cp.cuda.Stream.null.synchronize(); return time.perf_counter()-t,res
out={}
for name in ("base","pregather"):
    run(name); ts=[]
    for _ in range(4): dt,r=run(name); ts.append(round(dt,3))
    out[name]=r; print(name,ts)
print("equal",bool(cp.array_equal(out["base"],out["pregather"])))
