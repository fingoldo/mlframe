"""Same-process A/B of tied_replicate_mi kernel variants (base, sequential-read bound, count unroll, prefetch, PRIV_LEVELS 0): 256 columns x 100000 rows with ties.

Run with PYTHONPATH on the worktree's src; SC=<scale> sets the number of distinct levels. Results: audits/2026-10-22/profiles/_bootstrap_rank_mi_tied_gpu.md."""
import time, numpy as np, cupy as cp
from mlframe.feature_selection.filters import _bootstrap_rank_mi_tied_gpu as T, _bootstrap_rank_mi_njit as H
from mlframe.feature_selection.filters._kernel_types import render
from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache
rng=np.random.default_rng(0); n=100000; k=256
arr=np.round(rng.normal(size=(n,k))*float(__import__('os').environ.get('SC','40'))).astype(np.float32)   # ~300 levels, heavy ties
y=(rng.random(n)<0.3).astype(np.int64)
reps=[rng.integers(0,n,int(0.8*n)).astype(np.int64) for _ in range(10)]
prep=H._resample_tables(reps,y,n,10,True)
def ranks(bits):
    return H.dense_ranks_f32(bits)
bits=np.ascontiguousarray(arr).view(np.uint32)
def run(label):
    cp.cuda.Stream.null.synchronize(); t=time.perf_counter()
    out=T.tied_replicate_mi_stream(bits,prep,10,ranks); cp.cuda.Stream.null.synchronize()
    return time.perf_counter()-t, out
base_src=T._SRC
variants={"base":base_src,
 "seq_bound(WRONG)":base_src.replace("rj[ib[t]]","rj[t]").replace("r = ib[t];\n            const int L = rj[r];","r = t;\n            const int L = rj[r];").replace("ycode[r]","ycode[t]")}
def patched(src):
    # count pass: 4 independent gathers in flight
    a1="        for (int t = tid; t < m; t += blockDim.x) atomicAdd(&mycnt[rj[ib[t]]], 1);"
    b1="""        { const int bd = blockDim.x; int t = tid;
          for (; t + 3 * bd < m; t += 4 * bd) { const short l0 = rj[ib[t]], l1 = rj[ib[t + bd]], l2 = rj[ib[t + 2 * bd]], l3 = rj[ib[t + 3 * bd]];
              atomicAdd(&mycnt[l0], 1); atomicAdd(&mycnt[l1], 1); atomicAdd(&mycnt[l2], 1); atomicAdd(&mycnt[l3], 1); }
          for (; t < m; t += bd) atomicAdd(&mycnt[rj[ib[t]]], 1); }"""
    a2="        for (int t = tid; t < m; t += blockDim.x) atomicAdd(&start[rj[ib[t]]], 1);"
    b2="""        { const int bd = blockDim.x; int t = tid;
          for (; t + 3 * bd < m; t += 4 * bd) { const short l0 = rj[ib[t]], l1 = rj[ib[t + bd]], l2 = rj[ib[t + 2 * bd]], l3 = rj[ib[t + 3 * bd]];
              atomicAdd(&start[l0], 1); atomicAdd(&start[l1], 1); atomicAdd(&start[l2], 1); atomicAdd(&start[l3], 1); }
          for (; t < m; t += bd) atomicAdd(&start[rj[ib[t]]], 1); }"""
    assert a1 in src and a2 in src
    return src.replace(a1,b1).replace(a2,b2)
def prefetch(src):
    w1o="""    for (int t = t0 + lane; t - lane < t1; t += 32) {
        int q = 255, r = 0;
        if (t < t1) {
            r = ib[t];
            const int L = rj[r];
            q = sid[L];
            if (q == 255) atomicAdd(&myhist[bp[start[L]] * ncls + ycode[r]], 1);
        }"""
    w1n="""    int rn = 0, Ln = 0, yn = 0;
    { const int tt = t0 + lane; if (tt < t1) { rn = ib[tt]; Ln = rj[rn]; yn = ycode[rn]; } }
    for (int t = t0 + lane; t - lane < t1; t += 32) {
        int q = 255, r = 0;
        const int rc = rn, Lc = Ln, yc = yn;
        { const int tn = t + 32; if (tn < t1) { rn = ib[tn]; Ln = rj[rn]; yn = ycode[rn]; } }
        if (t < t1) {
            r = rc;
            const int L = Lc;
            q = sid[L];
            if (q == 255) atomicAdd(&myhist[bp[start[L]] * ncls + yc], 1);
        }"""
    w2o="""        for (int t = t0 + lane; t - lane < t1; t += 32) {
            int q = 255, r = 0;
            if (t < t1) { r = ib[t]; q = sid[rj[r]]; }"""
    w2n="""        int rn2 = 0, yn2 = 0, Ln2 = 0;
        { const int tt = t0 + lane; if (tt < t1) { rn2 = ib[tt]; Ln2 = rj[rn2]; yn2 = ycode[rn2]; } }
        for (int t = t0 + lane; t - lane < t1; t += 32) {
            int q = 255, r = 0, yc2 = 0;
            const int rc2 = rn2, Lc2 = Ln2; yc2 = yn2;
            { const int tn = t + 32; if (tn < t1) { rn2 = ib[tn]; Ln2 = rj[rn2]; yn2 = ycode[rn2]; } }
            if (t < t1) { r = rc2; q = sid[Lc2]; }"""
    assert w1o in src and w2o in src
    src=src.replace(w1o,w1n).replace(w2o,w2n)
    o3="myhist[bp[base + __popc(same & lt)] * ncls + ycode[r]]"
    assert o3 in src
    return src.replace(o3,"myhist[bp[base + __popc(same & lt)] * ncls + yc2]")
variants["priv0"]=base_src.replace("#define PRIV_LEVELS 64","#define PRIV_LEVELS 0")
res={}
for name,src in variants.items():
    T._MODULE=RawModuleCache(render(src))
    run(name)
    ts=[run(name)[0] for _ in range(3)]
    res[name]=run(name)[1][0]
    print(name, [round(x,3) for x in ts])
print("base vs priv0 equal:", np.array_equal(res["base"],res["priv0"]))
