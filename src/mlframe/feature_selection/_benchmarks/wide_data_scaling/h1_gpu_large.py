"""H1 large-p GPU-only: measured CUDA points at p=5000,10000 + GPU memory feasibility."""
import sys, time
import numpy as np
if __name__ == "__main__":
    sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/src")
    sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository")
    PROG = r"D:/Temp/synergy_scale_bench/progress.txt"
    def ck(m):
        open(PROG,"a").write(time.strftime("%H:%M:%S")+" | "+m+"\n"); print(m, flush=True)

    from mlframe.feature_selection.filters.batch_pair_mi_gpu import batch_pair_mi_cuda, _CUDA_AVAIL
    import cupy as cp

    N, NBINS = 8000, 8
    def make(p, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, p))
        codes = np.empty((N, p), dtype=np.int32)
        for j in range(p):
            q = np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1])
            codes[:, j] = np.searchsorted(q, X[:, j])
        y = ((codes[:, 0] >= NBINS // 2) ^ (codes[:, 1] >= NBINS // 2)).astype(np.int32)
        nbins = np.full(p, NBINS, dtype=np.int32)
        fy = np.bincount(y, minlength=2).astype(np.float64) / N
        return codes, nbins, y, fy

    # warmup
    c,nb,y,fy = make(20); a,b=np.triu_indices(20,1)
    batch_pair_mi_cuda(c,a.astype(np.int64),b.astype(np.int64),nb,y,fy)
    ck("GPU-large warmup done")

    for p in (5000, 10000):
        npairs = p*(p-1)//2
        codes,nbins,y,fy = make(p)
        a,b = np.triu_indices(p,1)
        a=a.astype(np.int64); b=b.astype(np.int64)
        idx_mb = (a.nbytes+b.nbytes)/1e6
        out_mb = npairs*8/1e6
        codes_mb = codes.nbytes/1e6
        free0,total = cp.cuda.runtime.memGetInfo()
        try:
            t0=time.perf_counter()
            mi = batch_pair_mi_cuda(codes,a,b,nbins,y,fy)
            dt=time.perf_counter()-t0
            free1,_ = cp.cuda.runtime.memGetInfo()
            ck("p=%d pairs=%d CUDA=%.2fs pps=%.3e | codes=%.0fMB idx=%.0fMB out=%.0fMB GPUfree_before=%.0fMB total=%.0fMB"%(
                p,npairs,dt,npairs/dt,codes_mb,idx_mb,out_mb,free0/1e6,total/1e6))
            ck("  planted-pair MI(0,1)=%.4f rank=%d/%d"%(mi[0], int((mi>mi[0]).sum()), npairs))
        except Exception as e:
            ck("p=%d CUDA FAIL: %r | host codes=%.0fMB idx=%.0fMB out=%.0fMB" % (p, e, codes_mb, idx_mb, out_mb))
        del codes, a, b
    ck("H1 GPU-large done")
