"""A/B bench: Welford single-pass fold vs the prior two-pass fused regression block (_regression_metrics.py).

Run: python src/mlframe/metrics/_benchmarks/bench_fused_regression_welford.py (PYTHONPATH=src, CUDA off).
Result @N=10M (16-thread Ryzen, py3.14): NEW single-pass 1.59-1.79x faster, SS_tot rel-diff ~1e-13..1e-15 (FP reduction-order, non-decision-altering). RESOLVED: shipped as default for fast_regression_metrics_block."""
import sys; sys.modules['cupy']=None
import scipy.stats, numba, numpy as np, time
from numba import njit, prange

NP = dict(cache=True, fastmath=True, nogil=True)

# OLD: pass1 (sum_abs,sum_sqr,max_abs,sum_y) + pass2 centred SS
@njit(**NP, parallel=True)
def p1_par(yt, yp, nt):
    n=yt.shape[0]; cs=(n+nt-1)//nt
    la=np.zeros(nt); ls=np.zeros(nt); lm=np.zeros(nt); ly=np.zeros(nt)
    for tid in prange(nt):
        st=tid*cs; en=min(st+cs,n); sa=0.0; ss=0.0; m=0.0; sy=0.0
        for i in range(st,en):
            e=yt[i]-yp[i]; ae=e if e>=0 else -e
            sa+=ae; ss+=e*e
            if ae>m: m=ae
            sy+=yt[i]
        la[tid]=sa; ls[tid]=ss; lm[tid]=m; ly[tid]=sy
    sa=0.0; ss=0.0; m=0.0; sy=0.0
    for tid in range(nt):
        sa+=la[tid]; ss+=ls[tid]
        if lm[tid]>m: m=lm[tid]
        sy+=ly[tid]
    return sa,ss,m,sy

@njit(**NP, parallel=True)
def p2_par(yt, ym):
    n=yt.shape[0]; ss=0.0
    for i in prange(n):
        d=yt[i]-ym; ss+=d*d
    return ss

def old(yt,yp,nt):
    n=yt.shape[0]
    sa,ss,m,sy=p1_par(yt,yp,nt); ym=sy/n
    sst=p2_par(yt,ym)
    return sa,ss,m,sst

# NEW: fused single pass with Welford per-thread + Chan combine
@njit(**NP, parallel=True)
def fused_par(yt, yp, nt):
    n=yt.shape[0]; cs=(n+nt-1)//nt
    la=np.zeros(nt); ls=np.zeros(nt); lm=np.zeros(nt)
    lcnt=np.zeros(nt); lmean=np.zeros(nt); lM2=np.zeros(nt)
    for tid in prange(nt):
        st=tid*cs; en=min(st+cs,n); sa=0.0; ss=0.0; mx=0.0
        cnt=0; mean=0.0; M2=0.0
        for i in range(st,en):
            e=yt[i]-yp[i]; ae=e if e>=0 else -e
            sa+=ae; ss+=e*e
            if ae>mx: mx=ae
            v=yt[i]; cnt+=1; d=v-mean; mean+=d/cnt; M2+=d*(v-mean)
        la[tid]=sa; ls[tid]=ss; lm[tid]=mx
        lcnt[tid]=cnt; lmean[tid]=mean; lM2[tid]=M2
    sa=0.0; ss=0.0; mx=0.0
    # Chan parallel combine
    Tcnt=0.0; Tmean=0.0; TM2=0.0
    for tid in range(nt):
        sa+=la[tid]; ss+=ls[tid]
        if lm[tid]>mx: mx=lm[tid]
        cb=lcnt[tid]
        if cb>0.0:
            if Tcnt==0.0:
                Tcnt=cb; Tmean=lmean[tid]; TM2=lM2[tid]
            else:
                delta=lmean[tid]-Tmean; tot=Tcnt+cb
                Tmean=Tmean+delta*cb/tot
                TM2=TM2+lM2[tid]+delta*delta*Tcnt*cb/tot
                Tcnt=tot
    return sa,ss,mx,TM2

np.random.seed(0)
nt=numba.get_num_threads()
for mean_ in (0.0, 11500.0):
    yt=(np.random.randn(10_000_000)+mean_).astype(np.float64)
    yp=(yt+np.random.randn(10_000_000)*0.5).astype(np.float64)
    # warm
    old(yt[:1000],yp[:1000],nt); fused_par(yt[:1000],yp[:1000],nt)
    o=old(yt,yp,nt); f=fused_par(yt,yp,nt)
    print(f"mean={mean_}: SS_tot old={o[3]:.6f} new={f[3]:.6f} reldiff={abs(o[3]-f[3])/abs(o[3]):.2e}")
    print(f"  sa diff={abs(o[0]-f[0]):.2e} ss diff={abs(o[1]-f[1]):.2e} max diff={abs(o[2]-f[2]):.2e}")
    def bench(fn,*a,r=20):
        ts=[]
        for _ in range(r):
            t=time.perf_counter(); fn(*a); ts.append(time.perf_counter()-t)
        return min(ts),np.median(ts)
    om,omed=bench(old,yt,yp,nt)
    fm,fmed=bench(fused_par,yt,yp,nt)
    print(f"  OLD min={om*1000:.2f}ms med={omed*1000:.2f}ms | NEW min={fm*1000:.2f}ms med={fmed*1000:.2f}ms | speedup={omed/fmed:.2f}x")
