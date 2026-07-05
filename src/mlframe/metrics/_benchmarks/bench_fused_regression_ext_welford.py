"""A/B bench: fully-fused single-pass (Welford + online co-moment) vs the 2-pass extended 12-metric block (_regression_extras.py).

Run: python src/mlframe/metrics/_benchmarks/bench_fused_regression_ext_welford.py (PYTHONPATH=src, CUDA off).
Result @N=10M: 1.06x@mean=0 / 0.99x@mean=11500 (identity ~1e-13). REJECTED: pass1 is ALU-bound on MAPE/SMAPE divisions, so eliminating pass2 memory read nets nothing while per-element Welford divisions add cost. Kept for re-test on other HW."""
import sys; sys.modules['cupy']=None
import scipy.stats, numba, numpy as np, time
from numba import njit, prange
NP=dict(cache=True, fastmath=True, nogil=True)

@njit(**NP, parallel=True)
def ext1_par(yt,yp,nt):
    n=yt.shape[0]; eps=np.finfo(np.float64).eps; chunk=(n+nt-1)//nt
    la=np.zeros(nt);ls=np.zeros(nt);lm=np.zeros(nt);ly=np.zeros(nt);lp=np.zeros(nt)
    lsig=np.zeros(nt);lape=np.zeros(nt);lsm=np.zeros(nt);lay=np.zeros(nt);lz=np.zeros(nt,np.int64)
    for tid in prange(nt):
        sa = 0.0
        ss = 0.0
        ma = 0.0
        sy = 0.0
        sp = 0.0
        sg = 0.0
        sape = 0.0
        ssm = 0.0
        say = 0.0
        nz = 0
        st = tid * chunk
        en = min(st + chunk, n)
        for i in range(st, en):
            a = yt[i]
            b = yp[i]
            e = a - b
            ae = e if e >= 0 else -e
            sa += ae
            ss += e * e
            if ae > ma:
                ma = ae
            sy += a
            sp += b
            sg += b - a
            ay = a if a >= 0 else -a
            say += ay
            dm = ay if ay >= eps else eps
            sape += ae / dm
            if a == 0.0:
                nz += 1
            ap = b if b >= 0 else -b
            ds = ay + ap
            if ds < eps:
                ds = eps
            ssm += 2.0 * ae / ds
        la[tid] = sa
        ls[tid] = ss
        lm[tid] = ma
        ly[tid] = sy
        lp[tid] = sp
        lsig[tid] = sg
        lape[tid] = sape
        lsm[tid] = ssm
        lay[tid] = say
        lz[tid] = nz
    sa = 0.0
    ss = 0.0
    ma = 0.0
    sy = 0.0
    sp = 0.0
    sg = 0.0
    sape = 0.0
    ssm = 0.0
    say = 0.0
    nz = 0
    for tid in range(nt):
        sa+=la[tid];ss+=ls[tid]
        if lm[tid]>ma:ma=lm[tid]
        sy+=ly[tid];sp+=lp[tid];sg+=lsig[tid];sape+=lape[tid];ssm+=lsm[tid];say+=lay[tid];nz+=lz[tid]
    return sa,ss,ma,sy,sp,sg,sape,ssm,say,nz

@njit(**NP, parallel=True)
def ext2_par(yt, yp, ym, pm):
    n = yt.shape[0]
    rm = ym - pm
    sst = 0.0
    ssp = 0.0
    sxy = 0.0
    ssr = 0.0
    for i in prange(n):
        dy=yt[i]-ym;dp=yp[i]-pm;sst+=dy*dy;ssp+=dp*dp;sxy+=dy*dp
        dr=(yt[i]-yp[i])-rm;ssr+=dr*dr
    return sst,ssp,sxy,ssr

def old(yt,yp,nt):
    r=ext1_par(yt,yp,nt);n=yt.shape[0];ym=r[3]/n;pm=r[4]/n
    return r, ext2_par(yt,yp,ym,pm)

# NEW: fully fused single pass with co-moment online updates for sst,ssp,sxy,ssr
@njit(**NP, parallel=True)
def fused_par(yt,yp,nt):
    n=yt.shape[0]; eps=np.finfo(np.float64).eps; chunk=(n+nt-1)//nt
    la=np.zeros(nt);ls=np.zeros(nt);lm=np.zeros(nt);lsig=np.zeros(nt)
    lape=np.zeros(nt);lsm=np.zeros(nt);lay=np.zeros(nt);lz=np.zeros(nt,np.int64)
    lcnt=np.zeros(nt);lmy=np.zeros(nt);lmp=np.zeros(nt);lmr=np.zeros(nt)
    lM2y=np.zeros(nt);lM2p=np.zeros(nt);lM2r=np.zeros(nt);lCxy=np.zeros(nt)
    for tid in prange(nt):
        sa = 0.0
        ss = 0.0
        ma = 0.0
        sg = 0.0
        sape = 0.0
        ssm = 0.0
        say = 0.0
        nz = 0
        cnt = 0.0
        my = 0.0
        mp = 0.0
        mr = 0.0
        M2y = 0.0
        M2p = 0.0
        M2r = 0.0
        Cxy = 0.0
        st = tid * chunk
        en = min(st + chunk, n)
        for i in range(st, en):
            a = yt[i]
            b = yp[i]
            e = a - b
            ae = e if e >= 0 else -e
            sa += ae
            ss += e * e
            if ae > ma:
                ma = ae
            sg += b - a
            ay = a if a >= 0 else -a
            say += ay
            dm = ay if ay >= eps else eps
            sape += ae / dm
            if a == 0.0:
                nz += 1
            ap = b if b >= 0 else -b
            ds = ay + ap
            if ds < eps:
                ds = eps
            ssm += 2.0 * ae / ds
            cnt += 1.0
            dy = a - my
            my += dy / cnt
            dy2 = a - my
            M2y += dy * dy2
            dp = b - mp
            mp += dp / cnt
            dp2 = b - mp
            M2p += dp * dp2
            Cxy += dy * dp2
            r = e
            dr = r - mr
            mr += dr / cnt
            M2r += dr * (r - mr)
        la[tid] = sa
        ls[tid] = ss
        lm[tid] = ma
        lsig[tid] = sg
        lape[tid] = sape
        lsm[tid] = ssm
        lay[tid] = say
        lz[tid] = nz
        lcnt[tid] = cnt
        lmy[tid] = my
        lmp[tid] = mp
        lmr[tid] = mr
        lM2y[tid] = M2y
        lM2p[tid] = M2p
        lM2r[tid] = M2r
        lCxy[tid] = Cxy
    sa = 0.0
    ss = 0.0
    ma = 0.0
    sg = 0.0
    sape = 0.0
    ssm = 0.0
    say = 0.0
    nz = 0
    Tc = 0.0
    Tmy = 0.0
    Tmp = 0.0
    Tmr = 0.0
    TM2y = 0.0
    TM2p = 0.0
    TM2r = 0.0
    TCxy = 0.0
    for tid in range(nt):
        sa += la[tid]
        ss += ls[tid]
        if lm[tid] > ma:
            ma = lm[tid]
        sg += lsig[tid]
        sape += lape[tid]
        ssm += lsm[tid]
        say += lay[tid]
        nz += lz[tid]
        cb = lcnt[tid]
        if cb > 0.0:
            if Tc == 0.0:
                Tc = cb
                Tmy = lmy[tid]
                Tmp = lmp[tid]
                Tmr = lmr[tid]
                TM2y = lM2y[tid]
                TM2p = lM2p[tid]
                TM2r = lM2r[tid]
                TCxy = lCxy[tid]
            else:
                tot=Tc+cb;f=Tc*cb/tot
                ddy=lmy[tid]-Tmy;ddp=lmp[tid]-Tmp;ddr=lmr[tid]-Tmr
                TM2y+=lM2y[tid]+ddy*ddy*f
                TM2p+=lM2p[tid]+ddp*ddp*f
                TM2r+=lM2r[tid]+ddr*ddr*f
                TCxy+=lCxy[tid]+ddy*ddp*f
                Tmy+=ddy*cb/tot;Tmp+=ddp*cb/tot;Tmr+=ddr*cb/tot;Tc=tot
    sy=Tmy*nz*0+0.0  # placeholder; we need sum_y for y_mean too
    return sa,ss,ma,sg,sape,ssm,say,nz,Tmy,Tmp,TM2y,TM2p,TM2r,TCxy,Tc

np.random.seed(0); nt=numba.get_num_threads()
for mean_ in (0.0,11500.0):
    yt=(np.random.randn(10_000_000)+mean_); yp=yt+np.random.randn(10_000_000)*0.7
    old(yt[:1000],yp[:1000],nt); fused_par(yt[:1000],yp[:1000],nt)
    (r1,r2)=old(yt,yp,nt); f=fused_par(yt,yp,nt)
    sst_o,ssp_o,sxy_o,ssr_o=r2
    sst_n,ssp_n,ssr_n,sxy_n=f[10],f[11],f[12],f[13]
    def rel(a,b):return abs(a-b)/(abs(a)+1e-30)
    print(f"mean={mean_}: sst rel={rel(sst_o,sst_n):.2e} ssp rel={rel(ssp_o,ssp_n):.2e} sxy rel={rel(sxy_o,sxy_n):.2e} ssr rel={rel(ssr_o,ssr_n):.2e}")
    def bench(fn,*a,r=20):
        ts=[]
        for _ in range(r):
            t=time.perf_counter();fn(*a);ts.append(time.perf_counter()-t)
        return min(ts),np.median(ts)
    om,omed=bench(old,yt,yp,nt);fm,fmed=bench(fused_par,yt,yp,nt)
    print(f"  OLD min={om*1e3:.2f} med={omed*1e3:.2f}ms | NEW min={fm*1e3:.2f} med={fmed*1e3:.2f}ms | speedup={omed/fmed:.2f}x")
