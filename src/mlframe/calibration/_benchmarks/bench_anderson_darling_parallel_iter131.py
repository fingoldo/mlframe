import sys; sys.modules['cupy']=None
import time, numpy as np, numba
from numba import njit, prange

@njit(cache=True, nogil=True, fastmath=True)
def ad_serial(sorted_pit, n):
    eps=1e-12; acc=0.0
    for k in range(n):
        a=sorted_pit[k]
        if a<eps: a=eps
        elif a>1.0-eps: a=1.0-eps
        b=sorted_pit[n-1-k]
        if b<eps: b=eps
        elif b>1.0-eps: b=1.0-eps
        acc += (2*(k+1)-1)*(np.log(a)+np.log(1.0-b))
    return -n - (1.0/n)*acc

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def ad_par(sorted_pit, n):
    eps=1e-12; acc=0.0
    for k in prange(n):
        a=sorted_pit[k]
        if a<eps: a=eps
        elif a>1.0-eps: a=1.0-eps
        b=sorted_pit[n-1-k]
        if b<eps: b=eps
        elif b>1.0-eps: b=1.0-eps
        acc += (2*(k+1)-1)*(np.log(a)+np.log(1.0-b))
    return -n - (1.0/n)*acc

n=10_000_000
if __name__ == "__main__":
    rng=np.random.default_rng(0)
    pit=np.sort(rng.random(n))
    ad_serial(pit,n); ad_par(pit,n)
    import math
    def best(f,r=7):
        ts=[]
        for _ in range(r):
            t=time.perf_counter(); v=f(pit,n); ts.append(time.perf_counter()-t)
        return min(ts), v
    ms,vs=best(ad_serial); mp,vp=best(ad_par)
    # sort cost
    t=time.perf_counter(); np.sort(rng.random(n)); tsort=time.perf_counter()-t
    print(f"serial={ms*1000:.2f}ms par={mp*1000:.2f}ms speedup={ms/mp:.2f}x sort={tsort*1000:.2f}ms")
    print(f"identity diff={abs(vs-vp):.3e} rel={abs(vs-vp)/abs(vs):.3e}")
