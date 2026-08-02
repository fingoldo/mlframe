import sys; sys.modules['cupy'] = None  # type: ignore[assignment]
import time, numpy as np
from typing import Callable

from mlframe.calibration.quality import _anderson_darling_kernel as ad_serial, _anderson_darling_kernel_parallel as ad_par

n = 10_000_000
if __name__ == "__main__":
    rng=np.random.default_rng(0)
    pit=np.sort(rng.random(n))
    ad_serial(pit,n); ad_par(pit,n)
    def best(f: "Callable[[np.ndarray, int], float]", r: int = 7) -> "tuple[float, float]":
        ts = []
        for _ in range(r):
            t=time.perf_counter(); v=f(pit,n); ts.append(time.perf_counter()-t)
        return min(ts), v
    ms,vs=best(ad_serial); mp,vp=best(ad_par)
    # sort cost
    t=time.perf_counter(); np.sort(rng.random(n)); tsort=time.perf_counter()-t
    print(f"serial={ms*1000:.2f}ms par={mp*1000:.2f}ms speedup={ms/mp:.2f}x sort={tsort*1000:.2f}ms")
    print(f"identity diff={abs(vs-vp):.3e} rel={abs(vs-vp)/abs(vs):.3e}")
