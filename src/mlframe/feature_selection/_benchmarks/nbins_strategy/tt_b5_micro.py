"""B5: cost of one candidate transform per element (numba, one thread, scalar libm), to see which transform dominates the feature criterion.
Each kernel does  s += f(x_i) * r_i  over 4e6 float32 elements (the same accumulate the fused criterion does); ns per element = min over 9 repeats of the wall time (single thread).
"""
import math
import time

import numpy as np
from numba import njit

N = 4_000_000
rng = np.random.default_rng(0)
x = rng.standard_normal(N).astype(np.float32)
r = rng.standard_normal(N).astype(np.float32)
MU, EPS = 0.0, 1e-3


def mk(f):
    jf = njit(inline="always", fastmath=True)(f)

    @njit(fastmath=True)
    def run(x, r):
        s = 0.0
        for i in range(x.shape[0]):
            s += jf(float(x[i])) * r[i]
        return s
    return run


FUN = {
    "identity": lambda v: v,
    "abs_c + sq_c (arith)": lambda v: abs(v - MU) + (v - MU) * (v - MU),
    "signed_sqrt": lambda v: math.sqrt(abs(v)) if v >= 0 else -math.sqrt(abs(v)),
    "signed_square": lambda v: v * abs(v),
    "log_abs": lambda v: math.log(abs(v) + EPS),
    "cbrt": lambda v: np.cbrt(v),
    "tanh_z": lambda v: math.tanh(v),
    "exp_z": lambda v: math.exp(min(max(v, -8.0), 8.0)),
    "inv": lambda v: (1.0 if v >= 0 else -1.0) / (abs(v) + EPS),
}


def best_cpu(fn, *a):
    best = 1e9
    for _ in range(9):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best / N * 1e9


if __name__ == "__main__":
    res = {}
    for name, f in FUN.items():
        run = mk(f)
        run(x[:1000], r[:1000])
        res[name] = best_cpu(run, x, r)
        print(f"{name:24s} {res[name]:7.2f} ns/elem", flush=True)
    tot = sum(res[k] for k in ("identity", "log_abs", "signed_sqrt", "signed_square", "cbrt", "abs_c + sq_c (arith)", "tanh_z", "exp_z", "inv"))
    print(f"sum over the 9 candidates (identity, log_abs, signed_sqrt, signed_square, cbrt, abs_c + sq_c, tanh_z, exp_z, inv): {tot:.1f} ns per element")
