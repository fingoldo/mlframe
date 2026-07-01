"""iter68 A/B: identity + warm-timing for compute_probabilistic_multiclass_error."""
import time
import numpy as np
from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error as f


def mk_b(n, s=0):
    rng = np.random.default_rng(s)
    yt = (rng.random(n) < 0.3).astype(np.int64)
    p = np.clip(rng.random(n) * 0.6 + yt * 0.3, 0, 1)
    return yt, p


def mk_m(n, k, s=0):
    rng = np.random.default_rng(s)
    yt = rng.integers(0, k, n).astype(np.int64)
    sc = rng.random((n, k)); sc /= sc.sum(1, keepdims=True)
    return yt, sc


def mk_shift(n, k, s=0):
    # non-0-indexed labels (e.g. 10,20,30) -> must still trigger remap
    yt, sc = mk_m(n, k, s)
    labelvals = np.array([10, 20, 30, 40, 50][:k])
    yt2 = labelvals[yt]
    return yt2, sc


# identity vs reference values (recomputed from current code; reference computed once below)
cases = [("bin", *mk_b(20000)), ("mc3", *mk_m(20000, 3)), ("mc5", *mk_m(20000, 5)), ("shift3", *mk_shift(20000, 3))]
for name, yt, sc in cases:
    v = f(yt, sc)
    print(f"{name}: {v!r}")

# timing
yt_b, p_b = mk_b(20000); yt_m, s_m = mk_m(20000, 3)
f(yt_b, p_b); f(yt_m, s_m)
N = 4000
best = 1e9
for _ in range(5):
    t = time.perf_counter()
    for _ in range(N):
        f(yt_b, p_b); f(yt_m, s_m)
    dt = time.perf_counter() - t
    best = min(best, dt)
print(f"best total {N} pairs: {best*1000:.1f} ms  ({best/N*1e6:.2f} us/pair)")
