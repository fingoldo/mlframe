"""Probe: does a chunked-parallel ECE kernel beat the serial single-pass at large n?

The serial kernel's docstring rejected `parallel=True` (naive prange overhead). This re-tests a
per-thread manual-chunk histogram reduction at the sizes the bootstrap-ECE loop actually sees.

bench-attempt-rejected (2026-07-05): chunked-parallel ECE = 0.58x@200k, 0.79x@1M (22 threads) --
SLOWER than serial at every size AND not bit-identical (per-thread partials reduce in a different
order -> FP divergence that can move the reported ECE). The kernel streams 2x8MB float64 arrays into
10 bins: it is memory-bandwidth bound, so extra threads add overhead + false-sharing with no compute
to amortise. Serial single-pass stays the default. Do not re-ship parallel ECE without a bandwidth win.
"""
from __future__ import annotations
import time
import numpy as np
import numba
from numba import njit


@njit(cache=True, nogil=True)
def _ece_serial(y, p, n_bins):
    n = p.size
    sum_p = np.zeros(n_bins); sum_y = np.zeros(n_bins); nf = 0.0
    for i in range(n):
        pi = p[i]; yi = y[i]
        if not (np.isfinite(pi) and np.isfinite(yi)):
            continue
        b = int(pi * n_bins)
        if b >= n_bins: b = n_bins - 1
        elif b < 0: b = 0
        sum_p[b] += pi; sum_y[b] += yi; nf += 1.0
    if nf == 0.0: return float("nan")
    tot = 0.0
    for b in range(n_bins):
        d = sum_y[b] - sum_p[b]
        tot += -d if d < 0 else d
    return tot / nf


@njit(cache=True, nogil=True, parallel=True)
def _ece_par(y, p, n_bins):
    n = p.size
    nt = numba.get_num_threads()
    pp = np.zeros((nt, n_bins)); py = np.zeros((nt, n_bins)); cnt = np.zeros(nt)
    for t in numba.prange(nt):
        s = t * n // nt; e = (t + 1) * n // nt
        for i in range(s, e):
            pi = p[i]; yi = y[i]
            if not (np.isfinite(pi) and np.isfinite(yi)):
                continue
            b = int(pi * n_bins)
            if b >= n_bins: b = n_bins - 1
            elif b < 0: b = 0
            pp[t, b] += pi; py[t, b] += yi; cnt[t] += 1.0
    nf = 0.0
    for t in range(nt):
        nf += cnt[t]
    if nf == 0.0: return float("nan")
    tot = 0.0
    for b in range(n_bins):
        sp = 0.0; sy = 0.0
        for t in range(nt):
            sp += pp[t, b]; sy += py[t, b]
        d = sy - sp
        tot += -d if d < 0 else d
    return tot / nf


def med(fn, it):
    xs = []
    for _ in range(it):
        t0 = time.perf_counter(); fn(); xs.append((time.perf_counter()-t0)*1e3)
    return float(np.median(xs))


def main():
    print("threads", numba.get_num_threads())
    rng = np.random.default_rng(0)
    for n in (200_000, 1_000_000):
        y = rng.integers(0, 2, n).astype(np.float64); p = rng.random(n)
        _ece_serial(y, p, 10); _ece_par(y, p, 10)
        s = med(lambda: _ece_serial(y, p, 10), 15)
        pa = med(lambda: _ece_par(y, p, 10), 15)
        print(f"n={n:>9} serial={s:.3f}ms par={pa:.3f}ms {s/pa:.2f}x  equal={_ece_serial(y,p,10)==_ece_par(y,p,10)}")


if __name__ == "__main__":
    main()
