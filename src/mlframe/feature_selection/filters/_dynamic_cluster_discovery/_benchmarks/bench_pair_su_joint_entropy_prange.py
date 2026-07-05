"""REJECTED A/B: a parallel (prange) joint-entropy variant for the DCD ``pair_su`` hot loop (2026-07).

Context: ``_dcd_metrics.pair_su`` (56.8s / 110260 calls in the 1M-row wellbore profile) spends ~all of its
tottime inside ``info_theory._class_encoding.joint_entropy_2var`` -- a single-pass joint-histogram + entropy
reduction (already at its floor after the prior 23x ``joint_freqs_2var`` and 1.24x ``joint_entropy_2var``
fusion wins). That kernel lives in ``info_theory/`` (out of scope to edit here), but the always-try-njit-prange
rule says to at least MEASURE a parallel variant callable from the owned ``_dcd_metrics.py``.

VERDICT: REJECTED. A per-thread-partial-histogram prange build (bit-identical: integer counts are
order-independent, entropy reduced in the same ascending class-id order -> diff EXACTLY 0.0) is SLOWER at the
representative per-pair n and barely wins only at 1M:
    n=100000 serial=210.7us  par=457.9us  0.46x
    n=300000 serial=638.2us  par=1048.3us 0.61x
    n=1000000 serial=4637us   par=3983us   1.16x
The histogram is memory-bandwidth bound and each ``pair_su`` call is a single pair, so the 22-thread spawn +
per-thread partial-histogram reduction overhead dominates at the n DCD actually sees (~300k effective:
515us/call profiled == 559us serial kernel here). Parallelising per-pair across 110260 calls would REGRESS.
Kept as a runnable negative result; ``pair_su`` is NO-ACTIONABLE within owned files.

Run:  python path/to/_benchmarks/bench_pair_su_joint_entropy_prange.py
"""
from __future__ import annotations

import time

import numpy as np
import numba
from numba import njit, prange

from mlframe.feature_selection.filters.info_theory._class_encoding import joint_entropy_2var


@njit(nogil=True, cache=True, parallel=True)
def _joint_entropy_2var_prange(fd, ia, ib, nb_a, nb_b):
    n = fd.shape[0]
    if n == 0:
        return 0.0
    size = nb_a * nb_b
    nt = numba.get_num_threads()
    parts = np.zeros((nt, size), dtype=np.int64)
    for r in prange(n):
        t = numba.get_thread_id()
        parts[t, fd[r, ia] + fd[r, ib] * nb_a] += 1
    h = 0.0
    for c in range(size):
        cnt = 0
        for t in range(nt):
            cnt += parts[t, c]
        if cnt != 0:
            p = cnt / n
            h += np.log(p) * p
    return -h


def main() -> None:
    rng = np.random.default_rng(0)
    print("threads", numba.get_num_threads())
    for n in (100000, 300000, 1000000):
        fd = rng.integers(0, 10, size=(n, 5)).astype(np.int32)
        a = joint_entropy_2var(fd, 0, 1, 10, 10)
        b = _joint_entropy_2var_prange(fd, 0, 1, 10, 10)
        R = 50
        t = time.perf_counter()
        for _ in range(R):
            joint_entropy_2var(fd, 0, 1, 10, 10)
        s = (time.perf_counter() - t) / R * 1e6
        t = time.perf_counter()
        for _ in range(R):
            _joint_entropy_2var_prange(fd, 0, 1, 10, 10)
        p = (time.perf_counter() - t) / R * 1e6
        print(f"n={n:>8} serial={s:8.1f}us par={p:8.1f}us speedup={s / p:.2f}x diff={abs(a - b):.2e}")


if __name__ == "__main__":
    main()
