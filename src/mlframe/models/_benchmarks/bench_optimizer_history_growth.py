"""Bench: concatenate-per-submit history growth vs capacity-doubling buffer.

Models the submit_evaluations storage pattern in models/optimization.py: each submit appends a
small batch to a running (known_candidates, known_evaluations) history. The current code does
np.concatenate([history, batch]) per submit -> O(n) copy each call -> O(n^2) over the run.

Run: python -m mlframe.models._benchmarks.bench_optimizer_history_growth
"""
from __future__ import annotations

import time

import numpy as np


def run_concat(n_submits, batch):
    kc = np.array([], dtype=np.int64)
    ke = np.array([], dtype=np.float64)
    for _ in range(n_submits):
        nb = np.arange(batch)
        ne = np.random.random(batch)
        kc = np.concatenate([kc, np.asarray(nb)]).astype(int)
        ke = np.concatenate([ke, np.asarray(ne)])
        # simulate per-iter read of the full history (model fit reshape)
        _ = kc.reshape(-1, 1)
    return kc.size


def run_buffer(n_submits, batch):
    cap = 16
    kc = np.empty(cap, dtype=np.int64)
    ke = np.empty(cap, dtype=np.float64)
    ln = 0
    for _ in range(n_submits):
        nb = np.arange(batch)
        ne = np.random.random(batch)
        need = ln + batch
        if need > cap:
            while cap < need:
                cap *= 2
            kc2 = np.empty(cap, dtype=np.int64); kc2[:ln] = kc[:ln]; kc = kc2
            ke2 = np.empty(cap, dtype=np.float64); ke2[:ln] = ke[:ln]; ke = ke2
        kc[ln:need] = nb
        ke[ln:need] = ne
        ln = need
        _ = kc[:ln].reshape(-1, 1)
    return ln


def bestof(fn, *a, n=5):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    for n_submits, batch in ((200, 1), (1000, 1), (2000, 1), (1000, 8)):
        c = bestof(run_concat, n_submits, batch)
        b = bestof(run_buffer, n_submits, batch)
        print(f"submits={n_submits:5d} batch={batch:2d}  concat {c*1e3:8.3f}ms  buffer {b*1e3:8.3f}ms  ({c/b:.2f}x)")


if __name__ == "__main__":
    main()
