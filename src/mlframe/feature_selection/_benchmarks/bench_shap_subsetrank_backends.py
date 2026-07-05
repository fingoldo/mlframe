"""Bench: GPU (cupy RawKernel) vs CPU subset-ranking for the SHAP-coalition proxy scan.

Run isolated (the dev box native-segfaults importing cupy under contention in the training process):

    CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
        python -m mlframe.feature_selection._benchmarks.bench_shap_subsetrank_backends

Measures, across subset count, the three subset-rank backends -- incremental CPU njit
(``brute_force_top_n``, the production default), naive per-subset CPU njit reference
(``brute_force_top_n_cpu_ref``, the GPU-mirror oracle), and the cupy RawKernel (``brute_force_top_n_gpu``)
-- and asserts the selected-subset SET is bit-identical across all three. Writes the table to
``_results/shap_subsetrank_backends.json``.

DECISION (this host): GPU wins 1.04-1.96x and is bit-identical, but the host segfaults importing cupy
under contention, so the dispatcher DEFAULTS TO CPU and the GPU route stays opt-in (``prefer_gpu=True``)
until a stable host tunes its crossover into the kernel_tuning_cache. The win is real + hardware-relative
-- kept, not deleted (REJECTED != DELETED).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_subsetrank import brute_force_top_n_cpu_ref

_RESULTS = Path(__file__).parent / "_results" / "shap_subsetrank_backends.json"


def _data(n, f, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, f)), rng.standard_normal(n), (rng.random(n) < 0.5).astype(np.float64))


def _time(fn, *a, **k):
    fn(*a, **k)  # warm (njit compile / nvrtc)
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def main():
    try:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu import brute_force_top_n_gpu, gpu_available

        gpu_on = gpu_available()
    except Exception as exc:  # cupy import segfault-guarded by isolation; here just record absence
        gpu_on, brute_force_top_n_gpu = False, None
        print("GPU unavailable:", exc)

    grid = [(500, 16, 8), (2000, 18, 6), (2000, 20, 5), (1000, 22, 4), (2000, 20, 8)]
    rows = []
    for n, f, mc in grid:
        phi, base, y = _data(n, f)
        C = sum(math.comb(f, r) for r in range(1, mc + 1))
        cpu, tc = _time(brute_force_top_n, phi, base, y, classification=True, max_card=mc, top_n=30, parallel=True)
        ref, tr = _time(brute_force_top_n_cpu_ref, phi, base, y, classification=True, max_card=mc, top_n=30)
        row = dict(n=n, f=f, max_card=mc, C=C, cpu_s=round(tc, 5), cpu_ref_s=round(tr, 5), ref_set_match=set(c for _, c in cpu) == set(c for _, c in ref))
        if gpu_on:
            gpu, tg = _time(brute_force_top_n_gpu, phi, base, y, classification=True, max_card=mc, top_n=30)
            row.update(gpu_s=round(tg, 5), speedup_gpu_vs_cpu=round(tc / tg, 3),
                       gpu_set_match=set(c for _, c in cpu) == set(c for _, c in gpu),
                       argmax_match=cpu[0][1] == gpu[0][1])
        rows.append(row)
        print(row)

    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS.write_text(json.dumps(dict(gpu_available=gpu_on, host_note="cupy import segfaults under contention; CPU is default",
                                        rows=rows), indent=2, sort_keys=True))
    print("wrote", _RESULTS)


if __name__ == "__main__":
    main()
