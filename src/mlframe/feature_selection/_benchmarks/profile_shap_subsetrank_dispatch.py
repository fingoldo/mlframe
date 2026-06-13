"""cProfile: the CPU-default subset-rank dispatcher adds no measurable overhead vs the bare kernel.

    CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
        python -m mlframe.feature_selection._benchmarks.profile_shap_subsetrank_dispatch

The dispatcher's CPU path is ``brute_force_top_n`` plus a constant-time ``_total_subsets`` +
``_gpu_min_subsets`` route decision (no GPU import on the default path). Confirms "no actionable
speedup / no regression" for the wrapper.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_subsetrank import brute_force_top_n_dispatch


def main():
    rng = np.random.default_rng(0)
    n, f, mc = 2000, 20, 7
    phi = rng.standard_normal((n, f)); base = rng.standard_normal(n); y = (rng.random(n) < 0.5).astype(np.float64)
    # warm both
    brute_force_top_n(phi, base, y, classification=True, max_card=mc, top_n=30, parallel=True)
    brute_force_top_n_dispatch(phi, base, y, classification=True, max_card=mc, top_n=30)

    def wall(fn):
        t = time.perf_counter()
        for _ in range(3):
            fn()
        return (time.perf_counter() - t) / 3

    t_bare = wall(lambda: brute_force_top_n(phi, base, y, classification=True, max_card=mc, top_n=30, parallel=True))
    t_disp = wall(lambda: brute_force_top_n_dispatch(phi, base, y, classification=True, max_card=mc, top_n=30))
    print(f"bare kernel  : {t_bare*1000:.2f} ms")
    print(f"dispatch(cpu): {t_disp*1000:.2f} ms   overhead {(t_disp-t_bare)*1000:.3f} ms ({(t_disp/t_bare-1)*100:+.2f}%)")

    pr = cProfile.Profile(); pr.enable()
    for _ in range(3):
        brute_force_top_n_dispatch(phi, base, y, classification=True, max_card=mc, top_n=30)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    main()
