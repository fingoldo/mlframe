"""cProfile harness for ``interaction_proxy_top_n`` -- confirm the top-k gate bounds the pairwise cost.

The interaction term is O(k^2) per subset and O(k^2 * n) memory where k=interaction_top_k, NOT O(P^2):
a wide proxy (P=200) with k=30 must cost the same as a narrow one. We profile two widths at fixed k and
confirm ``build_pair_table`` (the only O(k^2 * n) allocation) and the per-subset scorer stay bounded.

Run:
  CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
      python -m mlframe.feature_selection._benchmarks.profile_shap_interaction_proxy
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interaction_proxy import (
    build_pair_table,
    interaction_proxy_top_n,
)


def _synthetic(n, P, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n, P))
    Phi = rng.normal(size=(n, P, P)) * 0.3
    Phi = 0.5 * (Phi + np.transpose(Phi, (0, 2, 1)))
    base = rng.normal(size=n) * 0.1
    y = (rng.uniform(size=n) < 0.5).astype(float)
    return phi, Phi, base, y


def _time(n, P, k, reps=3):
    phi, Phi, base, y = _synthetic(n, P)
    # warm numba
    interaction_proxy_top_n(phi, Phi, base, y, classification=True, metric="brier",
                            min_card=1, max_card=4, top_n=20, interaction_top_k=k)
    t0 = time.perf_counter()
    for _ in range(reps):
        interaction_proxy_top_n(phi, Phi, base, y, classification=True, metric="brier",
                                min_card=1, max_card=4, top_n=20, interaction_top_k=k)
    return (time.perf_counter() - t0) / reps


def run():
    n, k = 1500, 30
    print("Gate check: per-subset interaction cost is O(k^2), independent of P (k fixed=30):")
    for P in (40, 120, 200):
        dt = _time(n, P, k)
        # pair-table memory: O(k^2 * n) floats, not O(P^2 * n)
        mem_mb = (k * k * n * 8) / 1e6
        print(f"  P={P:4d}  k={k}  search_wall={dt*1000:7.1f} ms  pair_table_mem={mem_mb:.1f} MB (O(k^2*n))")
    print("\nIf wall is ~flat across P, the top-k gate works (the pairwise term ignores P-k tail features).")

    print("\n=== cProfile (n=1500, P=200, k=30) top cumulative ===")
    phi, Phi, base, y = _synthetic(1500, 200)
    interaction_proxy_top_n(phi, Phi, base, y, classification=True, metric="brier",
                            min_card=1, max_card=4, top_n=20, interaction_top_k=30)  # warm
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        interaction_proxy_top_n(phi, Phi, base, y, classification=True, metric="brier",
                                min_card=1, max_card=4, top_n=20, interaction_top_k=30)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    run()
