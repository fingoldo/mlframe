"""cProfile + warm-microbench harness for ShapProxiedFS.fit (profile-and-optimize discipline).

Profiles ``ShapProxiedFS.fit`` at two representative classification shapes (a primary n~800/p~25/
n_splits=3 and a smaller n~400/p~18), sorts by BOTH cumulative and tottime (top 30), and filters the
mlframe-side hotspots from the sklearn/shap/numpy deep-stack attribution noise.

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection._benchmarks.bench_shapproxied_fit_profile

VERDICT (2026-06-11): RESOLVED -- 1.39x on the brute-force search stage (the dominant mlframe-side hotspot).

Top mlframe-side hotspots at the primary shape (n=800, p=25, max_card=10 -> 7.1M subsets; fit wall ~26s):
  1. ``_shap_proxy_search.brute_force_top_n`` -- tottime 14.0s, cumtime 14.2s, 1 call (~53% of fit wall).
     A wall-clock decomposition of the njit kernel showed 97% of its time is ``score_margin`` (the Brier
     sigmoid/exp reduction over n rows x 7.1M combos), 3% margin assembly. cProfile attributes the njit
     time to the Python caller because numba frames are invisible to it -- a warm microbench confirmed the
     kernel is real compute (12.4s parallel / 48s serial @8-chunk), NOT attribution noise.
  2. ``_shap_proxy_revalidate._loss_from_predictions`` -- cumtime ~21.7s, but ~all of it is the joblib /
     HistGB predict pool (sklearn-internal, out of scope: the MRMR-fit-core / model path).
  3. ``_shap_proxy_revalidate._slice_cols_to_numpy`` -- cumtime ~27s, dominated by the same pandas/joblib
     deep stack (sklearn-internal); its own tottime is ~0.003s -- attribution noise, not a kernel.

OPTIMIZATION SHIPPED: the chunk-parallel brute-force kernels were dispatched with a HARDCODED
``n_chunks=8``, leaving every core past the 8th idle on a many-core host. On a 22-core box, raising the
fan-out to ``2 * NUMBA_NUM_THREADS`` (=44) ran the n=540/f=25/7.1M-subset search 1.39x faster (warm:
13.1s -> 9.4s) and the smaller n=400/f=18 shape 1.09x, with the FULL top-N output (losses AND combos)
BIT-IDENTICAL to the 8-chunk and serial paths (chunking only partitions the combo enumeration; each
subset's loss and the deterministic ``_merge_topn`` are unchanged). ``brute_force_top_n`` now defaults
``n_chunks=None`` -> ``_resolve_brute_force_n_chunks()`` (HW-aware, kernel_tuning_cache override hook,
memoized). Regression sensor: ``test_brute_force_n_chunks_bit_identical_to_default``.

REJECTED (kept here as a written negative result): a per-combo early-abort on ``score_margin`` (Brier/
log-loss/MAE/MSE terms are all non-negative, so a monotone partial sum could abort once it exceeds the
running worst-kept top-N threshold -- bit-identical for kept combos). Measured a WASH-to-loss (per-element
abort 1.04x; 64-row blocked abort 0.97x) because (a) the threshold barely prunes -- Brier losses across
subsets are tightly clustered near the 30th-best, so almost every combo runs the full sum, and (b) the
abort branch defeats SIMD vectorization of the exp loop, costing about what it saves. ``score_margin``
itself is already fastmath + at the scalar-exp throughput floor (~7.3 ns/row), so no further per-element
win there. Do not re-attempt the abort without a workload whose losses are widely separated.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def make_data(n=800, n_inf=8, n_noise=12, n_corr=5, seed=0):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, n_inf))
    noise = rng.normal(size=(n, n_noise))
    corr = inf[:, :n_corr] + 0.3 * rng.normal(size=(n, n_corr))
    X = pd.DataFrame(
        np.column_stack([inf, noise, corr]),
        columns=[f"inf{i}" for i in range(n_inf)] + [f"noise{i}" for i in range(n_noise)]
        + [f"corr{i}" for i in range(n_corr)],
    )
    coefs = np.linspace(1.0, 0.3, n_inf)
    logit = inf @ coefs
    y = (logit + 0.4 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _make_selector(seed=0):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="bruteforce", max_features=10,
        top_n=30, n_splits=3, n_revalidation_models=3, random_state=seed, verbose=False,
        model=_small_model(),
    )


def _small_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=0)


def profile_fit(X, y, label):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS  # noqa: F401

    sel = _make_selector()
    pr = cProfile.Profile()
    pr.enable()
    sel.fit(X, y)
    pr.disable()
    for sort_key in ("cumulative", "tottime"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(sort_key).print_stats(30)
        print(f"\n=== [{label}] cProfile top 30 by {sort_key} ===")
        print(s.getvalue())
    return sel


if __name__ == "__main__":
    for (n, p_inf, p_noise, p_corr, lbl) in [
        (800, 8, 12, 5, "primary n=800 p=25"),
        (400, 6, 8, 4, "small n=400 p=18"),
    ]:
        X, y = make_data(n=n, n_inf=p_inf, n_noise=p_noise, n_corr=p_corr)
        t0 = time.perf_counter()
        sel = profile_fit(X, y, lbl)
        print(f"[{lbl}] full fit wall: {time.perf_counter()-t0:.2f}s; selected={len(sel.selected_features_)}")
