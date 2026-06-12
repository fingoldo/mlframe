"""cProfile harness + warm micro-bench for the cluster_aggregate feature-selection step.

WHAT IT PROFILES
    ``run_cluster_aggregate_step`` (filters/_cluster_aggregate.py) on a representative frame:
    ~5 correlated reflection clusters of ~4 cols each + independent noise, n~2000, running the
    full cluster discovery (pairwise corr + MI edges + PC1 homogeneity gate) AND all 9 aggregation
    methods (mean_z / mean_inv_var / median / pca_pc1 / pca_pc2 / factor_score / median_z /
    signed_max_abs / signed_l2_sum). Also a SMALL shape (n=600, 3 clusters) so attribution-overhead
    skew is visible.

HOW TO RUN
    CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python \
        src/mlframe/feature_selection/_benchmarks/profile_cluster_aggregate.py

FINDINGS (2026-06-11, store-py3.14, CPU-only, warmed numba):
    Profile (representative n=2000, 5 clusters x 4 + noise) -- mlframe-side hotspots by tottime,
    cProfile-noise filtered (sklearn / numpy deep-stack attribution removed):

      #1  np.linalg.svd            -- called via _derive_weights for EVERY SVD-needing method
          (mean_inv_var, pca_pc1, pca_pc2, factor_score) on the SAME Z, per cluster. The Layer-50
          ``svd_cache`` dict that collapses those 4 SVDs into 1 EXISTS but the orchestration loop in
          ``run_cluster_aggregate_step`` calls ``_derive_weights(Z, method)`` with svd_cache=None,
          so each cluster pays 4x the SVD (plus 2x in _pc1_communalities for mean_inv_var+factor_score
          re-deriving the same comm). The cache is the obvious, already-built, bit-identical fix.
      #2  mi (info_theory)         -- one joint-MI eval per (cluster, method): 9 per cluster. Off-limits
          numeric kernel; genuine work (the supervised gate), not wasted. No prune available.
      #3  apply_recipe / nanpercentile -- per-method quantile-edge build + replay. Genuine per-method
          fit work (each method has a different continuous aggregate -> different edges). Not redundant.

    THE WINS SHIPPED (both bit-identical, default-on):
      A) Discarded-work prune (dominant). The fit loop already holds ``_agg_continuous`` (the exact
         pre-bin aggregate) AND the fit-time quantile ``edges``; ``apply_recipe(recipe, X)`` was
         re-extracting every member column from ``X``, re-nan_to_num-ing, re-standardizing with the
         stored mean/std/signs, re-combining into the SAME ``_agg_continuous``, then binning it.
         Replaced with a direct ``searchsorted(edges[1:-1], _agg_continuous, side="right")`` -- the
         IDENTICAL reduction ``_apply_cluster_aggregate`` uses -- skipping the full member round trip
         (9 methods x k cols of pandas getitem + nan_to_num + column_stack per cluster). The
         no-quantization / no-edges branch still defers to ``apply_recipe`` for exact behaviour.
      B) SVD-reuse. Thread ONE svd_cache dict per cluster through the method loop so the SVD of Z is
         computed once and reused by mean_inv_var/pca_pc1/pca_pc2/factor_score, and communalities once
         (reused by mean_inv_var+factor_score). The Layer-50 cache existed but the orchestration loop
         passed svd_cache=None, re-SVDing the same Z up to 4x/cluster.

    MEASURED (warm, 20 reps, median): rep n=2000 5x4  47.66ms -> 26.24ms (1.82x); small n=600 3x4
    16.75ms -> 10.69ms (1.57x). cProfile total 0.880s -> 0.586s; svd calls 375 -> 150, nan_to_num
    5205 -> 1755, frame.__getitem__ 7380 -> 1980. Both wins bit-identical BY CONSTRUCTION; pinned by
    tests/feature_selection/test_cluster_aggregate.py::
    test_fit_binned_column_is_bit_identical_to_apply_recipe_replay (PASS on the fast path; FAILS under
    a simulated 1% aggregate drift -> a genuine selection-shift sensor, not a tautology). Full
    cluster_aggregate selector test set green (67 + adversarial/knobs 86).
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cluster_aggregate import (
    CLUSTER_AGGREGATE_METHODS,
    run_cluster_aggregate_step,
)


def _make_frame(n=2000, n_clusters=5, k=4, noise=0.7, n_noise=6, seed=0):
    """n_clusters reflection clusters of k members each (A_i = lambda_i*z + eps) + n_noise independent
    noise columns. Target driven by the cluster latents so the aggregates can out-score single members."""
    rng = np.random.default_rng(seed)
    cols = {}
    latents = []
    for c in range(n_clusters):
        z = rng.normal(size=n)
        latents.append(z)
        lam = rng.uniform(0.8, 1.2, size=k)
        for j in range(k):
            cols[f"c{c}_m{j}"] = lam[j] * z + noise * rng.normal(size=n)
    for j in range(n_noise):
        cols[f"noise{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    score = sum(latents[c] for c in range(n_clusters))
    y = (score > np.median(score)).astype(np.int64)
    return X, y


def _quantize(X, nbins=8):
    """Equi-frequency bin every column to int32 codes (mirrors the MRMR binned ``data`` matrix)."""
    arr = X.to_numpy(dtype=np.float64)
    out = np.empty(arr.shape, dtype=np.int32)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        edges = np.nanpercentile(col, np.linspace(0, 100, nbins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        out[:, j] = np.clip(np.digitize(col, edges[1:-1]), 0, nbins - 1).astype(np.int32)
    return out, np.full(arr.shape[1], nbins, dtype=np.int64)


def _build_call(n=2000, n_clusters=5, k=4, seed=0, nbins=8):
    """Build the full kwargs for one run_cluster_aggregate_step call (target appended as last col)."""
    X, y = _make_frame(n=n, n_clusters=n_clusters, k=k, seed=seed)
    feat_names = list(X.columns)
    Xf = X.copy()
    Xf["__y__"] = np.asarray(y)
    data, nb = _quantize(Xf, nbins=nbins)
    cols = list(Xf.columns)
    target_indices = [len(cols) - 1]
    return dict(
        data=data, cols=cols, nbins=nb, X=Xf, target_indices=target_indices,
        feature_names_in_=feat_names, categorical_idx=(),
        cached_MIs=None, engineered_recipes={},
        quantization_nbins=nbins, quantization_method="quantile",
        quantization_dtype=np.dtype(np.int32),
        methods=CLUSTER_AGGREGATE_METHODS,
        mi_prevalence=1.0, min_member_relevance=0.0, min_cluster_size=3,
        max_cluster_size=12, corr_threshold=0.5, homogeneity_tau=0.5,
        max_candidates=200, mode="augment", verbose=0, dtype=np.int32,
    )


def _run_once(kw):
    out = run_cluster_aggregate_step(**{k: (v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) and k == "engineered_recipes" else v))) for k, v in kw.items()})
    return out[4], out[7]  # n_added, summary


def _warm_bench(label, kw, reps=20):
    # warm
    for _ in range(3):
        _run_once(kw)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _run_once(kw)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    n_added, summary = _run_once(kw)
    print(f"  [{label}] n_added={n_added}  median={np.median(times)*1e3:.2f}ms  min={times.min()*1e3:.2f}ms")
    return np.median(times)


def _bitident_check(seeds=(0, 1, 7)):
    """Bit-identity gate: the orchestration's direct-bin fast path must reproduce ``apply_recipe``'s
    replay column EXACTLY for every (cluster, method). Rebuilds each accepted recipe and compares the
    stored summary's selection to a from-scratch apply_recipe replay over the same X."""
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    ok = True
    for seed in seeds:
        kw = _build_call(n=2000, n_clusters=5, k=4, seed=seed)
        er: dict = {}
        kw2 = dict(kw); kw2["engineered_recipes"] = er
        out = run_cluster_aggregate_step(**kw2)
        n_added = out[4]
        # For every accepted recipe, the binned column appended to `data` (out[0] tail cols) must
        # equal apply_recipe(recipe, X) bit-for-bit -- proving the fast path == the replay path.
        data_out = out[0]
        Xf = kw["X"]
        added = data_out[:, data_out.shape[1] - n_added:]
        for j, (name, recipe) in enumerate(list(er.items())[-n_added:] if n_added else []):
            replay = apply_recipe(recipe, Xf).astype(added.dtype)
            col = added[:, j]
            if not np.array_equal(col, replay):
                ok = False
                print(f"  MISMATCH seed={seed} recipe={name}: {(col != replay).sum()} of {col.size} cells differ")
        print(f"  seed={seed}: n_added={n_added}, all accepted cols bit-identical to apply_recipe replay: "
              f"{'YES' if ok else 'NO'}")
    print(f"BIT-IDENTITY: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    print("=== bit-identity: direct-bin fast path == apply_recipe replay ===")
    _bitident_check()

    print("\n=== warm micro-bench (current code) ===")
    kw_big = _build_call(n=2000, n_clusters=5, k=4, seed=0)
    kw_small = _build_call(n=600, n_clusters=3, k=4, seed=1)
    _warm_bench("rep n=2000 5x4", kw_big)
    _warm_bench("small n=600 3x4", kw_small)

    print("\n=== cProfile (representative n=2000, 5x4) -- top 30 by tottime ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(15):
        _run_once(kw_big)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(30)
    print(s.getvalue())

    print("=== cProfile (representative) -- top 25 by cumulative ===")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).strip_dirs().sort_stats("cumulative").print_stats(25)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
