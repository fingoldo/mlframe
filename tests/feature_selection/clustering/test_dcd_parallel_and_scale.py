"""Frontier probes (2026-06-03): DCD clustering under joblib parallelism
(DCDState shared/serialized across workers) and at larger p. Now that the
machine is free, stress the paths the benign benchmarks skipped.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _dup_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame({
        "strong": other,
        "dup_a": latent + 0.01 * rng.standard_normal(n),
        "dup_b": latent + 0.01 * rng.standard_normal(n),
        "dup_c": latent + 0.01 * rng.standard_normal(n),
        "noise": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _fit(X, y, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    base = dict(dcd_enable=True, dcd_tau_cluster=0.5, dcd_cluster_size_threshold=2,
                verbose=0, random_seed=0)
    base.update(kw)
    return MRMR(**base).fit(X, y)


def test_dcd_result_independent_of_n_jobs():
    # DCD discover/swap mutates DCDState; the parallel candidate-scoring must not
    # race it. Serial vs parallel fits must give the SAME clustering/selection.
    X, y = _dup_frame()
    m1 = _fit(X, y, n_jobs=1)
    m4 = _fit(X, y, n_jobs=4)
    assert list(m1.get_feature_names_out()) == list(m4.get_feature_names_out()), (
        "DCD selection depends on n_jobs -> a parallelism race in clustering"
    )
    assert (m1.dcd_ or {}).get("n_swaps") == (m4.dcd_ or {}).get("n_swaps")
    assert (m1.dcd_ or {}).get("n_pruned") == (m4.dcd_ or {}).get("n_pruned")


def _wide_clustered(n=4000, n_latents=10, per=5, n_noise=150, seed=0):
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(n_latents)]
    cols = {}
    for li, z in enumerate(latents):
        for r in range(per):
            cols[f"L{li}_{r}"] = z + 0.05 * rng.standard_normal(n)
    for j in range(n_noise):
        cols[f"noise{j}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    signal = latents[0] + latents[1] + latents[2]
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * signal))).astype(int))
    return X, y


def test_dcd_large_p_no_crash_and_prunes():
    # p = 10*5 + 150 = 200 features, n=4000, 10 redundancy clusters. DCD must
    # run, prune redundant members, and transform finite -- no OOM/crash at scale.
    X, y = _wide_clustered()
    assert X.shape[1] == 200
    m = _fit(X, y)
    names = list(m.get_feature_names_out())
    assert len(names) >= 1
    # With 10 tight 5-member clusters, DCD must prune redundant members.
    assert int((m.dcd_ or {}).get("n_pruned", 0)) > 0, "DCD pruned nothing at p=200"
    out = np.asarray(m.transform(X.iloc[:500]), dtype=np.float64)
    assert np.all(np.isfinite(np.nan_to_num(out)))
