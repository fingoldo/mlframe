"""Frontier probe (2026-06-03): all SU-clustering kernel paths must produce the
SAME partition. cluster_correlated_features_su routes to CPU-scalar, CPU-bitmap,
GPU-scalar, or GPU-bitmap depending on flags/width/bin-count. After the GPU
memory-gate fix, re-verify cross-path parity across bin counts (incl. above and
below the bitmap_max_n_bins cutoff).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import entropy  # noqa: F401  (warm import)
from mlframe.feature_selection._shap_proxy_cluster_su import (
    cluster_correlated_features_su,
    cluster_su_gpu_available,
)


def _bins(n=1500, seed=0, card=10):
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    cols, nb = {}, {}
    for i in range(40):
        if i < 5:
            x = z0 + 0.15 * rng.standard_normal(n)
        elif i < 10:
            x = z1 + 0.15 * rng.standard_normal(n)
        else:
            x = rng.standard_normal(n)
        e = np.unique(np.quantile(x, np.linspace(0, 1, card + 1)))
        b = np.searchsorted(e[1:-1], x, side="right").astype(np.int32)
        cols[f"f{i}"] = b
        nb[f"f{i}"] = int(b.max()) + 1
    return cols, nb, [f"f{i}" for i in range(40)]


def _labels(bins, nb, names, **kw):
    return cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, nbins_per_feature=nb, **kw)


@pytest.mark.parametrize("card", [4, 10, 14])  # 14 > bitmap_max_n_bins(12) -> scalar fallback
def test_cpu_bitmap_matches_cpu_scalar(card):
    from sklearn.metrics import adjusted_rand_score
    bins, nb, names = _bins(card=card)
    scalar = _labels(bins, nb, names, use_gpu=False, use_bitmap=False)
    bitmap = _labels(bins, nb, names, use_gpu=False, use_bitmap=True)
    assert adjusted_rand_score(scalar, bitmap) == 1.0, (
        f"CPU bitmap vs scalar partition differs at card={card}"
    )


@pytest.mark.skipif(not cluster_su_gpu_available(), reason="no GPU")
@pytest.mark.parametrize("card", [4, 10, 14])
def test_gpu_matches_cpu(card):
    from sklearn.metrics import adjusted_rand_score
    bins, nb, names = _bins(card=card)
    cpu = _labels(bins, nb, names, use_gpu=False, use_bitmap=False)
    gpu = _labels(bins, nb, names, use_gpu=True, use_bitmap=False)
    assert adjusted_rand_score(cpu, gpu) == 1.0, f"GPU vs CPU differs at card={card}"
