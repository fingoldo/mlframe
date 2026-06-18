"""biz_val: ``cluster_weighting`` denoises a correlated cluster better than a plain average.

When a cluster's members share a latent factor but carry HETEROGENEOUS noise loadings (some members
clean, some very noisy), the default ``pca_pc1`` weighting down-weights the noisy members along PC1,
so the collapsed cluster representative tracks the latent factor more tightly than ``mean_z`` (equal
1/k average, which lets the noisy members drag the representative). The representative's correlation
with the latent factor is the measurable win this selector exploits downstream.

Measured dev run (seed=0): corr(rep, z) pca_pc1=0.937 vs mean_z=0.899. Floor carries seed headroom.
"""

from __future__ import annotations

import numpy as np


def _hetero_noise_cluster(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)  # shared latent factor
    noise_scales = [0.2, 0.3, 1.5, 2.0, 2.5]  # heterogeneous member reliabilities
    M = np.column_stack([z + s * rng.normal(size=n) for s in noise_scales])
    labels = np.zeros(M.shape[1], dtype=np.int64)
    return M, labels, z


def _rep_corr(weighting, seed=0):
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster import build_unit_matrix

    M, labels, z = _hetero_noise_cluster(seed=seed)
    units, _, _ = build_unit_matrix(M, labels, weighting=weighting)
    return abs(np.corrcoef(units[:, 0], z)[0, 1])


def test_biz_val_cluster_weighting_pca_pc1_denoises_better_than_mean_z():
    pca = _rep_corr("pca_pc1")
    mean = _rep_corr("mean_z")
    assert pca >= 0.90, f"pca_pc1 rep corr {pca:.4f} below floor"
    assert pca - mean >= 0.02, f"pca_pc1 ({pca:.4f}) should beat mean_z ({mean:.4f}) by >=0.02 on hetero-noise cluster"


def test_biz_val_cluster_weighting_pca_pc1_wins_across_seeds():
    deltas = [_rep_corr("pca_pc1", s) - _rep_corr("mean_z", s) for s in range(4)]
    wins = sum(d > 0 for d in deltas)
    assert wins >= 3, f"pca_pc1 should beat mean_z on majority of seeds; deltas={deltas}"
