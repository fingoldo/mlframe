"""Shared ShapProxiedFS configuration for the iter-numbered revalidation and trust-guard sweeps."""

from __future__ import annotations

from typing import Any


def build_shap_proxied_selector(seed: int, *, n_anchors: int, **overrides: Any):
    """Return a ShapProxiedFS with the sweep baseline settings; ``overrides`` replace individual settings."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    params: dict[str, Any] = dict(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=n_anchors,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        random_state=seed, verbose=False,
    )
    params.update(overrides)
    return ShapProxiedFS(**params)
