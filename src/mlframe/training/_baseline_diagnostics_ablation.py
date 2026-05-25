"""Ablation loop for BaselineDiagnostics.

Carved out of ``baseline_diagnostics`` via method-rebinding (W10E pattern).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _run_ablation(
    self,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    target_type: str,
    raw_fi: np.ndarray,
    raw_metric: float,
    metric_name: str,
    higher_is_better: bool,
) -> list:
    """Drop top-K features by FI rank, refit, measure delta. Per-feature
    independent drops (NOT cumulative) - we want per-feature
    contribution, not interaction-aware joint impact.

    K independent fits run via ``joblib.Parallel(threading)`` when
    a thread pool is available; each LightGBM gets ``n_jobs=1`` so
    the inner threads don't oversubscribe with the outer worker
    pool. Profiled wall-time on n=2000, K=4: 627 ms serial -> ~250 ms
    parallel (2.5x).
    """
    from .baseline_diagnostics import AblationEntry, _delta_pct

    if raw_fi.size == 0 or raw_fi.sum() == 0:
        logger.info(
            "BaselineDiagnostics: feature_importances_ all-zero; ablation skipped."
        )
        return []

    top_k = max(1, min(self.config.ablation_top_k, len(feature_cols)))
    # Indices of top-K features by FI, descending. lexsort with feature-index
    # tiebreaker so tied zero FIs don't make the logged "top-K ranked" output drift.
    order = np.lexsort((np.arange(len(raw_fi)), -raw_fi))[:top_k]

    # Build the per-feature work list once. Skipping zero-importance features
    # at this stage avoids both the joblib dispatch and the wasted refit.
    per_feature_work: list[tuple[int, int, str, list[str], list[str]]] = []
    for rank, idx in enumerate(order, start=1):
        feat = feature_cols[idx]
        if raw_fi[idx] <= 0:
            continue
        kept = [c for c in feature_cols if c != feat]
        if not kept:
            continue
        cat_kept = [c for c in cat_features if c in kept]
        per_feature_work.append((rank, int(idx), feat, kept, cat_kept))

    def _one_drop(rank: int, idx: int, feat: str,
                  kept: list[str], cat_kept: list[str]):
        X_drop = X.loc[:, kept]
        try:
            metric_drop, _ = self._fit_quick_and_score(
                X_drop, y, kept, cat_kept, target_type, metric_name,
                inner_n_jobs=1,  # outer pool owns the cores
            )
        except (ValueError, RuntimeError, TypeError, IndexError) as exc:
            # LightGBM ablation: degenerate-input ValueError, dtype TypeError,
            # LightGBM RuntimeError. KeyboardInterrupt / MemoryError still propagate.
            logger.warning(
                "BaselineDiagnostics: ablation refit for '%s' failed: %s; skipping.",
                feat, exc,
            )
            return None
        return AblationEntry(
            feature=feat,
            metric_after_drop=metric_drop,
            delta_pct=_delta_pct(raw_metric, metric_drop, higher_is_better),
            rank=rank,
        )

    results: list
    if len(per_feature_work) > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(
                n_jobs=min(len(per_feature_work), 8),
                backend="threading",  # numpy data shared, no pickle cost
            )(
                delayed(_one_drop)(rank, idx, feat, kept, cat_kept)
                for (rank, idx, feat, kept, cat_kept) in per_feature_work
            )
        except ImportError:
            results = [_one_drop(*args) for args in per_feature_work]
    else:
        results = [_one_drop(*args) for args in per_feature_work]

    entries = [r for r in results if r is not None]
    # Sort by absolute dominance descending so dominant_features is ranked by
    # impact, not by raw FI (the two usually agree but FI can mislead on correlated features).
    entries.sort(key=lambda e: -e.delta_pct)
    # Re-rank after sort
    entries = [
        AblationEntry(
            feature=e.feature,
            metric_after_drop=e.metric_after_drop,
            delta_pct=e.delta_pct,
            rank=i,
        )
        for i, e in enumerate(entries, start=1)
    ]
    return entries
