"""Opt-in multi-seed ensemble/union mode for ``stochastic_bandit_selection``.

A single bandit run's "top_feats" lock-in pool depends on the RNG seed: which weak-but-real features
cross ``lock_in_threshold`` before the epoch budget runs out is itself a stochastic outcome. Running
several independent seeds and unioning their locked-in pools recovers a more complete feature set than any
one seed, and the per-seed agreement (what fraction of seeds selected each feature) is a direct stability
diagnostic -- a feature selected by 9/10 seeds is a trustworthy signal, one selected by 1/10 is noise that
a single-seed run got lucky (or unlucky) with.

This module is purely additive: it reuses the single-seed core loop from ``stochastic_bandit_selection``
unmodified, so the original ``stochastic_bandit_selection`` public function and its return value are
bit-identical to before this module existed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.feature_selection.stochastic_bandit_selection import _stochastic_bandit_selection_core


@dataclass
class EnsembleSelectionResult:
    """Result of a multi-seed ensemble/union bandit search."""

    union_top_feats: List[str]
    """Union, across all seeds, of each run's locked-in "top_feats" pool plus its best-scoring subset."""

    stability: Dict[str, float]
    """Per-feature fraction of seeds (in [0, 1]) whose selected set (locked-in ∪ best subset) contained it."""

    per_seed_best_subsets: List[List[str]] = field(default_factory=list)
    """Each seed's best-CV-score subset, in seed order -- for diagnostics/debugging."""

    per_seed_selected_feats: List[List[str]] = field(default_factory=list)
    """Each seed's full selected set (locked-in ∪ best subset), in seed order."""


def stochastic_bandit_selection_ensemble(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    subset_size: int,
    seeds: Sequence[int],
    n_epochs: int = 200,
    cv: Optional[object] = None,
    up_factor: float = 1.05,
    down_factor: float = 0.97,
    lock_in_threshold: float = 3.0,
    moving_average_window: int = 10,
) -> EnsembleSelectionResult:
    """Run ``len(seeds)`` independent single-seed bandit searches and union their selected feature pools.

    Parameters
    ----------
    estimator, X, y, scoring, subset_size, n_epochs, cv, up_factor, down_factor, lock_in_threshold,
    moving_average_window
        Forwarded to each single-seed run, identical semantics to ``stochastic_bandit_selection``.
    seeds
        Random seeds for the independent runs. At least 2 seeds are required for a meaningful stability
        diagnostic (a single seed trivially has stability 1.0 for every feature it selects).

    Returns
    -------
    EnsembleSelectionResult
        ``union_top_feats`` is the recall-maximizing union of per-seed selections; ``stability`` maps each
        selected feature to the fraction of seeds that selected it, a direct measure of how much a single
        run's outcome should be trusted.
    """
    if len(seeds) < 1:
        raise ValueError("seeds must contain at least one random seed")

    per_seed_best_subsets: List[List[str]] = []
    per_seed_selected_feats: List[List[str]] = []

    for seed in seeds:
        best_subset, locked_in_feats = _stochastic_bandit_selection_core(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scoring,
            subset_size=subset_size,
            n_epochs=n_epochs,
            cv=cv,
            up_factor=up_factor,
            down_factor=down_factor,
            lock_in_threshold=lock_in_threshold,
            moving_average_window=moving_average_window,
            random_state=seed,
        )
        selected = sorted(set(locked_in_feats) | set(best_subset))
        per_seed_best_subsets.append(best_subset)
        per_seed_selected_feats.append(selected)

    n_seeds = len(seeds)
    all_feats = sorted(set.union(*(set(s) for s in per_seed_selected_feats)))
    stability = {feat: sum(1 for s in per_seed_selected_feats if feat in s) / n_seeds for feat in all_feats}

    return EnsembleSelectionResult(
        union_top_feats=all_feats,
        stability=stability,
        per_seed_best_subsets=per_seed_best_subsets,
        per_seed_selected_feats=per_seed_selected_feats,
    )


__all__ = ["stochastic_bandit_selection_ensemble", "EnsembleSelectionResult"]
