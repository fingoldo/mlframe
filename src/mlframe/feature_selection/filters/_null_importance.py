"""Null (target-permutation) importance: a per-feature significance test against a shuffled-target baseline.

Raw model importance (gain/split/Gini) is not a significance test — a pure-noise feature can still pick up
nonzero importance by chance (a high-cardinality noise column especially: a tree finds SOME split that
reduces impurity purely by luck). ``null_importance_filter`` builds each feature's own null distribution of
importance under a target with NO real relationship to the features (shuffled ``y``, refit ``n_shuffles``
times) and keeps a feature only when its REAL importance clears a high percentile of its OWN null draws —
a feature is significant relative to what pure chance alone would produce for it, not relative to an
arbitrary global threshold. Distinct from feature-permutation MI elsewhere in this package (which shuffles a
FEATURE column to estimate that feature's own marginal contribution); this shuffles the TARGET to build a
chance baseline for every feature's importance score.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def null_importance_filter(
    X: Any,
    y: np.ndarray,
    importance_fn: Callable[[Any, np.ndarray], np.ndarray],
    n_shuffles: int = 50,
    percentile: float = 95.0,
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
    """Keep only features whose real importance clears a high percentile of their own null distribution.

    Parameters
    ----------
    X
        Feature matrix/frame, passed through to ``importance_fn`` unchanged.
    y
        ``(n_samples,)`` target array.
    importance_fn
        ``importance_fn(X, y) -> (n_features,)`` array of per-feature importances (e.g. a closure that fits
        a model and returns ``model.feature_importances_``). Called once with the real ``y`` and once per
        shuffle with a permuted ``y`` — ``n_shuffles + 1`` total fits, so keep this cheap (few trees /
        shallow depth) relative to the caller's main model.
    n_shuffles
        Number of target-permutation refits used to build each feature's null distribution.
    percentile
        A feature is kept when its real importance exceeds this percentile of its own null-importance draws
        (default 95 -> roughly a 5% false-keep rate under the null, by construction of the percentile test).
    random_state
        Seed for the shuffle sequence (reproducible null distribution).

    Returns
    -------
    dict[str, np.ndarray]
        ``real_importance`` ``(n_features,)``, ``null_importances`` ``(n_shuffles, n_features)``,
        ``threshold`` ``(n_features,)`` (the per-feature percentile cutoff), ``keep_mask`` ``(n_features,)``
        boolean.
    """
    rng = np.random.default_rng(random_state)
    real_importance = np.asarray(importance_fn(X, y), dtype=np.float64)
    n_features = real_importance.shape[0]

    y_arr = np.asarray(y)
    null_importances = np.empty((n_shuffles, n_features), dtype=np.float64)
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y_arr)
        null_importances[i] = np.asarray(importance_fn(X, y_shuffled), dtype=np.float64)

    threshold = np.percentile(null_importances, percentile, axis=0)
    keep_mask = real_importance > threshold

    return {
        "real_importance": real_importance,
        "null_importances": null_importances,
        "threshold": threshold,
        "keep_mask": keep_mask,
    }


__all__ = ["null_importance_filter"]
