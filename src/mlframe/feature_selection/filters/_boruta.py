"""Boruta-style shadow-feature all-relevant selection.

Distinct from ``null_importance_filter`` (which shuffles the TARGET to build a chance baseline for each
feature's own importance) and from MRMR (a MINIMAL-redundant selector that deliberately drops correlated
features once one representative is kept). Boruta instead answers "is this feature relevant AT ALL" (an
ALL-relevant selector): it appends a per-column-shuffled "shadow" copy of every real feature, fits an
importance function over several iterations, and confirms a real feature only when it repeatedly beats the
best (max) shadow importance more often than chance -- via a two-sided binomial test against p=0.5, per the
original Boruta algorithm (Kursa & Rudnicki 2010).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


def boruta_select(
    X: Any,
    y: np.ndarray,
    importance_fn: Callable[[Any, np.ndarray], np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    n_iterations: int = 20,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Dict[str, Any]:
    """All-relevant feature selection via repeated shadow-feature importance comparison.

    Parameters
    ----------
    X
        Feature matrix/frame with ``n_features`` columns.
    y
        ``(n_samples,)`` target array.
    importance_fn
        ``importance_fn(X_with_shadows, y) -> (2 * n_features,)`` array of per-column importances for a
        matrix that is ``X``'s columns followed by their shuffled shadow copies, in that order (real column
        ``j`` at index ``j``, its shadow at index ``n_features + j``).
    feature_names
        Names for the real columns; inferred from ``X.columns`` if available, else ``f0, f1, ...``.
    n_iterations
        Number of shadow-shuffle-and-refit rounds. Each round, a real feature "wins" if its importance beats
        the max shadow importance that round.
    alpha
        Two-sided binomial-test significance level for confirming (win rate significantly > 0.5) or rejecting
        (win rate significantly < 0.5) a feature; features that never reach significance in either direction
        after ``n_iterations`` are ``"tentative"``.
    random_state
        Seed for the per-iteration shadow shuffles.

    Returns
    -------
    dict
        ``hit_counts`` ``(n_features,)`` int (rounds where the real feature beat max-shadow),
        ``win_rate`` ``(n_features,)`` float, ``decision`` (list of ``"confirmed"``/``"rejected"``/``"tentative"``
        per feature), ``feature_names`` (list).
    """
    from scipy.stats import binomtest

    if hasattr(X, "columns"):
        cols = list(X.columns)
        n_features = len(cols)
        names = list(feature_names) if feature_names is not None else cols
        is_frame = True
    else:
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
        is_frame = False

    rng = np.random.default_rng(random_state)
    hit_counts = np.zeros(n_features, dtype=np.int64)

    for _ in range(n_iterations):
        if is_frame:
            import pandas as pd

            shadow = X[cols].apply(lambda col: rng.permutation(col.to_numpy()), axis=0)
            shadow.columns = [f"{c}__shadow" for c in cols]
            X_shadowed = pd.concat([X[cols].reset_index(drop=True), shadow.reset_index(drop=True)], axis=1)
        else:
            X_arr = np.asarray(X)
            shadow_arr = np.empty_like(X_arr)
            for j in range(n_features):
                shadow_arr[:, j] = rng.permutation(X_arr[:, j])
            X_shadowed = np.concatenate([X_arr, shadow_arr], axis=1)

        importances = np.asarray(importance_fn(X_shadowed, y), dtype=np.float64)
        real_importances = importances[:n_features]
        max_shadow_importance = float(np.max(importances[n_features:]))
        hit_counts += (real_importances > max_shadow_importance).astype(np.int64)

    win_rate = hit_counts / n_iterations
    decisions = []
    for count in hit_counts:
        result = binomtest(int(count), n_iterations, p=0.5, alternative="two-sided")
        if result.pvalue < alpha and count / n_iterations > 0.5:
            decisions.append("confirmed")
        elif result.pvalue < alpha and count / n_iterations < 0.5:
            decisions.append("rejected")
        else:
            decisions.append("tentative")

    return {
        "hit_counts": hit_counts,
        "win_rate": win_rate,
        "decision": decisions,
        "feature_names": names,
    }


__all__ = ["boruta_select"]
