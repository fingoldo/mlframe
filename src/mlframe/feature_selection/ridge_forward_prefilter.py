"""``ridge_coefficient_prefilter``: a cheap Ridge-coefficient-based fast pre-filter ahead of MRMR/RFECV.

Source: 1st_home-credit-default-risk.md, Bojan -- "very simple forward feature selection using just Ridge
regression" reduced ~1600 features to ~240 with almost no CV loss, before combining with teammates' features.

MRMR/DCD's information-theoretic screening is expensive at thousands of candidate features (that's WHY they
exist -- they find genuinely nonlinear/redundancy-aware structure a linear model misses). This module is
explicitly NOT a replacement: it's a much cheaper linear surrogate meant to run FIRST, when the raw feature
count is high enough that running the expensive pipeline directly is impractical, pruning down to a
manageable candidate pool before handing off to MRMR for the real (non-linear-aware) selection pass.

Algorithm: one Ridge fit on ALL (standardized) features gives a fast ``|coefficient|`` importance ranking
(a single O(features) fit, not a per-feature greedy refit loop -- the whole point is to be cheap at
thousands of features). Then a handful of candidate pool sizes (log-spaced) are cross-validated, and the
SMALLEST pool whose CV score is within ``tol`` of the best observed score is returned -- Bojan's own
"almost no CV loss" criterion, made explicit and tunable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def ridge_coefficient_prefilter(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    candidate_sizes: Optional[Sequence[int]] = None,
    cv: int = 3,
    tol: float = 0.01,
    is_classifier: bool = False,
    alpha: float = 1.0,
    random_state: int = 42,
) -> List[str]:
    """Prune ``feature_names`` down to the smallest Ridge-ranked prefix that keeps CV score within ``tol``
    of the best observed candidate-size CV score.

    Parameters
    ----------
    X
        ``(n_samples, n_features)``, columns aligned with ``feature_names``.
    y
        Target -- continuous (regression, ``is_classifier=False``) or class labels (``is_classifier=True``).
    feature_names
        Names for each column of ``X``.
    candidate_sizes
        Pool sizes to cross-validate, ascending. Defaults to a log-spaced sweep up to ``n_features``
        (``[16, 32, 64, 128, 256, 512, 1024, ...]`` capped at ``n_features``, deduplicated).
    cv
        Number of cross-validation folds for the candidate-size sweep.
    tol
        Max ALLOWED relative drop from the best observed CV score (score comparison assumes HIGHER is
        better -- R^2 for regression, accuracy for classification) for a smaller pool to be preferred.
    is_classifier
        Use ``RidgeClassifier``/accuracy scoring instead of ``Ridge``/R^2.
    alpha
        Ridge regularization strength.

    Returns
    -------
    list of str
        Feature names in the selected prefilter pool, ordered by ``|coefficient|`` (most important first).
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError("ridge_coefficient_prefilter: feature_names length must match X.shape[1]")

    n_features = X.shape[1]
    if candidate_sizes is None:
        sizes = []
        size = 16
        while size < n_features:
            sizes.append(size)
            size *= 2
        sizes.append(n_features)
        candidate_sizes = sorted(set(s for s in sizes if s <= n_features))

    X_std = StandardScaler().fit_transform(X)

    ranker = RidgeClassifier(alpha=alpha, random_state=random_state) if is_classifier else Ridge(alpha=alpha, random_state=random_state)
    ranker.fit(X_std, y)
    coefs = np.asarray(ranker.coef_)
    if coefs.ndim > 1:
        coefs = np.abs(coefs).max(axis=0)  # multiclass: worst-case (max) importance across classes
    else:
        coefs = np.abs(coefs)
    ranked_idx = np.argsort(-coefs)

    scoring = "accuracy" if is_classifier else "r2"
    best_score = -np.inf
    size_scores = {}
    for size in candidate_sizes:
        cols = ranked_idx[:size]
        model = RidgeClassifier(alpha=alpha, random_state=random_state) if is_classifier else Ridge(alpha=alpha, random_state=random_state)
        score = float(np.mean(cross_val_score(model, X_std[:, cols], y, cv=cv, scoring=scoring)))
        size_scores[size] = score
        best_score = max(best_score, score)

    for size in candidate_sizes:
        if size_scores[size] >= best_score - tol:
            selected_idx = ranked_idx[:size]
            break
    else:
        selected_idx = ranked_idx  # pragma: no cover -- unreachable, largest candidate always satisfies the check

    return [feature_names[i] for i in selected_idx]


__all__ = ["ridge_coefficient_prefilter"]
