"""Held-out CV loss of a scaled linear model, the one scorer shared by the pure-form side check and the compaction's linear-value check.

Log-loss of a scaled logistic model for a classification target, MAE of OLS otherwise: the scorers and the fold split the usability-aware retention commits with
(:func:`mlframe.feature_selection.filters._usability_aware_selection._usability_scorers`).
"""

from __future__ import annotations

import numpy as np


def cv_loss(design: np.ndarray, y: np.ndarray, classification: bool, n_folds: int, seed: int) -> float:
    """Mean CV loss of a scaled linear model on the columns of ``design``; ``inf`` when it cannot be fit (non-finite design, one class)."""
    from sklearn.model_selection import KFold

    from ._usability_aware_selection import _scrub, _usability_scorers

    scorers = _usability_scorers(classification, y)
    if scorers is None:
        return float("inf")
    mk, logloss, y_enc, _n_classes = scorers
    x = _scrub(np.asarray(design, dtype=np.float64))  # Intentional float64: the retention gates score float64 columns
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.isfinite(x).all() or float(np.std(x)) <= 1e-12:
        return float("inf")
    losses = []
    for tr, va in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x):
        model = mk()
        if classification:
            model.fit(x[tr], y_enc[tr])
            losses.append(logloss(y_enc[va], model.predict_proba(x[va])))
        else:
            model.fit(x[tr], y[tr])
            losses.append(float(np.mean(np.abs(y[va] - model.predict(x[va])))))
    return float(np.mean(losses))
