"""The downstream model panel and its scoring.

The pre-registration requires a panel, minimum `{logistic, LightGBM}`, with the selector-by-model
interaction reported: a selector that wins for a linear model and loses for a gradient-boosted one is a
common and important result that a single-model design structurally cannot see.

A wrapper arm must be run with an internal estimator differing from at least one panel member, or it is
being scored on its own objective; `assert_wrapper_estimator_differs` makes that a runtime check rather
than a convention.

Metrics come from `mlframe.metrics` (`fast_brier_score_loss`, `fast_log_loss_binary`, `fast_roc_auc`,
`average_precision_score`) rather than sklearn: `tests/test_meta/test_no_sklearn_metrics_in_production.py`
bans several sklearn metrics in production code and the fast kernels are the documented path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from mlframe.metrics import average_precision_score, fast_brier_score_loss, fast_log_loss_binary, fast_roc_auc

logger = logging.getLogger(__name__)

__all__ = [
    "PANEL_MEMBERS",
    "panel_factories",
    "assert_wrapper_estimator_differs",
    "score_predictions",
    "base_rate_scores",
    "normalized_skill",
    "fit_and_score_panel",
]

# Minimum panel mandated by the pre-registration. A run may add members; it may not drop one.
PANEL_MEMBERS: Sequence[str] = ("logistic", "lightgbm")


def panel_factories() -> Dict[str, Callable[[], Any]]:
    """Return `{member: factory}` for the downstream panel; each factory builds a fresh unfitted model."""

    def _logistic() -> Any:
        """Standardised logistic regression -- the linear panel member."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    def _lightgbm() -> Any:
        """LightGBM -- the gradient-boosted panel member, near-invariant to selection at unlimited K."""
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=4, verbose=-1)

    return {"logistic": _logistic, "lightgbm": _lightgbm}


def assert_wrapper_estimator_differs(arm_name: str, internal_estimator: Optional[str], panel: Sequence[str] = PANEL_MEMBERS) -> None:
    """Raise when a wrapper arm's internal estimator matches every panel member.

    `internal_estimator=None` means the arm is not a wrapper (no internal model), which is always fine.
    """
    if internal_estimator is None:
        return
    others = [m for m in panel if m != internal_estimator]
    if not others:
        raise ValueError(
            f"arm {arm_name!r} optimises {internal_estimator!r}, which is the only panel member: it would be "
            "scored on its own objective. Add a panel member the arm does not optimise."
        )


def score_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Return the metric bundle for one set of held-out predicted probabilities."""
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    yp = np.asarray(y_prob, dtype=np.float64)
    return {
        "roc_auc": float(fast_roc_auc(yt, yp)),
        "average_precision": float(average_precision_score(yt, yp)),
        "brier": float(fast_brier_score_loss(yt, yp)),
        "log_loss": float(fast_log_loss_binary(yt, yp)),
    }


def base_rate_scores(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Score the constant train-prevalence predictor on the holdout.

    This is the value a crashed cell is charged in the intention-to-treat aggregate, and the reference
    point of the normalized-skill scale.
    """
    rate = float(np.mean(np.asarray(y_train, dtype=np.float64)))
    const = np.full(len(y_test), rate, dtype=np.float64)
    out = score_predictions(y_test, const)
    # A constant score has no ranking information; both ranking metrics are degenerate, not 'skilful'.
    out["roc_auc"] = 0.5
    return out


def normalized_skill(brier: float, brier_base_rate: float, brier_bayes: float = 0.0) -> Optional[float]:
    """`(Brier_baserate - Brier_method) / (Brier_baserate - Brier_Bayes)`, the pre-registered ROPE scale.

    On the real-data leg no Bayes ceiling exists, so `brier_bayes` defaults to 0.0 and the number is a
    skill fraction against the attainable-in-principle floor; the caller records which case applies.
    """
    denom = brier_base_rate - brier_bayes
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return None
    return float((brier_base_rate - brier) / denom)


def fit_and_score_panel(
    x_train: Any,
    y_train: np.ndarray,
    x_test: Any,
    y_test: np.ndarray,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fit every panel member on `columns` of the training frame and score it on the honest holdout.

    Returns `{"models": {member: metrics-or-error}, "n_model_fits": int, "base_rate": metrics}`.
    `n_model_fits` counts the downstream fits this call performed; it is the deterministic cost axis and
    is accumulated alongside whatever the arm itself reports.
    """
    cols: Optional[List[str]] = list(columns) if columns is not None else None
    if cols is not None and not cols:
        return {"models": {}, "n_model_fits": 0, "base_rate": base_rate_scores(y_train, y_test), "empty_selection": True}

    xtr = x_train[cols] if cols is not None else x_train
    xte = x_test[cols] if cols is not None else x_test

    models: Dict[str, Any] = {}
    fits = 0
    for member, factory in panel_factories().items():
        try:
            clf = factory()
            clf.fit(xtr, y_train)
            fits += 1
            prob = np.asarray(clf.predict_proba(xte))[:, 1]
            models[member] = score_predictions(y_test, prob)
        except Exception as exc:  # a single panel member failing must not lose the other member's row
            logger.warning("panel member %s failed: %s: %s", member, type(exc).__name__, exc)
            models[member] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"models": models, "n_model_fits": fits, "base_rate": base_rate_scores(y_train, y_test), "empty_selection": False}
