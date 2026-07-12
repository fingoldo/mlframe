"""Empirically pick which CV scheme actually predicts out-of-time performance -- don't just follow the textbook rule.

"Prefer GroupKFold for correlated rows" is good general advice, but not universally true: an AmExpert-2019
team found plain KFold gave BETTER out-of-time results than GroupKFold for their data. Rather than picking a
CV scheme by rule-of-thumb, this diagnostic fits a quick model under each candidate scheme and reports which
one's CV score most closely tracks a genuine out-of-time holdout score -- the empirically correct choice for
THIS dataset, not the textbook-default one.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from mlframe.evaluation.cv_delta_triage import triage_cv_delta


def compare_cv_schemes(
    X: Any,
    y: np.ndarray,
    schemes: Dict[str, Iterable[Tuple[np.ndarray, np.ndarray]]],
    ooo_time_idx: Tuple[np.ndarray, np.ndarray],
    model_factory: Callable[[], Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    significance_alpha: Optional[float] = None,
) -> dict:
    """Score several candidate CV schemes against a genuine out-of-time holdout; report the best proxy.

    Parameters
    ----------
    X, y
        Full feature/target arrays.
    schemes
        ``{scheme_name: cv_splits}`` -- each ``cv_splits`` is an iterable of ``(train_idx, test_idx)`` index
        pairs (e.g. from ``KFold().split(X)``, ``GroupKFold().split(X, groups=...)``,
        ``TimeSeriesSplit().split(X)``).
    ooo_time_idx
        ``(train_idx, holdout_idx)`` for a SEPARATE, genuine out-of-time holdout (e.g. the most recent chunk
        of data, held out from all `schemes`' folds) -- the ground truth every scheme's CV score is compared
        against.
    model_factory
        Zero-arg factory returning a fresh sklearn-compatible estimator (``.fit(X, y)`` / ``.predict(X)``).
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``, LOWER is better (e.g. RMSE, log-loss) -- used consistently
        for both each scheme's mean CV score and the out-of-time holdout score, so they're directly comparable.
    significance_alpha
        Opt-in. When ``None`` (default), ``best_scheme`` is picked purely by point-estimate ``gap_to_ooo_time``
        as before -- output is bit-identical to the pre-existing behavior. When set to a float, each OTHER
        scheme is paired-tested (by fold index) against the point-estimate winner using
        :func:`mlframe.evaluation.cv_delta_triage.triage_cv_delta` on their per-fold ``|fold_score -
        ooo_time_score|`` gaps (reusing the existing CV noise-band machinery rather than re-deriving a
        significance test) -- the point-estimate winner's smaller gap must clear the fold-resampling noise
        floor to count as a genuine win, not just a lower number. Requires every scheme in ``schemes`` to
        produce the SAME number of folds (a paired test needs fold-to-fold correspondence); raises
        ``ValueError`` otherwise.

    Returns
    -------
    dict
        ``ooo_time_score`` (the ground-truth out-of-time score), ``scheme_scores`` (``{name: {"cv_score",
        "gap_to_ooo_time"}}``), ``best_scheme`` (the name whose ``cv_score`` is closest to
        ``ooo_time_score``, i.e. the most trustworthy CV proxy for this dataset). When ``significance_alpha``
        is given, also: ``significance`` (``{other_name: triage_cv_delta_result}`` for every non-winning
        scheme, comparing its fold-level gap against the point-estimate winner's) and
        ``best_scheme_significant`` (``True`` only if the point-estimate winner's gap advantage clears the
        noise band against EVERY other scheme; ``False`` if any comparison is inconclusive, ``None`` when
        there are fewer than 2 schemes to compare).
    """
    is_frame = hasattr(X, "iloc")

    def _fit_eval(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
        X_train = X.iloc[train_idx] if is_frame else X[train_idx]
        X_test = X.iloc[test_idx] if is_frame else X[test_idx]
        model = model_factory()
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        return float(metric_fn(y[test_idx], pred))

    ooo_train_idx, ooo_holdout_idx = ooo_time_idx
    ooo_time_score = _fit_eval(ooo_train_idx, ooo_holdout_idx)

    scheme_scores: Dict[str, dict] = {}
    fold_gaps: Dict[str, np.ndarray] = {}
    for name, cv_splits in schemes.items():
        fold_scores = [_fit_eval(train_idx, test_idx) for train_idx, test_idx in cv_splits]
        cv_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
        scheme_scores[name] = {"cv_score": cv_score, "gap_to_ooo_time": abs(cv_score - ooo_time_score)}
        fold_gaps[name] = np.abs(np.asarray(fold_scores, dtype=np.float64) - ooo_time_score)

    best_scheme = min(scheme_scores, key=lambda n: scheme_scores[n]["gap_to_ooo_time"]) if scheme_scores else None

    result: dict = {"ooo_time_score": ooo_time_score, "scheme_scores": scheme_scores, "best_scheme": best_scheme}

    if significance_alpha is not None:
        other_names = [n for n in schemes if n != best_scheme]
        significance: Dict[str, dict] = {}
        best_scheme_significant: Optional[bool]
        if best_scheme is None or not other_names:
            best_scheme_significant = None
        else:
            best_gain = -fold_gaps[best_scheme]
            best_scheme_significant = True
            for other in other_names:
                other_gain = -fold_gaps[other]
                if other_gain.shape != best_gain.shape:
                    raise ValueError(
                        "compare_cv_schemes: significance_alpha requires every scheme to produce the same "
                        f"number of folds for a paired test; {best_scheme!r} has {best_gain.shape[0]} fold(s), "
                        f"{other!r} has {other_gain.shape[0]}"
                    )
                triage = triage_cv_delta(
                    baseline_fold_scores=other_gain,
                    candidate_fold_scores=best_gain,
                    change_source="feature_engineering",
                    alpha=significance_alpha,
                )
                significance[other] = triage
                if not (triage["actionable"] and triage["delta"] > 0):
                    best_scheme_significant = False
        result["significance"] = significance
        result["best_scheme_significant"] = best_scheme_significant

    return result


__all__ = ["compare_cv_schemes"]
