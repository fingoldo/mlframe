"""Model-agnostic conformal prediction intervals + coverage for the suite finalize phase.

The training suite ships only point predictions; this module turns a fitted regressor
plus a held-out calibration set into distribution-free prediction intervals with a
finite-sample marginal-coverage guarantee, and measures achieved coverage on the test
split. It is model-agnostic (reads only predictions + residuals), so it works for any
ordinary estimator; ``CompositeTargetEstimator`` keeps its own richer methods.

Two calibration sources with DIFFERENT guarantees (kept honest, never conflated):
- disjoint calibration slice + the shipped model  -> split-conformal, marginal coverage >= 1-alpha.
- out-of-fold residuals (no held-out slice spent) -> CV+/Jackknife+ flavour, weaker >= 1-2alpha worst case.

Validity also depends on exchangeability, which non-iid splits break; ``infer_split_structure``
tags the structure so the caller dispatches the right variant (split / online / Mondrian).
This first increment implements the regression iid + CV+ paths and the structure tag;
temporal/grouped carving + classification sets are layered on top in later steps.

Reuses ``composite.conformal.conformal_quantile`` (the finite-sample radius), the conditional
sigma-model helpers, and ``metrics.quantile`` coverage metrics -- no re-implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import numpy as np

SplitStructure = str  # one of: "iid", "temporal", "grouped", "temporal_grouped", "stratified"


def infer_split_structure(
    *,
    time_column: Optional[str] = None,
    cv_strategy: Optional[str] = None,
    use_groups: bool = False,
    bucket_stratify: bool = False,
    wholeday_splitting: bool = False,
) -> SplitStructure:
    """Map ``TrainingSplitConfig`` flags to a structure tag driving slice carving + conformal variant.

    A temporal signal (a ``time_column`` or a forward-walk ``cv_strategy``) combined with grouping
    (``use_groups``/``wholeday_splitting``) is ``temporal_grouped``; either alone is ``temporal`` /
    ``grouped``. ``bucket_stratify`` without temporal/group structure is ``stratified``. The default
    (no structure) is ``iid`` -> plain split-conformal is valid.
    """
    temporal = bool(time_column) or str(cv_strategy or "").lower() in ("timeseries", "purged")
    grouped = bool(use_groups) or bool(wholeday_splitting)
    if temporal and grouped:
        return "temporal_grouped"
    if temporal:
        return "temporal"
    if grouped:
        return "grouped"
    if bucket_stratify:
        return "stratified"
    return "iid"


def conformal_supports_split_guarantee(structure: SplitStructure) -> bool:
    """True only for structures where plain split-conformal's exchangeability holds at the row level.

    ``iid`` and ``stratified`` (class-conditional handled by Mondrian-by-class downstream, but the
    pooled marginal guarantee still holds) qualify; ``temporal*``/``grouped`` need online / Mondrian
    variants and return False so the caller does not silently ship invalid marginal coverage.
    """
    return structure in ("iid", "stratified")


def split_conformal_intervals(
    y_pred_test: np.ndarray,
    residuals_cal: np.ndarray,
    alphas: Iterable[float],
    *,
    score: str = "absolute",
    y_pred_cal: Optional[np.ndarray] = None,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Split-conformal intervals per ``alpha`` from calibration residuals.

    ``score="absolute"`` -> constant-width band ``pred +- q``. ``score="normalized"`` ->
    locally-adaptive width ``pred +- q * sigma_hat(pred)`` (restores conditional coverage on
    heteroscedastic targets); requires ``y_pred_cal`` to fit the bin-conditional residual scale.
    """
    from .composite.conformal import conformal_quantile

    y_pred_test = np.asarray(y_pred_test, dtype=np.float64).reshape(-1)
    abs_res = np.abs(np.asarray(residuals_cal, dtype=np.float64).reshape(-1))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    if score == "normalized":
        if y_pred_cal is None:
            raise ValueError("score='normalized' requires y_pred_cal to fit the conditional sigma model")
        from .composite.conformal import _fit_sigma_model, _sigma_for

        y_pred_cal = np.asarray(y_pred_cal, dtype=np.float64).reshape(-1)
        edges, sigma, sig_cal = _fit_sigma_model(y_pred_cal, abs_res)
        sig_test = _sigma_for(edges, sigma, y_pred_test)
        norm_res = abs_res / np.where(sig_cal > 0, sig_cal, 1.0)
        for a in alphas:
            q = conformal_quantile(norm_res, float(a))
            half = q * sig_test
            out[float(a)] = (y_pred_test - half, y_pred_test + half)
        return out

    for a in alphas:
        q = conformal_quantile(abs_res, float(a))
        out[float(a)] = (y_pred_test - q, y_pred_test + q)
    return out


def cv_plus_intervals(
    y_pred_test: np.ndarray,
    oof_residuals: np.ndarray,
    alphas: Iterable[float],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """CV+/Jackknife+ symmetric-score intervals from out-of-fold residuals (no held-out slice spent).

    Honest guarantee is the weaker ``>= 1-2alpha`` (Barber et al. 2021), NOT split-conformal's
    ``1-alpha`` -- the OOF residuals come from K fold-models, not the shipped model, so this is the
    correct label for the no-data-spent path. Symmetric construction: ``pred +- Q`` with ``Q`` the
    finite-sample radius of the pooled OOF absolute residuals.
    """
    from .composite.conformal import conformal_quantile

    y_pred_test = np.asarray(y_pred_test, dtype=np.float64).reshape(-1)
    abs_res = np.abs(np.asarray(oof_residuals, dtype=np.float64).reshape(-1))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for a in alphas:
        q = conformal_quantile(abs_res, float(a))
        out[float(a)] = (y_pred_test - q, y_pred_test + q)
    return out


def coverage_report(
    y_true_test: np.ndarray,
    intervals_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]],
) -> dict[float, dict[str, float]]:
    """Per-alpha achieved coverage / mean width / Winkler on the honest test split."""
    from ..metrics.quantile import coverage, mean_interval_width, winkler_score

    y_true_test = np.asarray(y_true_test, dtype=np.float64).reshape(-1)
    rep: dict[float, dict[str, float]] = {}
    for a, (lo, hi) in intervals_by_alpha.items():
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        finite = np.isfinite(lo) & np.isfinite(hi)
        rep[float(a)] = {
            "nominal_coverage": float(1.0 - a),
            "achieved_coverage": float(coverage(y_true_test, lo, hi)),
            "mean_width": float(mean_interval_width(lo, hi)) if finite.any() else float("inf"),
            "winkler": float(winkler_score(y_true_test, lo, hi, float(a))),
            "frac_finite": float(np.mean(finite)),
        }
    return rep


def conformal_classification_report(
    *,
    test_probs: np.ndarray,
    test_target: np.ndarray,
    calib_probs: np.ndarray,
    calib_target: np.ndarray,
    classes: np.ndarray,
    alphas: Sequence[float] = (0.1,),
    score: str = "lac",
    structure: SplitStructure = "iid",
) -> dict[str, Any]:
    """Split-conformal prediction SETS for classification with finite-sample marginal coverage >= 1-alpha.

    ``score="lac"`` (least-ambiguous): nonconformity = ``1 - p(true|x)`` -> smallest sets. ``score="aps"``
    (adaptive): cumulative sorted-softmax mass up to the true label -> better conditional coverage, larger
    sets. Calibrated on the disjoint calib slice; coverage = fraction of test rows whose true label is in
    the predicted set, plus mean set size (efficiency). Reuses the standalone ``conformal_set_threshold``.
    """
    from .composite.conformal_classification import conformal_set_threshold

    cp = np.asarray(calib_probs, dtype=np.float64)
    tp = np.asarray(test_probs, dtype=np.float64)
    classes = np.asarray(classes)
    k = classes.size
    if cp.ndim != 2 or tp.ndim != 2 or cp.shape[1] != k or tp.shape[1] != k:
        raise ValueError(f"probs must be (n, {k}) to match {k} classes; got calib {cp.shape}, test {tp.shape}")
    cls_index = {c: i for i, c in enumerate(classes.tolist())}
    cal_true = np.array([cls_index.get(c, -1) for c in np.asarray(calib_target).tolist()])
    test_true = np.array([cls_index.get(c, -1) for c in np.asarray(test_target).tolist()])
    valid = cal_true >= 0
    cp, cal_true = cp[valid], cal_true[valid]

    def _aps_scores(probs: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        csum = np.cumsum(sorted_p, axis=1)
        rank_of_true = np.argmax(order == true_idx[:, None], axis=1)
        return csum[np.arange(probs.shape[0]), rank_of_true]

    if score == "aps":
        cal_scores = _aps_scores(cp, cal_true)
    else:
        cal_scores = 1.0 - cp[np.arange(cp.shape[0]), cal_true]

    per_alpha: dict[float, dict[str, float]] = {}
    for a in alphas:
        thr = conformal_set_threshold(cal_scores, float(a))
        if score == "aps":
            order = np.argsort(-tp, axis=1)
            sorted_p = np.take_along_axis(tp, order, axis=1)
            csum = np.cumsum(sorted_p, axis=1)
            in_set_sorted = csum <= thr
            in_set_sorted[:, 0] = True  # always keep the top label (non-empty set)
            in_set = np.zeros_like(in_set_sorted)
            np.put_along_axis(in_set, order, in_set_sorted, axis=1)
        else:
            in_set = (1.0 - tp) <= thr
            in_set[np.arange(tp.shape[0]), np.argmax(tp, axis=1)] = True  # non-empty fallback
        covered = in_set[np.arange(tp.shape[0]), test_true] & (test_true >= 0)
        per_alpha[float(a)] = {
            "nominal_coverage": float(1.0 - a),
            "achieved_coverage": float(np.mean(covered)),
            "mean_set_size": float(np.mean(in_set.sum(axis=1))),
        }
    return {
        "method": "conformal_set",
        "score": score,
        "structure": structure,
        "guarantee": "marginal>=1-alpha",
        "split_conformal_valid_for_structure": conformal_supports_split_guarantee(structure),
        "alphas": [float(a) for a in alphas],
        "n_classes": int(k),
        "per_alpha": per_alpha,
    }


def conformal_regression_report(
    *,
    y_pred_test: np.ndarray,
    y_true_test: np.ndarray,
    residuals_cal: Optional[np.ndarray] = None,
    y_pred_cal: Optional[np.ndarray] = None,
    oof_residuals: Optional[np.ndarray] = None,
    alphas: Sequence[float] = (0.1,),
    score: str = "normalized",
    structure: SplitStructure = "iid",
) -> dict[str, Any]:
    """Orchestrate: build intervals (split-conformal if a calib slice is given, else CV+) + coverage.

    Returns a metadata-ready dict; the slice-vs-OOF choice fixes the guarantee label so a reader
    never mistakes the weaker CV+ bound for split-conformal. Falls back to absolute score when the
    normalized sigma model is not computable (no ``y_pred_cal``).
    """
    if residuals_cal is not None and np.asarray(residuals_cal).size > 0:
        eff_score = score if (score == "normalized" and y_pred_cal is not None) else "absolute"
        intervals = split_conformal_intervals(
            y_pred_test,
            residuals_cal,
            alphas,
            score=eff_score,
            y_pred_cal=y_pred_cal,
        )
        method = "split_conformal"
        guarantee = "marginal>=1-alpha"
    elif oof_residuals is not None and np.asarray(oof_residuals).size > 0:
        eff_score = "absolute"
        intervals = cv_plus_intervals(y_pred_test, oof_residuals, alphas)
        method = "cv_plus"
        guarantee = "marginal>=1-2alpha"
    else:
        raise ValueError("conformal_regression_report needs residuals_cal or oof_residuals")

    return {
        "method": method,
        "score": eff_score,
        "structure": structure,
        "guarantee": guarantee,
        "split_conformal_valid_for_structure": conformal_supports_split_guarantee(structure),
        "alphas": [float(a) for a in alphas],
        "per_alpha": coverage_report(y_true_test, intervals),
        "intervals": {float(a): (lo, hi) for a, (lo, hi) in intervals.items()},
    }
