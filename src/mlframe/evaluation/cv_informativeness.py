"""CV-informativeness sanity check: does group-CV carry any real cross-group signal at all?

A common silent failure mode in group-based CV (GroupKFold across chunks/time-windows/entities) is that the
held-out group's target is essentially UNRELATED to the other groups -- so no amount of modeling on the
training groups beats a trivial baseline that just reads off target STATISTICS FROM the held-out group itself
(a "leaked" dummy, using information a real held-out-group prediction could never have). A tabular-playground
writeup hit exactly this: none of several model families beat a leaked-stats DummyRegressor on the held-out
chunk, meaning group-CV had zero cross-group informativeness and no CV-driven decision was trustworthy for
model selection -- the team switched entirely to LB-based selection. This diagnostic operationalizes that
sanity check: for each fold, compare a real model's held-out score against the LEAKED-DUMMY baseline computed
FROM the held-out fold's own target, and reports whether the real model actually clears it.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Tuple

import numpy as np


def _classify_fold_trend(fold_sizes: np.ndarray, margins: np.ndarray, corr_threshold: float) -> dict:
    """Classify whether per-fold informativeness margins decay as held-out fold size shrinks.

    ``margin`` is ``real_score - dummy_score`` oriented so positive always means "beat the dummy" (already
    flipped by the caller for ``maximize=False``). Distinguishes two failure modes that call for different
    remediation:

    - ``uniformly_uninformative``: every fold is about equally (un)informative regardless of its size --
      a fundamentally bad target/feature relationship, not fixable by re-folding.
    - ``decaying_with_group_sparsity``: informativeness rises with fold size (smaller held-out groups are
      the uninformative ones) -- likely fixable by using fewer, larger folds.
    - ``no_clear_trend``: neither pattern is clean enough to call (e.g. too few folds, or all fold sizes equal).
    """
    n_folds = len(fold_sizes)
    if n_folds < 3 or np.ptp(fold_sizes) == 0:
        correlation = float("nan")
        classification = "no_clear_trend"
    else:
        # np.corrcoef returns nan if either series is constant (e.g. all margins identical).
        with np.errstate(invalid="ignore"):
            correlation = float(np.corrcoef(fold_sizes, margins)[0, 1])
        if np.isnan(correlation):
            classification = "uniformly_uninformative" if np.all(margins <= 0) else "no_clear_trend"
        elif correlation >= corr_threshold:
            classification = "decaying_with_group_sparsity"
        elif np.all(margins <= 0):
            classification = "uniformly_uninformative"
        else:
            classification = "no_clear_trend"

    return {
        "fold_sizes": fold_sizes.tolist(),
        "margins": margins.tolist(),
        "correlation": correlation,
        "classification": classification,
    }


def cv_informativeness_check(
    X: Any,
    y: np.ndarray,
    cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool = True,
    leaked_dummy_stat: str = "mean",
    check_trend: bool = False,
    trend_corr_threshold: float = 0.5,
) -> dict:
    """Per-fold, compare a real model's held-out score against a leaked-stats dummy baseline.

    Parameters
    ----------
    X, y
        Full feature/target arrays.
    cv_splits
        Iterable of ``(train_idx, test_idx)`` index-array pairs (e.g. from a fitted splitter's ``.split()``).
    model_factory
        Zero-arg factory returning a fresh sklearn-compatible estimator (``.fit(X, y)`` / ``.predict(X)``).
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``.
    maximize
        ``True`` when higher ``metric_fn`` is better; ``False`` for a loss.
    leaked_dummy_stat
        ``"mean"`` or ``"median"`` -- the leaked-dummy prediction is this statistic of ``y[test_idx]`` itself
        (impossible in real deployment; the point is it's an UPPER-INFORMATIVENESS-FLOOR the real model,
        which only sees TRAIN data, should still be measured against).
    check_trend
        Opt-in (default ``False``, output is otherwise unchanged). When ``True``, additionally reports whether
        the per-fold informativeness margin (``real_score`` vs ``dummy_score``, oriented positive-is-better)
        trends with held-out fold size -- distinguishing "uniformly uninformative CV" (every fold equally bad,
        likely a fundamentally bad target/feature setup, not fixable by re-folding) from "informativeness
        decaying with group sparsity" (smaller folds are the uninformative ones, likely fixable by using fewer,
        larger folds). See ``trend_diagnostic`` in the return value.
    trend_corr_threshold
        Only used when ``check_trend=True``. Pearson correlation between fold size and margin at or above this
        threshold is classified as ``"decaying_with_group_sparsity"``.

    Returns
    -------
    dict
        ``fold_results`` (list of dicts: ``real_score``, ``dummy_score``, ``beats_dummy``),
        ``fraction_folds_informative`` (fraction of folds where the real model beat the leaked dummy),
        ``informative`` (bool: ``fraction_folds_informative`` >= 0.5 -- CV carries real cross-group signal;
        ``False`` is the writeup's red flag -- CV-driven decisions are untrustworthy, consider an alternative
        selection signal),
        ``trend_diagnostic`` (only present when ``check_trend=True``: dict with ``fold_sizes``, ``margins``,
        ``correlation``, ``classification`` -- one of ``"uniformly_uninformative"``,
        ``"decaying_with_group_sparsity"``, ``"no_clear_trend"``).
    """
    if leaked_dummy_stat not in ("mean", "median"):
        raise ValueError(f"cv_informativeness_check: leaked_dummy_stat must be 'mean' or 'median'; got {leaked_dummy_stat!r}")

    y = np.asarray(y)
    is_frame = hasattr(X, "iloc")
    fold_results = []
    fold_sizes = []
    margins = []
    for train_idx, test_idx in cv_splits:
        X_train = X.iloc[train_idx] if is_frame else X[train_idx]
        X_test = X.iloc[test_idx] if is_frame else X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        real_pred = model.predict(X_test)
        real_score = float(metric_fn(y_test, real_pred))

        dummy_value = float(np.mean(y_test)) if leaked_dummy_stat == "mean" else float(np.median(y_test))
        dummy_pred = np.full_like(y_test, dummy_value, dtype=np.float64)
        dummy_score = float(metric_fn(y_test, dummy_pred))

        beats_dummy = real_score > dummy_score if maximize else real_score < dummy_score
        fold_results.append({"real_score": real_score, "dummy_score": dummy_score, "beats_dummy": beats_dummy})
        if check_trend:
            fold_sizes.append(len(test_idx))
            margins.append(real_score - dummy_score if maximize else dummy_score - real_score)

    fraction_informative = float(np.mean([r["beats_dummy"] for r in fold_results])) if fold_results else float("nan")
    result = {
        "fold_results": fold_results,
        "fraction_folds_informative": fraction_informative,
        "informative": fraction_informative >= 0.5,
    }
    if check_trend:
        result["trend_diagnostic"] = _classify_fold_trend(
            np.asarray(fold_sizes, dtype=np.float64), np.asarray(margins, dtype=np.float64), trend_corr_threshold
        )
    return result


__all__ = ["cv_informativeness_check"]
