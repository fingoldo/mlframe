"""Conditional-redundancy pass of the hybrid selector: drop a kept column that is a near-twin of another kept column and adds nothing to a held-out fit once the rest is present.

The hybrid vote clusters columns whose absolute correlation reaches ``corr_thr`` (0.92) and keeps one representative per cluster. A noisy copy of a kept column that sits just below that
threshold (the bridge ``b = a + 0.6 e`` of a graded chain a ~ b ~ c has correlation 0.86 with ``a``) is never clustered with it, and any member of the vote can pick it, so it takes a slot
beside the column it copies. This pass looks only at such pairs: a kept column ``x`` with another kept column ``z`` at ``|corr| >= twin_corr`` is dropped when a held-out linear fit of the target on
all the kept columns without ``x`` is not worse than the fit with ``x`` by more than its own sampling error (a one-sided paired test on the per-row losses, at level ``alpha`` corrected for the
number of candidates). Columns without a high-correlation partner are never tested, so the pass cannot remove a column that carries a private (for instance interaction) signal.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_FOLDS = 3
_MIN_ROWS = 60


def _row_losses(Z: np.ndarray, y: np.ndarray, classification: bool, folds: np.ndarray, model_kind: str = "linear") -> Optional[np.ndarray]:
    """Held-out loss of every row (log loss or absolute error) of a model on the standardised columns ``Z`` (``"linear"``: logistic / ridge, ``"hgb"``: gradient-boosted trees); ``None`` when the fit is not possible."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    out = np.empty(Z.shape[0], dtype=np.float64)  # Intentional float64: statistic vector (per-row losses entering a mean and a standard error)
    classes = np.unique(y) if classification else None
    for fo in range(_FOLDS):
        tr, va = folds != fo, folds == fo
        if classification:
            if np.unique(y[tr]).size < 2:
                return None
            model = (HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=0) if model_kind == "hgb" else LogisticRegression(max_iter=300)).fit(Z[tr], y[tr])
            proba = np.clip(model.predict_proba(Z[va]), 1e-12, 1.0)
            idx = np.searchsorted(model.classes_, y[va])
            out[va] = -np.log(proba[np.arange(proba.shape[0]), np.clip(idx, 0, proba.shape[1] - 1)])
        else:
            model = (HistGradientBoostingRegressor(max_iter=60, max_depth=4, random_state=0) if model_kind == "hgb" else Ridge(alpha=1.0)).fit(Z[tr], y[tr])
            out[va] = np.abs(y[va] - model.predict(Z[va]))
    del classes
    return out


def _gain_is_significant(without: np.ndarray, with_x: np.ndarray, alpha: float, n_candidates: int) -> bool:
    """Whether the mean per-row loss reduction from ``without`` to ``with_x`` is significant at one-sided ``alpha / n_candidates`` (a normal approximation of the paired t statistic)."""
    from scipy.stats import norm

    d = without - with_x
    d = d[np.isfinite(d)]
    if d.size < 3:
        return True
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    if sd <= 1e-15:
        return mean > 0.0
    return bool(mean / (sd / np.sqrt(d.size)) >= float(norm.isf(float(alpha) / max(1, int(n_candidates)))))


def prune_conditional_twins(
    X: Any, y: Any, selected: Sequence[str], fi: dict, *, twin_corr: float = 0.8, alpha: float = 0.05, classification: Optional[bool] = None, seed: int = 0, model_kind: str = "linear"
) -> list:
    """``selected`` without the near-twin columns that add nothing given the others (see the module docstring); the order of ``selected`` is kept.

    ``X`` is the frame the columns come from (pandas), ``fi`` the shared importance used to decide which member of a twin pair is the candidate (the lower one; ties by name, later dropped).
    """
    try:
        import pandas as pd

        selected = list(selected)
        numeric = [c for c in selected if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        if len(numeric) < 2:
            return selected
        n = len(X)
        if n < _MIN_ROWS:
            return selected
        yv = np.asarray(y).ravel()
        if classification is None:
            from sklearn.utils.multiclass import type_of_target

            classification = type_of_target(yv) in ("binary", "multiclass")
        frame = X[numeric].to_numpy(dtype=np.float64)
        frame = np.where(np.isfinite(frame), frame, np.nanmean(np.where(np.isfinite(frame), frame, np.nan), axis=0)[None, :])
        sd = frame.std(axis=0)
        if np.any(sd <= 1e-12):
            keep_idx = np.flatnonzero(sd > 1e-12)
            numeric = [numeric[i] for i in keep_idx]
            frame, sd = frame[:, keep_idx], sd[keep_idx]
            if len(numeric) < 2:
                return selected
        Z = (frame - frame.mean(axis=0)) / sd
        corr = np.abs(np.corrcoef(Z, rowvar=False))
        np.fill_diagonal(corr, 0.0)
        rank = {c: (float(fi.get(c, 0.0)), str(c)) for c in numeric}
        candidates = []
        for i, c in enumerate(numeric):
            partners = [numeric[j] for j in np.flatnonzero(corr[i] >= float(twin_corr))]
            # the lower-importance member of a twin pair (ties: the later name) is the candidate to drop
            if any(rank[c] < rank[p] or (rank[c] == rank[p] and str(c) > str(p)) for p in partners):
                candidates.append(c)
        if not candidates:
            return selected
        rng = np.random.default_rng(int(seed))
        folds = np.arange(n) % _FOLDS
        rng.shuffle(folds)
        keep = list(numeric)
        dropped: list = []
        for c in sorted(candidates, key=lambda col: rank[col]):
            if c not in keep or len(keep) < 2:
                continue
            idx_all = [numeric.index(k) for k in keep]
            idx_wo = [numeric.index(k) for k in keep if k != c]
            with_x = _row_losses(Z[:, idx_all], yv, bool(classification), folds, model_kind)
            without = _row_losses(Z[:, idx_wo], yv, bool(classification), folds, model_kind)
            if with_x is None or without is None:
                continue
            if not _gain_is_significant(without, with_x, alpha, len(candidates)):
                keep.remove(c)
                dropped.append(c)
        if dropped:
            logger.info("hybrid selector: dropped %s as near-twin(s) that add nothing given the other selected columns", dropped)
        return [c for c in selected if c not in dropped]
    except Exception as exc:  # the pass is a refinement: a failure leaves the vote's selection as it was
        logger.warning("hybrid selector: the conditional-twin pass failed (%s: %s); the selection is left as it was", type(exc).__name__, exc)
        return list(selected)
