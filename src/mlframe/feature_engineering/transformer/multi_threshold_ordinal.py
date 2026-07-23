"""Multi-threshold ordinal stack: K binary classifiers at quantile thresholds.

Iter 96 mechanism. Agent C #4 ranked.

For regression: K=7 thresholds at {0.1, 0.25, ..., 0.9}; fit K OOF binary classifiers for 1{y > t_k}.
For binary: K=3 stratified sub-population classifiers.

Per query emit 5 features: max(p_k), entropy of (p_k), monotonicity-violations count, mean(p_k),
quantile-y-prediction (interpolated rank).
"""
from __future__ import annotations
import logging
from typing import Any, Literal, Optional
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_multi_threshold_ordinal_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray] = None,
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "multthr",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-threshold ordinal stack: fit K binary LightGBM classifiers (regression quantile thresholds, or binary top-3-importance sub-population splits), then aggregate their per-query probabilities into max/mean/entropy, monotonicity-violation count, and interpolated quantile-rank features."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("multi_threshold_ordinal requires lightgbm") from exc
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        """Fit the per-threshold (or per-sub-population) classifiers on the (optionally scaled) train fold and derive the 5 aggregate query-row features from their predicted probabilities."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        if task == "regression":
            thresholds = np.quantile(y_t, [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
            preds = np.zeros((Xq_s.shape[0], len(thresholds)), dtype=np.float32)
            for i, t in enumerate(thresholds):
                target = (y_t > t).astype(np.int32)
                if target.sum() == 0 or target.sum() == target.shape[0]:
                    preds[:, i] = float(target.mean())
                    continue
                m = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1,
                                       random_state=int(fold_seed) + i, verbose=-1, n_jobs=-1).fit(Xt_s, target)
                preds[:, i] = np.asarray(m.predict_proba(Xq_s))[:, 1].astype(np.float32)
        else:
            # binary: 3 sub-population classifiers (top-3 importance features)
            preds = np.zeros((Xq_s.shape[0], 3), dtype=np.float32)
            m_base = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            importances = m_base.feature_importances_
            # Wave 62 (2026-05-20): lexsort with feature-index tiebreak so tied
            # LGB feature_importances_ (often rounded to ints) give deterministic
            # top-3 across runs.
            _imp = np.asarray(importances)
            top3 = np.lexsort((-np.arange(len(_imp)), _imp))[-3:]
            for i, j in enumerate(top3):
                median_j = float(np.median(Xt_s[:, j]))
                target = ((y_t > 0.5) & (Xt_s[:, j] > median_j)).astype(np.int32)
                if target.sum() == 0:
                    preds[:, i] = float(target.mean())
                    continue
                m = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1,
                                       random_state=int(fold_seed) + i + 10, verbose=-1, n_jobs=-1).fit(Xt_s, target)
                preds[:, i] = np.asarray(m.predict_proba(Xq_s))[:, 1].astype(np.float32)
        max_p = preds.max(axis=1).astype(np.float32)
        mean_p = preds.mean(axis=1).astype(np.float32)
        p_clip = np.clip(preds, 1e-6, 1 - 1e-6)
        entropy = (-p_clip * np.log(p_clip) - (1 - p_clip) * np.log(1 - p_clip)).mean(axis=1).astype(np.float32)
        # Monotonicity violations: count adjacent pairs where p[k] < p[k+1] (should be non-increasing for ordinal targets)
        diffs = np.diff(preds, axis=1)
        n_violations = (diffs > 0).sum(axis=1).astype(np.float32)
        # Quantile-y-prediction: interpolated rank where p crosses 0.5. Vectorised: np.argmax on a
        # boolean array returns the FIRST True index (matching the per-row np.where(...)[0][0] this
        # replaced), but also returns 0 on an all-False row (no crossing) -- the has_cross mask
        # overrides those rows to preds.shape[1], matching the original per-row fallback exactly.
        below_half = preds < 0.5
        has_cross = below_half.any(axis=1)
        first_cross = np.argmax(below_half, axis=1)
        rank_pred = np.where(has_cross, first_cross, preds.shape[1]).astype(np.float32)
        return np.column_stack([max_p, mean_p, entropy, n_violations, rank_pred])

    def _make_df(feats):
        """Slice the ``_process`` output columns into a name-tagged dict (max_p/mean_p/entropy/n_violations/rank_pred), cast to the requested output dtype."""
        cols = {}
        cols[f"{column_prefix}_max_p"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_p"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_entropy"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_violations"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_rank_pred"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f, seed)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100).astype(dtype, copy=False)
        logger.info("multi_threshold_ordinal: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
