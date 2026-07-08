"""iter113 mechanism. Class-balanced baseline disagreement for rare-positive binary classification.

Direct extension of [baseline_disagreement.py](baseline_disagreement.py) (iter69) targeting the
mechanism-boundary gap on rare-positive binary classification, where iter69 HURTS (mammography
1.3% pos: -1.05% CB AUC) due to baseline LGB/Ridge fitting trivial-majority predictions on
folds with too few positives.

Fix: pass `class_weight='balanced'` to all binary baselines. This adjusts the per-row loss
weight inversely to class frequency, forcing the baselines to fit the minority class instead
of trivially predicting the majority. Regression task path is unchanged.

Output is identical to iter69: 3 baseline preds + mean + std + range + lgb_depth_diff +
lgb_vs_linear = 8 features.

Cost: same as iter69 (3 fits per fold).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_3baselines_balanced_predict_on_query(
    Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit 3 baselines with class_weight='balanced' for binary, return (p1, p2, p3)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("baseline_disagreement_balanced requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, class_weight="balanced", random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        p1 = m1.predict_proba(Xq)[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, class_weight="balanced", random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        p2 = m2.predict_proba(Xq)[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            p3 = m3.predict_proba(Xq)[:, 1].astype(np.float32)
        except Exception:
            p3 = np.full(Xq.shape[0], float(y_t.mean()), dtype=np.float32)
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        p1 = m1.predict(Xq).astype(np.float32)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        p2 = m2.predict(Xq).astype(np.float32)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        p3 = m3.predict(Xq).astype(np.float32)
    return p1, p2, p3


def compute_baseline_disagreement_balanced_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "blagreementbal",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Class-balanced baseline disagreement (iter113).

    Output: 8 features per row (identical to iter69 baseline_disagreement).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Core per-fold pipeline: scale, fit 3 balanced baselines (2 LightGBM depths + a linear model) and predict on the query rows, then derive disagreement stats (mean/std/range across the 3, LGBM-pair diff, LGBM-mean-vs-linear diff)."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        p1, p2, p3 = _fit_3baselines_balanced_predict_on_query(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        stack = np.stack([p1, p2, p3], axis=1)
        mean = stack.mean(axis=1)
        std = stack.std(axis=1)
        rng = stack.max(axis=1) - stack.min(axis=1)
        lgb_diff = p1 - p2
        lgb_vs_linear = ((p1 + p2) / 2.0) - p3
        return np.column_stack([p1, p2, p3, mean, std, rng, lgb_diff, lgb_vs_linear])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Split the flat ``_process`` output into the named output columns, cast to the requested output ``dtype``."""
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_p_lgbd3"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_lgbd5"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_linear"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_std"] = feats[:, 4].astype(dtype, copy=False)
        cols[f"{column_prefix}_range"] = feats[:, 5].astype(dtype, copy=False)
        cols[f"{column_prefix}_depth_diff"] = feats[:, 6].astype(dtype, copy=False)
        cols[f"{column_prefix}_lgb_vs_linear"] = feats[:, 7].astype(dtype, copy=False)
        return cols

    n_features = 8

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("baseline_disagreement_balanced: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
