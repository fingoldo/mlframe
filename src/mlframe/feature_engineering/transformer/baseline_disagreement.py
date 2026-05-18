"""Baseline-disagreement-as-feature: per-query predictions + disagreement stats over 3 baselines.

Iter 69 mechanism. Structurally orthogonal to iter 60-68 anchor/band routing — uses ensemble
disagreement DIRECTLY as a feature, not as anchor-selection criterion.

Mechanism:
1. Fit 3 baselines on (X_train, y_train):
   - LGB depth=3 (50 estimators) — shallow boosting
   - LGB depth=5 (50 estimators) — deeper boosting
   - Ridge regression (regression) or logistic regression (binary) — linear model class
2. Per query, predict using each baseline → 3 predictions per query.
3. Compute disagreement statistics:
   - 3 individual baseline predictions
   - mean of 3 predictions
   - std of 3 predictions (= ensemble disagreement)
   - range (max - min)
   - LGB_depth3 - LGB_depth5 (depth-disagreement: captures non-linearity scale)
   - LGB_avg - Ridge (boosting-vs-linear disagreement: captures interaction signal)

8 features total per query.

Why this is structurally novel:
- Iter 60-68 use baseline residuals on TRAIN rows to select anchors, then route query via softmax similarity.
- Iter 69 uses baseline PREDICTIONS on QUERY rows directly + cross-baseline disagreement statistics.
- "Is this query in a region where models disagree?" — high std → uncertain region; downstream boosting
  can use this as a meta-feature to allocate capacity to uncertain regions.

Leakage discipline:
- Mode A (X_query=None): per fold, fit baselines on train_idx → predict on val_idx. Predictions on val
  rows come from baselines that never saw those rows' labels. Honest OOF.
- Mode B (X_query given): fit baselines on full X_train → predict on X_query. X_query labels never used.

Cost: 3× baseline fits per fold + predictions. Sub-second.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_3baselines_predict_on_query(
    Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit 3 baselines on (Xt, y_t), predict on Xq. Return (p1, p2, p3) each shape (n_q,)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("baseline_disagreement requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        p1 = m1.predict_proba(Xq)[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1,
                                random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        p2 = m2.predict_proba(Xq)[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            p3 = m3.predict_proba(Xq)[:, 1].astype(np.float32)
        except Exception:
            p3 = np.full(Xq.shape[0], float(y_t.mean()), dtype=np.float32)
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                               random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        p1 = m1.predict(Xq).astype(np.float32)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                               random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        p2 = m2.predict(Xq).astype(np.float32)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        p3 = m3.predict(Xq).astype(np.float32)
    return p1, p2, p3


def compute_baseline_disagreement_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "blagreement",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Baseline-disagreement-as-feature.

    Output: 3 baseline preds + mean + std + range + lgb_diff + lgb_vs_linear = 8 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        p1, p2, p3 = _fit_3baselines_predict_on_query(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        stack = np.stack([p1, p2, p3], axis=1)
        mean = stack.mean(axis=1)
        std = stack.std(axis=1)
        rng = stack.max(axis=1) - stack.min(axis=1)
        lgb_diff = p1 - p2          # shallow vs deep
        lgb_vs_linear = ((p1 + p2) / 2.0) - p3  # boosting average vs linear
        return np.column_stack([p1, p2, p3, mean, std, rng, lgb_diff, lgb_vs_linear])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
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
    out = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("baseline_disagreement: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
