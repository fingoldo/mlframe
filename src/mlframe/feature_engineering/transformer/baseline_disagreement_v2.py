"""iter102 mechanism. Baseline-disagreement v2: iter69 + ExtraTrees as 4th orthogonal baseline.

Direct extension of [baseline_disagreement.py](baseline_disagreement.py) (iter69, which is the
strongest-surviving transformer-FE record: +4.92% R2 median on year-prediction 100k, multi-seed
generalization across 3 regression datasets of 25x size range).

Why ExtraTrees: LGB's gradient-based splits and Ridge's linear projection cover two ends of the
bias/variance spectrum, but BOTH operate on global gradient signal. ExtraTrees uses RANDOMIZED
splits (variance from random feature + random threshold per split), so its predictions encode
a fundamentally different inductive bias than LGB. Regions where LGB and ExtraTrees disagree
are regions where the gradient-based vs random-tree partitioning of feature space matters - and
those regions carry information boostings cannot reconstruct from raw X.

Mechanism:
1. Fit 4 baselines on (X_train, y_train):
   - LGB depth=3 (50 estimators) - shallow gradient boosting
   - LGB depth=5 (50 estimators) - deeper gradient boosting
   - Ridge / LogisticRegression - linear model
   - ExtraTreesRegressor / Classifier (100 estimators, max_depth=8) - randomized split forest
2. Per query, predict using each baseline -> 4 predictions per query.
3. Compute 11 features:
   - 4 raw baseline predictions (p_lgbd3, p_lgbd5, p_linear, p_et)
   - mean of 4 predictions
   - std of 4 predictions (disagreement, broader than iter69's 3-model std)
   - range (max - min)
   - LGB depth-diff: p_lgbd3 - p_lgbd5
   - LGB vs linear: (p_lgbd3 + p_lgbd5)/2 - p_linear
   - LGB vs ET: (p_lgbd3 + p_lgbd5)/2 - p_et (boost-vs-random-forest gap)
   - ET vs linear: p_et - p_linear (random-tree vs linear gap)

Leakage discipline (same as iter69):
- Mode A: per-fold refit on train_idx, predict on val_idx.
- Mode B: fit on full X_train, predict on X_query.

Cost: 4 baseline fits per fold vs iter69's 3. ExtraTrees on n=100k with d~100 is ~10s; total
overhead ~25% over iter69. Acceptable for the additional signal source.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_4baselines_predict_on_query(
    Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit 4 baselines on (Xt, y_t), predict on Xq. Return (p_lgbd3, p_lgbd5, p_linear, p_et)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("baseline_disagreement_v2 requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        p1 = m1.predict_proba(Xq)[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        p2 = m2.predict_proba(Xq)[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            p3 = m3.predict_proba(Xq)[:, 1].astype(np.float32)
        except Exception:
            p3 = np.full(Xq.shape[0], float(y_t.mean()), dtype=np.float32)
        m4 = ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=int(seed) + 3, n_jobs=-1)
        m4.fit(Xt, y_t.astype(np.int32))
        p4 = m4.predict_proba(Xq)[:, 1].astype(np.float32)
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
        m4 = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=int(seed) + 3, n_jobs=-1)
        m4.fit(Xt, y_t)
        p4 = m4.predict(Xq).astype(np.float32)
    return p1, p2, p3, p4


def compute_baseline_disagreement_v2_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "blagreementv2",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Baseline-disagreement v2 (iter102): iter69 + ExtraTrees.

    Output: 4 baseline preds + mean + std + range + 4 pairwise differences = 12 features.
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
        p1, p2, p3, p4 = _fit_4baselines_predict_on_query(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        stack = np.stack([p1, p2, p3, p4], axis=1)
        mean = stack.mean(axis=1)
        std = stack.std(axis=1)
        rng = stack.max(axis=1) - stack.min(axis=1)
        lgb_diff = p1 - p2
        lgb_vs_linear = ((p1 + p2) / 2.0) - p3
        lgb_vs_et = ((p1 + p2) / 2.0) - p4
        et_vs_linear = p4 - p3
        return np.column_stack([p1, p2, p3, p4, mean, std, rng, lgb_diff, lgb_vs_linear, lgb_vs_et, et_vs_linear])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_p_lgbd3"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_lgbd5"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_linear"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_et"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean"] = feats[:, 4].astype(dtype, copy=False)
        cols[f"{column_prefix}_std"] = feats[:, 5].astype(dtype, copy=False)
        cols[f"{column_prefix}_range"] = feats[:, 6].astype(dtype, copy=False)
        cols[f"{column_prefix}_depth_diff"] = feats[:, 7].astype(dtype, copy=False)
        cols[f"{column_prefix}_lgb_vs_linear"] = feats[:, 8].astype(dtype, copy=False)
        cols[f"{column_prefix}_lgb_vs_et"] = feats[:, 9].astype(dtype, copy=False)
        cols[f"{column_prefix}_et_vs_linear"] = feats[:, 10].astype(dtype, copy=False)
        return cols

    n_features = 11

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
        logger.info("baseline_disagreement_v2: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
