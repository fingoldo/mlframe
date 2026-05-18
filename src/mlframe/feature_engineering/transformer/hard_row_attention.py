"""Hard-row attention: top-K hardest training rows (by |residual|) as individual anchors.

Iter 65 mechanism. Structurally different from band-attention family (iters 57-64): instead of
partitioning rows into bands and using band CENTROIDS as anchors, this picks K=16 individual
hardest training rows by |residual| from a baseline LightGBM, then routes queries via softmax
through those K row positions directly.

Mechanism:
1. Fit 50-iter LightGBM baseline → in-sample predictions ŷ.
2. Compute |residual| = |y - ŷ|; pick top-K rows (largest |residual| = boosting's hardest cases).
3. For each hard row i: anchor position μ_i = X[hard_i], y_i, residual_i.
4. Per query: softmax(−||q − μ_i||² / temp) over K hard-row anchors.
5. Output: K weights + entropy + agg_y (over hard rows' y) + agg_residual + best_hard_idx + min_dist + agg_signed_residual
   = K + 6 features.

Why this is structurally novel:
- Iter 57-64 band attention: bands are aggregates (centroids of 20% of rows each) → coarse.
- Iter 53 inducing-attention: K=16 K-means anchors on ALL rows → target-agnostic.
- Iter 65: K=16 hard-row anchors selected by RESIDUAL → target-aware AND row-level granular.

The signal: "is this query near boosting's hardest cases?" If yes, the boosting's downstream output
needs caution / extra capacity here. If no, the query is in well-fit territory.

Leakage discipline: identical to iter 60-63 — baseline fit on train fold, hard rows selected from train fold
only, query routed through their X positions.

Cost: 50-iter LGB + top-K argpartition + softmax — sub-second.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("hard_row_attention requires lightgbm") from exc
    if task == "binary":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=int(seed), verbose=-1, n_jobs=-1,
        )
        model.fit(Xt, y_t.astype(np.int32))
        preds = model.predict_proba(Xt)[:, 1].astype(np.float32)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=int(seed), verbose=-1, n_jobs=-1,
        )
        model.fit(Xt, y_t)
        preds = model.predict(Xt).astype(np.float32)
    return preds


def compute_hard_row_attention_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_hard: int = 16,
    temp: float = 1.0,
    baseline_n_estimators: int = 50,
    baseline_max_depth: int = 3,
    standardize: bool = True,
    column_prefix: str = "hrattn",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Hard-row attention: top-K |residual| rows as anchors.

    Output: n_hard weights + entropy + agg_y + agg_abs_residual + agg_signed_residual + best_hard_idx + min_dist
    = n_hard + 6 features.
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
        preds_tr = _fit_baseline_predict(
            Xt_s, y_t, task=task, seed=fold_seed,
            n_estimators=baseline_n_estimators, max_depth=baseline_max_depth,
        )
        signed_residuals = (y_t - preds_tr).astype(np.float32)
        abs_residuals = np.abs(signed_residuals)
        # Pick top-K hardest rows by |residual|.
        n_train_rows = Xt_s.shape[0]
        k_eff = min(n_hard, n_train_rows)
        if k_eff < n_hard:
            # Pad with the same row if not enough training rows.
            top_idx = np.argsort(abs_residuals)[::-1][:k_eff]
            pad = np.full(n_hard - k_eff, top_idx[-1])
            top_idx = np.concatenate([top_idx, pad])
        else:
            # Use argpartition for top-K; then sort by |residual| descending for stable feature ordering.
            unsorted_top = np.argpartition(abs_residuals, -n_hard)[-n_hard:]
            order = np.argsort(abs_residuals[unsorted_top])[::-1]
            top_idx = unsorted_top[order]

        anchors_X = Xt_s[top_idx]                  # (n_hard, d)
        anchors_y = y_t[top_idx].astype(np.float32)
        anchors_abs_resid = abs_residuals[top_idx]
        anchors_signed_resid = signed_residuals[top_idx]

        diffs = Xq_s[:, None, :] - anchors_X[None, :, :]  # (n_q, n_hard, d)
        sq = (diffs ** 2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)  # (n_q, n_hard)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y = (weights * anchors_y[None, :]).sum(axis=-1).astype(np.float32)
        agg_abs_resid = (weights * anchors_abs_resid[None, :]).sum(axis=-1).astype(np.float32)
        agg_signed_resid = (weights * anchors_signed_resid[None, :]).sum(axis=-1).astype(np.float32)
        best_hard = weights.argmax(axis=-1).astype(np.float32)
        min_dist = sq.min(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y, agg_abs_resid, agg_signed_resid, best_hard, min_dist])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for a in range(n_hard):
            cols[f"{column_prefix}_w_h{a}"] = feats[:, a].astype(dtype, copy=False)
        col_idx = n_hard
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_agg"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_abs_resid_agg"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_signed_resid_agg"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_best_hard"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_min_dist"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_features = n_hard + 6
    out = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("hard_row_attention: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
