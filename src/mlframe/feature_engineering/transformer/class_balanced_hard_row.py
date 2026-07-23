"""Class-balanced hard-row attention: K/2 hardest positives + K/2 hardest negatives (binary)
or K/2 hardest top-quintile-y + K/2 hardest bottom-quintile-y (regression).

Iter 66 mechanism. Addresses iter 65's blindness to class imbalance and heavy-tail collapse.

Mechanism (binary):
1. Fit 50-iter LightGBM baseline → in-sample predictions p̂.
2. |residual| = |y - p̂|.
3. Within y=1 rows, pick K/2=8 with largest |residual| (hardest positives, FN-prone).
4. Within y=0 rows, pick K/2=8 with largest |residual| (hardest negatives, FP-prone).
5. Combined K=16 anchors with class balance (8 pos + 8 neg).

Mechanism (regression):
1. Fit 50-iter LightGBM baseline → in-sample predictions ŷ.
2. |residual| = |y - ŷ|.
3. Partition y into 5 quintiles.
4. Within Q5 (top-y) rows, pick K/2=8 hardest by |residual|.
5. Within Q1 (bottom-y) rows, pick K/2=8 hardest by |residual|.
6. Combined K=16 anchors covering both target extremes; ignores Q2-Q4 where boosting fits typically well.

Per query: softmax over K=16 anchor positions. Output: 16 weights + entropy + per-side agg_y +
per-side agg_residual + 2 best_anchor_indices + min_dist_per_side = ~24 features.

Why this is structurally novel vs iter 65:
- Iter 65: top-K by |residual| globally → for 1.3% positive mammography, K=16 may have all FN-prone positives
  or all FP-prone negatives, depending on boosting's class bias.
- Iter 66: forces 8 + 8 split → guaranteed class coverage. On regression: forces top+bottom y-extremes coverage.

For regression, the y-quintile constraint avoids iter 62's catastrophe (heavy-tailed signed residuals)
because anchors come from BOTH Q1 and Q5 in balanced count — no single tail can dominate the anchor set.

Leakage discipline: identical to iter 60-65.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Temperature-scaled, max-shifted softmax along the last axis; the shift keeps ``exp`` numerically stable for large negative-squared-distance scores."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    """Fit a shallow LightGBM baseline (classifier or regressor per ``task``) and return its IN-SAMPLE predictions, used only to rank rows by |residual| hardness."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("class_balanced_hard_row requires lightgbm") from exc
    if task == "binary":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=int(seed), verbose=-1, n_jobs=-1,
        )
        model.fit(Xt, y_t.astype(np.int32))
        preds = np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=int(seed), verbose=-1, n_jobs=-1,
        )
        model.fit(Xt, y_t)
        preds = np.asarray(model.predict(Xt)).astype(np.float32)
    return np.asarray(preds)


def _topk_within_subset(values: np.ndarray, subset_idx: np.ndarray, k: int) -> np.ndarray:
    """Return indices (into the original array) of the top-k entries WITHIN `subset_idx`, sorted by descending value."""
    if subset_idx.size == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, subset_idx.size)
    sub_values = values[subset_idx]
    # Wave 62 (2026-05-20): lexsort tiebreak for deterministic top-K within subset.
    sub_top = np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]
    return np.asarray(subset_idx[sub_top])


def compute_class_balanced_hard_row_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_hard_per_side: int = 8,
    temp: float = 1.0,
    baseline_n_estimators: int = 50,
    baseline_max_depth: int = 3,
    standardize: bool = True,
    column_prefix: str = "cbhrattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Class-balanced hard-row attention.

    Output: 2*n_hard_per_side weights + entropy + per-side agg_y + per-side agg_resid + best_pos + best_neg + min_dist_pos + min_dist_neg
    = 2*n_hard_per_side + 8 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_total_anchors = 2 * n_hard_per_side

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the baseline, pick the class-balanced (or y-extreme-balanced) hardest anchors, and compute softmax-attention features against the query rows."""
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

        if task == "binary":
            pos_mask_idx = np.where(y_t > 0.5)[0]
            neg_mask_idx = np.where(y_t <= 0.5)[0]
        else:
            # Regression: Q5 (top-y) and Q1 (bottom-y).
            q1, q4 = np.quantile(y_t, [0.2, 0.8])
            pos_mask_idx = np.where(y_t >= q4)[0]  # top-y subset
            neg_mask_idx = np.where(y_t <= q1)[0]  # bottom-y subset

        # Pick K/2 hardest within each side.
        pos_top = _topk_within_subset(abs_residuals, pos_mask_idx, n_hard_per_side)
        neg_top = _topk_within_subset(abs_residuals, neg_mask_idx, n_hard_per_side)
        # Empty-side sentinel anchors: previously an all-empty side fell back to REPEATING training
        # row index 0, n_hard_per_side times, as that side's "hardest anchors" -- an arbitrary row
        # that could belong to the opposite class. Instead, place the empty side's anchors far
        # outside the standardised data manifold (softmax-negligible distance weight against any
        # real query row) with a neutral y/residual value, matching multi_baseline_hard_row.py's
        # identical fix.
        _FAR = 1e4
        d = Xt_s.shape[1]
        if pos_top.size > 0 and pos_top.size < n_hard_per_side:
            pos_top = np.concatenate([pos_top, np.full(n_hard_per_side - pos_top.size, pos_top[-1])])
        if neg_top.size > 0 and neg_top.size < n_hard_per_side:
            neg_top = np.concatenate([neg_top, np.full(n_hard_per_side - neg_top.size, neg_top[-1])])

        anchors_X_list = []
        anchors_y_list = []
        anchors_abs_list = []
        for top in (pos_top, neg_top):
            if top.size > 0:
                anchors_X_list.append(Xt_s[top])
                anchors_y_list.append(y_t[top].astype(np.float32))
                anchors_abs_list.append(abs_residuals[top].astype(np.float32))
            else:
                anchors_X_list.append(np.full((n_hard_per_side, d), _FAR, dtype=np.float32))
                anchors_y_list.append(np.zeros(n_hard_per_side, dtype=np.float32))
                anchors_abs_list.append(np.zeros(n_hard_per_side, dtype=np.float32))
        anchors_X = np.concatenate(anchors_X_list, axis=0)  # (n_total, d)
        anchors_y = np.concatenate(anchors_y_list, axis=0)
        anchors_abs = np.concatenate(anchors_abs_list, axis=0)

        diffs = Xq_s[:, None, :] - anchors_X[None, :, :]  # (n_q, n_total, d)
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)  # (n_q, n_total)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)

        pos_w = weights[:, :n_hard_per_side]
        neg_w = weights[:, n_hard_per_side:]
        pos_y = anchors_y[:n_hard_per_side]
        neg_y = anchors_y[n_hard_per_side:]
        pos_abs = anchors_abs[:n_hard_per_side]
        neg_abs = anchors_abs[n_hard_per_side:]

        agg_y_pos = (pos_w * pos_y[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_neg = (neg_w * neg_y[None, :]).sum(axis=-1).astype(np.float32)
        agg_resid_pos = (pos_w * pos_abs[None, :]).sum(axis=-1).astype(np.float32)
        agg_resid_neg = (neg_w * neg_abs[None, :]).sum(axis=-1).astype(np.float32)

        best_pos = pos_w.argmax(axis=-1).astype(np.float32)
        best_neg = neg_w.argmax(axis=-1).astype(np.float32)
        min_dist_pos = sq[:, :n_hard_per_side].min(axis=-1).astype(np.float32)
        min_dist_neg = sq[:, n_hard_per_side:].min(axis=-1).astype(np.float32)

        return np.column_stack([
            weights,
            entropy,
            agg_y_pos, agg_y_neg,
            agg_resid_pos, agg_resid_neg,
            best_pos, best_neg,
            min_dist_pos, min_dist_neg,
        ])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Name the flat ``feats`` columns; anchor-weight columns are tagged pos/neg for binary or topy/boty for regression, matching the two anchor sides `_process` built."""
        cols: dict[str, np.ndarray] = {}
        # pos weights
        for a in range(n_hard_per_side):
            tag = "pos" if task == "binary" else "topy"
            cols[f"{column_prefix}_w_{tag}_h{a}"] = feats[:, a].astype(dtype, copy=False)
        for a in range(n_hard_per_side):
            tag = "neg" if task == "binary" else "boty"
            cols[f"{column_prefix}_w_{tag}_h{a}"] = feats[:, n_hard_per_side + a].astype(dtype, copy=False)
        col_idx = n_total_anchors
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_agg_side1"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_agg_side2"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_abs_resid_side1"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_abs_resid_side2"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_best_side1"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_best_side2"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_mindist_side1"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_mindist_side2"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        return cols

    n_features = n_total_anchors + 9

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
        logger.info("class_balanced_hard_row: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
