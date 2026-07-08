"""Multi-temperature class-balanced hard-row attention.

Iter 67 mechanism. Extends iter 66 (8 hardest positives + 8 hardest negatives / K/2 top-y + K/2 bottom-y)
with multi-resolution temperature sweep (sharp 0.3 / medium 1.0 / soft 3.0).

Mechanism:
1. Same anchor selection as iter 66: K/2 hardest per class (binary) or K/2 hardest per top/bottom y-quintile (regression).
2. For each temperature t ∈ temps: compute softmax(−||q − μ_a||² / t) over 2 × K/2 = K anchors.
3. Per temperature: K weights + entropy + per-side agg_y + per-side agg_residual.
4. Output: 3 × (K + 5) = 63 features for default K=16.

Goal: consolidate iter 66's mammography LGB AUC record (+14.46%) and push CB AUC (+3.54%, under iter-53 +4.78%
by 1.24pp) and XGB AUC (+6.73%, under iter-30 +9.77% by 3.04pp) toward their records via multi-resolution
membership signal.

Leakage discipline: identical to iter 60-66.

Cost: same baseline LGB + 3× softmax+aggregate. Sub-second.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_DEFAULT_TEMPS: tuple[float, ...] = (0.3, 1.0, 3.0)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("multi_temp_cbhr requires lightgbm") from exc
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


def _topk_within_subset(values: np.ndarray, subset_idx: np.ndarray, k: int) -> np.ndarray:
    if subset_idx.size == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, subset_idx.size)
    sub_values = values[subset_idx]
    # Wave 62 (2026-05-20): lexsort tiebreak for deterministic top-K within subset.
    sub_top = np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]
    return subset_idx[sub_top]


def compute_multi_temp_cbhr_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_hard_per_side: int = 8,
    temps: Sequence[float] = _DEFAULT_TEMPS,
    baseline_n_estimators: int = 50,
    baseline_max_depth: int = 3,
    standardize: bool = True,
    column_prefix: str = "mtcbhrattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-temperature class-balanced hard-row attention.

    Output per temperature: K weights + entropy + agg_y_side1 + agg_y_side2 + agg_resid_side1 + agg_resid_side2
    = K + 5 features. Total = len(temps) × (K + 5).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_total_anchors = 2 * n_hard_per_side
    temps_list = tuple(float(t) for t in temps)
    n_temps = len(temps_list)
    features_per_temp = n_total_anchors + 5

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

        if task == "binary":
            pos_mask_idx = np.where(y_t > 0.5)[0]
            neg_mask_idx = np.where(y_t <= 0.5)[0]
        else:
            q1, q4 = np.quantile(y_t, [0.2, 0.8])
            pos_mask_idx = np.where(y_t >= q4)[0]
            neg_mask_idx = np.where(y_t <= q1)[0]

        pos_top = _topk_within_subset(abs_residuals, pos_mask_idx, n_hard_per_side)
        neg_top = _topk_within_subset(abs_residuals, neg_mask_idx, n_hard_per_side)
        if pos_top.size < n_hard_per_side:
            if pos_top.size > 0:
                pad = np.full(n_hard_per_side - pos_top.size, pos_top[-1])
                pos_top = np.concatenate([pos_top, pad])
            else:
                pos_top = np.zeros(n_hard_per_side, dtype=np.int64)
        if neg_top.size < n_hard_per_side:
            if neg_top.size > 0:
                pad = np.full(n_hard_per_side - neg_top.size, neg_top[-1])
                neg_top = np.concatenate([neg_top, pad])
            else:
                neg_top = np.zeros(n_hard_per_side, dtype=np.int64)

        anchors_idx = np.concatenate([pos_top, neg_top])
        anchors_X = Xt_s[anchors_idx]
        anchors_y = y_t[anchors_idx].astype(np.float32)
        anchors_abs = abs_residuals[anchors_idx].astype(np.float32)

        diffs = Xq_s[:, None, :] - anchors_X[None, :, :]
        sq = (diffs**2).sum(axis=-1)
        scores = -sq

        n_q = Xq_s.shape[0]
        out_blocks = np.zeros((n_q, n_temps * features_per_temp), dtype=np.float32)
        for ti, t in enumerate(temps_list):
            weights = _softmax(scores, temp=t)
            entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)

            pos_w = weights[:, :n_hard_per_side]
            neg_w = weights[:, n_hard_per_side:]
            pos_y = anchors_y[:n_hard_per_side]
            neg_y = anchors_y[n_hard_per_side:]
            pos_abs = anchors_abs[:n_hard_per_side]
            neg_abs = anchors_abs[n_hard_per_side:]

            agg_y_side1 = (pos_w * pos_y[None, :]).sum(axis=-1).astype(np.float32)
            agg_y_side2 = (neg_w * neg_y[None, :]).sum(axis=-1).astype(np.float32)
            agg_resid_side1 = (pos_w * pos_abs[None, :]).sum(axis=-1).astype(np.float32)
            agg_resid_side2 = (neg_w * neg_abs[None, :]).sum(axis=-1).astype(np.float32)

            base = ti * features_per_temp
            out_blocks[:, base : base + n_total_anchors] = weights
            out_blocks[:, base + n_total_anchors] = entropy
            out_blocks[:, base + n_total_anchors + 1] = agg_y_side1
            out_blocks[:, base + n_total_anchors + 2] = agg_y_side2
            out_blocks[:, base + n_total_anchors + 3] = agg_resid_side1
            out_blocks[:, base + n_total_anchors + 4] = agg_resid_side2

        return out_blocks

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for ti, t in enumerate(temps_list):
            base = ti * features_per_temp
            t_tag = f"t{t:g}".replace(".", "p")
            for a in range(n_hard_per_side):
                tag = "pos" if task == "binary" else "topy"
                cols[f"{column_prefix}_{t_tag}_w_{tag}_h{a}"] = feats[:, base + a].astype(dtype, copy=False)
            for a in range(n_hard_per_side):
                tag = "neg" if task == "binary" else "boty"
                cols[f"{column_prefix}_{t_tag}_w_{tag}_h{a}"] = feats[:, base + n_hard_per_side + a].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_entropy"] = feats[:, base + n_total_anchors].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_y_side1"] = feats[:, base + n_total_anchors + 1].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_y_side2"] = feats[:, base + n_total_anchors + 2].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_resid_side1"] = feats[:, base + n_total_anchors + 3].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_resid_side2"] = feats[:, base + n_total_anchors + 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    total_features = n_temps * features_per_temp
    out: np.ndarray = np.zeros((n_train, total_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("multi_temp_cbhr: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
