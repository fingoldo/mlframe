"""Multi-baseline hard-row attention: anchors hardest for ALL 3 baselines (ensemble disagreement).

Iter 68 mechanism. Twist on iter 66 class-balanced hard rows: select anchors by COMBINED |residual|
across 3 different baselines (LGB depth=3, LGB depth=5, Ridge), not just one LGB.

Mechanism:
1. Fit 3 baselines on (X_train, y_train):
   - LGB depth=3, 50 estimators (shallow boosting)
   - LGB depth=5, 50 estimators (deeper boosting)
   - Ridge regression (regression) or logistic regression (binary) (linear model)
2. Compute |residual_b| from each baseline b ∈ {0,1,2}.
3. Z-score normalize each |residual_b| to mean=0 std=1 across rows.
4. Combined hardness score = max across baselines of normalized residual.
5. Within each class (binary) or each top/bottom y-quintile (regression), pick K/2=8 hardest by combined score.

Why this is structurally novel:
- Iter 66: hardest by a single LGB |residual| → can be model-specific (depth=3 finds certain rows hard).
- Iter 68: hardest across ENSEMBLE of model classes (boosting + linear) → "truly hard" rows that NO simple
  model class fits well. Rows that LGB depth=5 fits but Ridge doesn't (or vice versa) are filtered OUT
  because they're "hard for only one class".

The rows that survive max(z-residual across all 3) are rows the downstream boosting will likely struggle with
regardless of which boosting it uses. Captures genuine difficulty, not model-specific artifacts.

Cost: 3× baseline fits (~150ms instead of ~50ms) + same softmax. Sub-second per fold.

Leakage discipline: identical to iter 60-67.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis, numerically stabilized by subtracting the per-row max before exponentiating."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _fit_3baselines_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int) -> list[np.ndarray]:
    """Fit 3 baselines, return list of 3 prediction arrays."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("multi_baseline_hard_row requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    preds_list: list[np.ndarray] = []
    if task == "binary":
        # Baseline 1: LGB depth=3
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        preds_list.append(np.asarray(m1.predict_proba(Xt))[:, 1].astype(np.float32))
        # Baseline 2: LGB depth=5
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        preds_list.append(np.asarray(m2.predict_proba(Xt))[:, 1].astype(np.float32))
        # Baseline 3: LogReg (linear model class).
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            preds_list.append(m3.predict_proba(Xt)[:, 1].astype(np.float32))
        except Exception as exc:
            # Fallback: constant baseline = class prior.
            logger.info("multi_baseline_hard_row: LogisticRegression fit failed (%s); falling back to constant class prior.", exc)
            preds_list.append(np.full(Xt.shape[0], float(y_t.mean()), dtype=np.float32))
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        preds_list.append(np.asarray(m1.predict(Xt)).astype(np.float32))
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        preds_list.append(np.asarray(m2.predict(Xt)).astype(np.float32))
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        preds_list.append(m3.predict(Xt).astype(np.float32))
    return preds_list


def _topk_within_subset(values: np.ndarray, subset_idx: np.ndarray, k: int) -> np.ndarray:
    """Return the (row-)indices in ``subset_idx`` with the top-k largest ``values``, deterministic on ties via index tiebreak."""
    if subset_idx.size == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, subset_idx.size)
    sub_values = values[subset_idx]
    # Wave 62 (2026-05-20): lexsort with within-subset index tiebreak so tied
    # residuals (duplicate rows) give deterministic top-K within subset.
    sub_top = np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]
    return np.asarray(subset_idx[sub_top])


def compute_multi_baseline_hard_row_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_hard_per_side: int = 8,
    temp: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "mbhrattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-baseline hard-row attention.

    Output: 2*n_hard_per_side weights + entropy + per-side agg_y + per-side agg_combined_hardness
    + best_pos + best_neg + min_dist_pos + min_dist_neg = 2*n_hard_per_side + 9 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_total_anchors = 2 * n_hard_per_side

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the 3 baselines, compute the combined (max-of-z-scored |residual|) hardness, pick the K/2 hardest anchors per class/quintile side, then attend each query row to those anchors via softmax(-squared-distance)."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        preds_list = _fit_3baselines_predict(Xt_s, y_t, task=task, seed=fold_seed)

        # Compute z-normalized |residual| per baseline, then max across baselines.
        combined = np.zeros(Xt_s.shape[0], dtype=np.float32)
        for preds in preds_list:
            abs_r = np.abs(y_t - preds).astype(np.float32)
            mu = float(abs_r.mean())
            sd = float(abs_r.std()) + 1e-9
            z = (abs_r - mu) / sd
            combined = np.maximum(combined, z)

        if task == "binary":
            pos_mask_idx = np.where(y_t > 0.5)[0]
            neg_mask_idx = np.where(y_t <= 0.5)[0]
        else:
            q1, q4 = np.quantile(y_t, [0.2, 0.8])
            pos_mask_idx = np.where(y_t >= q4)[0]
            neg_mask_idx = np.where(y_t <= q1)[0]

        pos_top = _topk_within_subset(combined, pos_mask_idx, n_hard_per_side)
        neg_top = _topk_within_subset(combined, neg_mask_idx, n_hard_per_side)
        # Empty-side sentinel anchors: previously np.zeros(...) fell back to REPEATING training row
        # index 0, n_hard_per_side times, as that side's "hardest anchors" -- an arbitrary row that
        # could belong to the opposite class, contaminating every val row's attention weights against
        # a phantom same-side anchor set. Instead, place the empty side's anchors far outside the
        # standardised data manifold (softmax-negligible distance weight against any real query row)
        # with a neutral y/combined value, so this side's aggregate columns degrade toward "no
        # signal" instead of silently reflecting a wrong-class row's real target.
        _FAR = 1e4
        d = Xt_s.shape[1]
        if pos_top.size > 0 and pos_top.size < n_hard_per_side:
            pos_top = np.concatenate([pos_top, np.full(n_hard_per_side - pos_top.size, pos_top[-1])])
        if neg_top.size > 0 and neg_top.size < n_hard_per_side:
            neg_top = np.concatenate([neg_top, np.full(n_hard_per_side - neg_top.size, neg_top[-1])])

        anchors_X_list = []
        anchors_y_list = []
        anchors_combined_list = []
        for top in (pos_top, neg_top):
            if top.size > 0:
                anchors_X_list.append(Xt_s[top])
                anchors_y_list.append(y_t[top].astype(np.float32))
                anchors_combined_list.append(combined[top].astype(np.float32))
            else:
                anchors_X_list.append(np.full((n_hard_per_side, d), _FAR, dtype=np.float32))
                anchors_y_list.append(np.zeros(n_hard_per_side, dtype=np.float32))
                anchors_combined_list.append(np.zeros(n_hard_per_side, dtype=np.float32))
        anchors_X = np.concatenate(anchors_X_list, axis=0)
        anchors_y = np.concatenate(anchors_y_list, axis=0)
        anchors_combined = np.concatenate(anchors_combined_list, axis=0)

        diffs = Xq_s[:, None, :] - anchors_X[None, :, :]
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)

        pos_w = weights[:, :n_hard_per_side]
        neg_w = weights[:, n_hard_per_side:]
        pos_y = anchors_y[:n_hard_per_side]
        neg_y = anchors_y[n_hard_per_side:]
        pos_h = anchors_combined[:n_hard_per_side]
        neg_h = anchors_combined[n_hard_per_side:]

        agg_y_pos = (pos_w * pos_y[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_neg = (neg_w * neg_y[None, :]).sum(axis=-1).astype(np.float32)
        agg_h_pos = (pos_w * pos_h[None, :]).sum(axis=-1).astype(np.float32)
        agg_h_neg = (neg_w * neg_h[None, :]).sum(axis=-1).astype(np.float32)
        best_pos = pos_w.argmax(axis=-1).astype(np.float32)
        best_neg = neg_w.argmax(axis=-1).astype(np.float32)
        min_dist_pos = sq[:, :n_hard_per_side].min(axis=-1).astype(np.float32)
        min_dist_neg = sq[:, n_hard_per_side:].min(axis=-1).astype(np.float32)

        return np.column_stack([
            weights, entropy,
            agg_y_pos, agg_y_neg, agg_h_pos, agg_h_neg,
            best_pos, best_neg, min_dist_pos, min_dist_neg,
        ])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Split the flat ``_process`` output (per-anchor weights + aggregate stats) into named columns."""
        cols: dict[str, np.ndarray] = {}
        for a in range(n_hard_per_side):
            tag = "pos" if task == "binary" else "topy"
            cols[f"{column_prefix}_w_{tag}_h{a}"] = feats[:, a].astype(dtype, copy=False)
        for a in range(n_hard_per_side):
            tag = "neg" if task == "binary" else "boty"
            cols[f"{column_prefix}_w_{tag}_h{a}"] = feats[:, n_hard_per_side + a].astype(dtype, copy=False)
        col_idx = n_total_anchors
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_side1"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_side2"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_hardness_side1"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_hardness_side2"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
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
        logger.info("multi_baseline_hard_row: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
