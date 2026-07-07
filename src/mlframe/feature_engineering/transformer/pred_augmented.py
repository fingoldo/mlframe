"""Prediction-augmented row-attention: similarity computed in (X || y_hat) space.

Mechanism:
1. Fit a small LightGBM auxiliary model on (X_train, y_train) via KFold OOF → ``y_hat_train_oof``.
2. For X_query, fit a single full-train LGB → ``y_hat_query``.
3. Augment X with the auxiliary prediction as an extra feature column: ``X_aug = concat(X, y_hat[:, None])``.
4. Run row-attention with ``y_train`` as target on the AUGMENTED space.

Why this might break the +5% bar on the lagging cells:
- Plain row-attention's similarity is X-only. Two rows can be close in X but have very different y if the relation is highly nonlinear locally.
- Pred-augmented similarity respects the auxiliary model's view: rows similar in both X AND in predicted y are considered close, so neighbour aggregates pool more
  semantically-consistent neighbours.
- The aux LGB already captures the obvious axis-aligned splits; row-attention then layers cross-row aggregation in a y-aware metric.

Different from `compute_residual_attention` (iter 3): residual attention runs row-attention with RESIDUALS as the target (kNN aggregation of error patterns).
Pred-augmented runs row-attention with raw y as target but in an augmented (X || y_hat) feature space (kNN aggregation in a y-aware metric).

Reference: stacking with kNN as the meta-model; aggregate-feature engineering.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._oof import apply_dedupe
from ._residual_oof import compute_oof_yhat_within
from ._utils import require_seed, validate_numeric_input
from .row_attention import compute_row_attention

logger = logging.getLogger(__name__)


def compute_pred_augmented_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    task: str = "auto",
    aux_n_estimators: int = 100,
    aux_max_depth: int = 5,
    aux_n_splits: int = 5,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    projection: Literal["random", "pls", "importance"] = "pls",
    column_prefix: str = "predaug",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Row-attention with similarity in (X || y_hat) augmented space.

    For each X_train row, the auxiliary y_hat is the OOF prediction from a small LGB; for each X_query row, the full-train LGB's prediction. This is added as
    a single extra column to X before standardisation, so the per-row representation in attention space combines raw X features with the aux-LGB belief about y.

    Returns: polars DataFrame matching ``compute_row_attention`` output shape (N, n_heads * len(aggregate) + optional extras).
    """
    import lightgbm as lgb
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)

    if task == "auto":
        unique_y = np.unique(y_train[~np.isnan(y_train)] if y_train.dtype.kind == "f" else y_train)
        task = "binary" if len(unique_y) == 2 else "regression"

    def _make_aux():
        common = dict(
            n_estimators=aux_n_estimators, max_depth=aux_max_depth, learning_rate=0.05,
            random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** aux_max_depth, 63),
        )
        return lgb.LGBMClassifier(**common) if task == "binary" else lgb.LGBMRegressor(**common)

    def _augment(X_base: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return np.concatenate([X_base.astype(dtype, copy=False), y_hat.reshape(-1, 1).astype(dtype, copy=False)], axis=1)

    def _yhat_full(X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
        model = _make_aux()
        model.fit(X_fit, y_fit)
        if task == "binary":
            return np.asarray(model.predict_proba(X_pred)[:, 1], dtype=np.float32)
        return np.asarray(model.predict(X_pred), dtype=np.float32)

    if X_query is not None:
        # Mode B: aux y_hat OOF for the train bank; query rows get y_hat from a full-train aux fit (leakage-safe — query rows are disjoint from train).
        y_hat_train = compute_oof_yhat_within(X_train, y_train, task=task, make_aux=_make_aux, aux_n_splits=aux_n_splits, seed=seed)
        X_train_aug = _augment(X_train, y_hat_train)
        X_query_aug = _augment(X_query, _yhat_full(X_train, y_train, X_query))
        logger.info("pred_augmented: y_hat OOF mean=%.4f std=%.4f, augmented X shape: train=%s, query=%s",
                    y_hat_train.mean(), y_hat_train.std(), X_train_aug.shape, X_query_aug.shape)
        return compute_row_attention(
            X_train=X_train_aug, y_train=y_train, X_query=X_query_aug, splitter=splitter,
            seed=seed, n_heads=n_heads, head_dim=min(head_dim, X_train_aug.shape[1] - 1), k=k,
            aggregate=aggregate, projection=projection, gpu_stage4=False, dedupe_threshold=None,
            column_prefix=column_prefix,
        )

    # Mode A: nest the aux OOF inside the caller's outer splitter so a val row's own target never leaks into the augmented similarity coordinate. For each outer fold
    # the bank's y_hat is an aux OOF restricted to the fold's train complement; the val rows' y_hat comes from a full-complement aux fit (val rows are out-of-complement).
    n_train = X_train.shape[0]
    fold_frames: list[pl.DataFrame] = []
    fold_val_idx: list[np.ndarray] = []
    for train_idx, val_idx in splitter.split(X_train):
        X_comp, y_comp = X_train[train_idx], y_train[train_idx]
        y_hat_comp = compute_oof_yhat_within(X_comp, y_comp, task=task, make_aux=_make_aux, aux_n_splits=aux_n_splits, seed=seed)
        y_hat_val = _yhat_full(X_comp, y_comp, X_train[val_idx])
        X_comp_aug = _augment(X_comp, y_hat_comp)
        X_val_aug = _augment(X_train[val_idx], y_hat_val)
        fold_out = compute_row_attention(
            X_train=X_comp_aug, y_train=y_comp, X_query=X_val_aug, splitter=splitter,
            seed=seed, n_heads=n_heads, head_dim=min(head_dim, X_comp_aug.shape[1] - 1), k=k,
            aggregate=aggregate, projection=projection, gpu_stage4=False, dedupe_threshold=None,
            column_prefix=column_prefix,
        )
        fold_frames.append(fold_out)
        fold_val_idx.append(np.asarray(val_idx))

    names = fold_frames[0].columns
    matrix: np.ndarray = np.zeros((n_train, len(names)), dtype=dtype)
    for frame, val_idx in zip(fold_frames, fold_val_idx):
        matrix[val_idx] = frame.select(names).to_numpy().astype(dtype, copy=False)
    matrix, names = apply_dedupe(matrix, list(names), dedupe_threshold=None)
    return pl.DataFrame({n: matrix[:, i] for i, n in enumerate(names)})
