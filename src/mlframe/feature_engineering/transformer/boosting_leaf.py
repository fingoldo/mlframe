"""Boosting-leaf encoding: GBDT+LR pattern adapted as a transformer-style FE block.

The classic Facebook GBDT+LR trick (He et al. 2014, "Practical Lessons from Predicting Clicks on Ads at Facebook") fits a small gradient-boosted ensemble, takes
each row's leaf-index per tree as a categorical feature, then one-hots them as inputs to a linear model. The leaves are *learned discriminative partitions* — each
tree's decision path encodes a piecewise-constant slice of the input space that's predictive for the target.

This module adapts the same idea as auxiliary features for downstream boostings (CB / XGB / LGB):

1. Fit a small LightGBM (defaults: 50 trees, depth=4) on `(X_train, y_train)`.
2. Extract leaf indices for train (via OOF KFold to avoid leak) and for test (single-pass).
3. Optional one-hot encoding; otherwise pass leaf indices as ordinal features (LGB/CB handle these natively).
4. Return as polars DataFrame for downstream concatenation.

Why this is "transformer-like": each tree's leaf assignment is effectively a soft cluster-membership signal — analogous to a learned discrete-attention head over
the input space. Multiple trees == multi-head attention with learned partitions.

Why this helps boostings: the auxiliary boosting's leaves capture global interactions that any single downstream tree may need many splits to approximate. The
downstream model gets pre-computed "this row belongs to subspace X" features for free. Famously this lifts LR by 3-5% AUC on Facebook's data; for boostings the
lift is smaller but typically still positive when the auxiliary boosting is small and diverse from the downstream one.

Reference: He et al. 2014 (Facebook GBDT+LR); Cheng & Koc 2016 (Wide & Deep).
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_boosting_leaf_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    *,
    seed: int,
    task: str = "auto",
    n_estimators: int = 50,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_splits: int = 5,
    encoding: str = "ordinal",
    column_prefix: str = "leaf",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Boosting-leaf encoding: fits a small LightGBM, extracts per-row leaf indices, returns them as features.

    Mode A (``X_query is None``): produces OOF leaf indices for train rows via KFold (each row's leaf comes from a boosting NOT trained on it).
    Mode B (``X_query is not None``): single-pass — fits one LightGBM on full `(X_train, y_train)`, extracts leaves for `X_query`. The train output is also returned
    on the same logic (Mode A pattern for `X_train`, Mode B for `X_query`); call the function twice if you only need one side.

    Parameters:
        ``task``         - "auto" (infer from y), "regression", or "binary". Auto: binary iff y has exactly 2 unique values.
        ``n_estimators`` - number of trees in the auxiliary boosting. Small (50) is right: we want a diverse-from-downstream signal, not an over-fitted clone.
        ``max_depth``    - depth cap. depth=4 → max 16 leaves per tree → cluster granularity comparable to a typical kNN k=16-32.
        ``encoding``     - "ordinal" (default, return raw leaf indices as float; LGB/CB/XGB all handle these as categorical implicitly) or "onehot" (explode to
                           binary indicator matrix; faster for linear downstream models, ballooning for tree downstream so default is ordinal).
        ``n_splits``     - KFold n_splits for OOF leaf extraction on train. 5 is standard.

    Output: polars DataFrame ``(N, n_estimators)`` for ordinal encoding (one column per tree, named ``{column_prefix}_t{tree_id}``) or
    ``(N, sum_per_tree_leaves)`` for onehot. Row order matches the relevant input.

    Returns features for the X_query input if provided, else for X_train (OOF).
    """
    import lightgbm as lgb
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
        if X_query.shape[1] != X_train.shape[1]:
            raise ValueError(f"X_query has d={X_query.shape[1]} but X_train has d={X_train.shape[1]}.")
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError(f"y_train len {y_train.shape[0]} != X_train rows {X_train.shape[0]}.")
    if encoding not in ("ordinal", "onehot"):
        raise ValueError(f"encoding must be 'ordinal' or 'onehot'; got {encoding!r}.")

    if task == "auto":
        unique_y = np.unique(y_train[~np.isnan(y_train)] if y_train.dtype.kind == "f" else y_train)
        task = "binary" if len(unique_y) == 2 else "regression"

    def _make_model():
        if task == "binary":
            return lgb.LGBMClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** max_depth, 63),
            )
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** max_depth, 63),
        )

    if X_query is None:
        # Mode A (OOF on train).
        leaf_out = np.empty((X_train.shape[0], n_estimators), dtype=np.int32)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_train_idx, fold_val_idx in splitter.split(X_train):
            model = _make_model()
            model.fit(X_train[fold_train_idx], y_train[fold_train_idx])
            leaf_out[fold_val_idx] = model.predict(X_train[fold_val_idx], pred_leaf=True).astype(np.int32, copy=False)
    else:
        # Mode B (single fit on full train, predict on X_query).
        model = _make_model()
        model.fit(X_train, y_train)
        leaf_out = model.predict(X_query, pred_leaf=True).astype(np.int32, copy=False)

    n_rows = leaf_out.shape[0]
    n_trees = leaf_out.shape[1]

    if encoding == "ordinal":
        col_names = [f"{column_prefix}_t{i}" for i in range(n_trees)]
        return pl.DataFrame({name: leaf_out[:, idx].astype(dtype, copy=False) for idx, name in enumerate(col_names)})

    # one-hot encoding: per tree, create binary indicators. Total feature count is the sum of unique leaves observed across trees (capped at 2^max_depth per tree).
    onehot_cols = []
    onehot_names = []
    for t in range(n_trees):
        unique_leaves = np.unique(leaf_out[:, t])
        # Skip degenerate trees that have only one leaf observed (would produce a constant column).
        if len(unique_leaves) < 2:
            continue
        for leaf in unique_leaves:
            indicator = (leaf_out[:, t] == leaf).astype(dtype, copy=False)
            onehot_cols.append(indicator)
            onehot_names.append(f"{column_prefix}_t{t}_l{leaf}")
    if not onehot_cols:
        # All trees degenerate; return a single zero column rather than an empty frame so downstream concat doesn't break.
        return pl.DataFrame({f"{column_prefix}_degenerate": np.zeros(n_rows, dtype=dtype)})
    return pl.DataFrame({name: col for name, col in zip(onehot_names, onehot_cols)})
