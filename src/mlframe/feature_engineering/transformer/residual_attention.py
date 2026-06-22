"""Self-supervised residual row-attention — captures what an auxiliary boosting misses in each row's neighbourhood.

Pipeline:
1. Fit a small LightGBM on `(X_train, y_train)` via KFold to get OOF predictions ``y_hat_train_oof``.
2. Compute residuals: ``residual_train = y_train - y_hat_train_oof`` (regression) or ``y_train - p_hat_oof`` (binary).
3. Run row-attention with ``residual_train`` as the target.

Output features per row: "weighted mean of neighbour RESIDUALS" — i.e., "in my neighbourhood, what error pattern is the auxiliary boosting making?"

Why this should help boostings downstream:
- Plain row-attention on raw y produces "neighbour mean of y" — boosting can sometimes approximate this with enough splits.
- Residual row-attention produces "neighbour mean of what was MISSED by a previous boosting" — by definition the downstream boosting starts from no information about residual structure (it's a feature engineered FROM a boosting fit, not predictable WITHIN one boosting iteration). This is the canonical "stacking" / "boosting-of-boosting" pattern lifted to a neighbourhood-aggregate.

Leakage discipline:
- Auxiliary LGB is fit via KFold OOF to get unbiased residuals for train rows.
- Mode A (X_query=None): the aux OOF is NESTED inside the caller's outer splitter. For each outer fold f, the residual bank that f's val rows attend to is produced by an aux OOF restricted to f's train complement only, so no row of fold f ever contributes to a complement-row residual. A flat aux KFold independent of the outer splitter would let a val row's own target leak into its attention feature: the complement residuals it attends to could come from an aux model that trained on fold f (including the val row itself).
- Mode B (X_query!=None): auxiliary LGB is fit once on full train → y_hat for X_query (residual unknown for X_query, that's fine — we're using train residuals as the bank's "target"); outer attention uses full train residual_bank.

Reference: stacking (Wolpert 1992); gradient boosting + residual learning (Friedman 2001); FB GBDT+LR but with neighbour aggregate as the "LR" output.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl
from ._oof import apply_dedupe
from ._residual_oof import compute_oof_residual_within
from ._utils import require_seed, validate_numeric_input
from .row_attention import compute_row_attention

logger = logging.getLogger(__name__)


def compute_residual_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    task: str = "auto",
    aux_n_estimators: int = 100,
    aux_max_depth: int = 5,
    aux_learning_rate: float = 0.05,
    aux_n_splits: int = 5,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    projection: Literal["random", "pls"] = "pls",
    column_prefix: str = "resid",
    dedupe_threshold: float | None = None,
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Residual row-attention: fits auxiliary LGB, takes OOF residuals, runs row-attention with residuals as target.

    Returns a polars DataFrame:
        - Mode A (``X_query is None``): OOF residual-attention features for X_train.
        - Mode B (``X_query is not None``): residual-attention features for X_query (using full-train residual bank).

    Output column names: ``{column_prefix}_h{head}_{agg}`` per the row-attention convention. Aggregate names with the "y_" prefix in the underlying call still
    refer to the y-target field, which is actually "residual" in this context. We keep the naming for consistency with the base API; users reading the output
    should mentally substitute "residual" for "y".

    Parameters specific to residual extraction:
        ``task``               - "auto" (infer from y), "regression", or "binary". Auto: binary iff y has exactly 2 unique values.
        ``aux_n_estimators``   - trees in the auxiliary LGB. 100 is enough for a smooth-target fit; more would overfit and shrink residuals to zero.
        ``aux_max_depth``      - depth cap; 5 is a typical small-LGB depth.
        ``aux_learning_rate``  - 0.05 is conservative; matches the downstream defaults to make residuals "what a similar LGB would have missed".
        ``aux_n_splits``       - KFold for auxiliary OOF predictions.

    Remaining kwargs (``n_heads, head_dim, k, aggregate, projection, ...``) forward directly to ``compute_row_attention``.
    """
    import lightgbm as lgb

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError(f"y_train length {y_train.shape[0]} != X_train rows {X_train.shape[0]}.")

    if task == "auto":
        unique_y = np.unique(y_train[~np.isnan(y_train)] if y_train.dtype.kind == "f" else y_train)
        task = "binary" if len(unique_y) == 2 else "regression"

    def _make_aux():
        common = dict(
            n_estimators=aux_n_estimators, max_depth=aux_max_depth, learning_rate=aux_learning_rate,
            random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** aux_max_depth, 63),
        )
        return lgb.LGBMClassifier(**common) if task == "binary" else lgb.LGBMRegressor(**common)

    common_attn = dict(
        seed=seed, n_heads=n_heads, head_dim=head_dim, k=k,
        aggregate=aggregate, projection=projection,
        gpu_stage4=False, dedupe_threshold=None, column_prefix=column_prefix,
    )

    if X_query is not None:
        # Mode B: aux fit once on full train → residual bank for the full train; query rows attend to the full bank.
        residual_train = compute_oof_residual_within(
            X_train, y_train, task=task, make_aux=_make_aux, aux_n_splits=aux_n_splits, seed=seed,
        )
        logger.info("residual_attention: aux LGB OOF residuals: mean=%.4f, std=%.4f, abs_mean=%.4f", residual_train.mean(), residual_train.std(), np.abs(residual_train).mean())
        return compute_row_attention(X_train=X_train, y_train=residual_train, X_query=X_query, splitter=splitter, **common_attn)

    # Mode A: nest the aux OOF inside the caller's outer splitter. For each outer fold f, the residual bank attended by f's val rows is built from an aux OOF
    # restricted to f's train complement only — so no row of fold f contributes to a complement-row residual (the cross-outer-fold target leak the flat aux KFold had).
    n_train = X_train.shape[0]
    fold_frames: list[pl.DataFrame] = []
    fold_val_idx: list[np.ndarray] = []
    abs_means: list[float] = []
    for train_idx, val_idx in splitter.split(X_train):
        residual_complement = compute_oof_residual_within(
            X_train[train_idx], y_train[train_idx], task=task, make_aux=_make_aux, aux_n_splits=aux_n_splits, seed=seed,
        )
        abs_means.append(float(np.abs(residual_complement).mean()))
        # Single-fold attention: complement residuals are the bank, this fold's val rows are the queries (Mode B mechanics, leakage-safe by construction here).
        fold_out = compute_row_attention(
            X_train=X_train[train_idx], y_train=residual_complement, X_query=X_train[val_idx], splitter=splitter, **common_attn,
        )
        fold_frames.append(fold_out)
        fold_val_idx.append(np.asarray(val_idx))
    logger.info("residual_attention: nested aux OOF residuals over %d outer folds: abs_mean=%.4f", len(fold_frames), float(np.mean(abs_means)) if abs_means else 0.0)

    # Scatter each fold's val-row features into the master output at val_idx. Column names are deterministic (dedupe disabled per fold), so all folds share a schema.
    names = fold_frames[0].columns
    matrix = np.zeros((n_train, len(names)), dtype=dtype)
    for frame, val_idx in zip(fold_frames, fold_val_idx):
        matrix[val_idx] = frame.select(names).to_numpy().astype(dtype, copy=False)

    matrix, names = apply_dedupe(matrix, list(names), dedupe_threshold=dedupe_threshold)
    return pl.DataFrame({n: matrix[:, i] for i, n in enumerate(names)})
