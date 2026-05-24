"""Gradient-boosted row-attention — each layer learns the previous layer's residual.

Architecture (n_boost_layers=3 example):
1. Layer 0: row-attention with raw y as target → produces y_mean_0 features per row.
2. residual_0 = y - y_mean_0 (per training row, computed from OOF y_mean_0 — leak-free).
3. Layer 1: row-attention with residual_0 as new target → produces y_mean_1.
4. residual_1 = residual_0 - y_mean_1.
5. Layer 2: row-attention with residual_1 as target → y_mean_2.
6. Final feature matrix: concat [y_mean_0, y_mean_1, y_mean_2] across heads.

This is literally gradient boosting in attention space. Each layer captures finer structure that the previous layer missed. Mathematically equivalent to a small ensemble of stacked attention layers with target-residual learning.

Why it should be stronger than:
- **iter 2 stacked** (feeds previous-layer OUTPUT as input X): that layer 2 sees layer-1-output-similarity-space, capturing "rows with similar neighbourhood patterns". Useful but limited to one notion of similarity.
- **iter 3 residual** (auxiliary LGB residual): only one boost layer, no depth.
- **boosted attention** combines both: multiple layers AND each one targets residuals. Should accumulate finer signal.

Leakage discipline:
- Layer 0: OOF y_mean_0 for train via splitter; row r's y_mean_0 excludes r.
- residual_0[r] = y[r] - y_mean_0_oof[r] is itself leak-free (the OOF y_mean is not derived from r's y).
- Layer 1's row-attention uses the *same* splitter; row r's y_mean_1 again excludes r from its key bank, so residual_1 stays leak-free.
- For Mode B (X_query!=None): each layer fits its bank on full X_train + current target, then applies to X_query — standard inference path.

Caveat / risk: residuals get smaller layer by layer. Past layer 2-3 the signal-to-noise of attention features drops sharply (residual variance shrinks). Default n_boost_layers=3 is a reasonable cap; users can raise if they want.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input
from .row_attention import compute_row_attention

logger = logging.getLogger(__name__)


def compute_boosted_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    n_boost_layers: int = 3,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    softmax_temp: float = 1.0,
    standardize: bool = True,
    projection: Literal["random", "pls"] = "pls",
    return_all_layers: bool = True,
    column_prefix: str = "boost",
    dtype: np.dtype = np.float32,
    learning_rate: float = 1.0,
) -> pl.DataFrame:
    """Multi-layer attention with residual target at each layer.

    Layer i target = residual after subtracting all previous layers' OOF y_mean output.
    ``learning_rate`` (default 1.0) scales each layer's contribution before subtraction — analogous to GBDT learning rate. Set lower (e.g. 0.5) for smoother
    convergence; default 1.0 makes each layer fully consume its residual.

    Returns: polars DataFrame, shape ``(N, n_boost_layers * n_heads)`` if return_all_layers else ``(N, n_heads)``. Mode A returns OOF features for X_train;
    Mode B returns features for X_query (built with full-train banks).

    For boostings: each layer's y_mean_i feature exposes a different "scale" of neighbourhood structure. Layer 0 captures the global mean-y pattern; layer 1
    captures the medium-scale residual structure; layer 2 captures fine-grained miss patterns. Downstream boosting can split on any of them.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if n_boost_layers < 1:
        raise ValueError(f"n_boost_layers must be >= 1; got {n_boost_layers}.")

    layer_outputs_train: list[np.ndarray] = []
    layer_outputs_query: list[np.ndarray] = []

    # Current "target" for next layer. Starts as raw y; gets reduced by each layer's OOF prediction.
    current_target_train = y_train.astype(np.float32, copy=True)

    for layer in range(n_boost_layers):
        layer_seed = seed + 1000 * layer
        # Mode A: OOF features for X_train with current_target_train as the "y". Forward caller dtype to compute_row_attention so the inner ndarray is produced
        # at the requested dtype directly; the trailing astype is then a no-op identity.
        out_train_df = compute_row_attention(
            X_train=X_train, y_train=current_target_train, X_query=None, splitter=splitter,
            seed=layer_seed, n_heads=n_heads, head_dim=head_dim, k=k,
            softmax_temp=softmax_temp, aggregate=("y_mean",), standardize=standardize,
            projection=projection, gpu_stage4=False, dedupe_threshold=None,
            dtype=dtype,
        )
        out_train = out_train_df.to_numpy().astype(dtype, copy=False)
        layer_outputs_train.append(out_train)

        if X_query is not None:
            # Mode B for X_query.
            out_query_df = compute_row_attention(
                X_train=X_train, y_train=current_target_train, X_query=X_query, splitter=splitter,
                seed=layer_seed, n_heads=n_heads, head_dim=head_dim, k=k,
                softmax_temp=softmax_temp, aggregate=("y_mean",), standardize=standardize,
                projection=projection, gpu_stage4=False, dedupe_threshold=None,
                dtype=dtype,
            )
            out_query = out_query_df.to_numpy().astype(dtype, copy=False)
            layer_outputs_query.append(out_query)

        # Update current target: average across heads (per row, single scalar prediction), subtract from current target.
        # The OOF y_mean averaged over n_heads is a single prediction per row; subtracting it from current target gives the new "what's left to explain" signal.
        pred_train = out_train.mean(axis=1)  # shape (N,)
        current_target_train = (current_target_train - learning_rate * pred_train).astype(np.float32, copy=False)
        residual_stats = (float(current_target_train.std()), float(np.abs(current_target_train).mean()))
        logger.info("boosted_attention: layer %d/%d done, residual std=%.4f abs_mean=%.4f", layer + 1, n_boost_layers, *residual_stats)

    if return_all_layers:
        train_concat = np.concatenate(layer_outputs_train, axis=1)
        if X_query is not None:
            query_concat = np.concatenate(layer_outputs_query, axis=1)
    else:
        train_concat = layer_outputs_train[-1]
        if X_query is not None:
            query_concat = layer_outputs_query[-1]

    final = query_concat if X_query is not None else train_concat
    n_cols = final.shape[1]
    if return_all_layers:
        names = [f"{column_prefix}_L{layer}_h{h}" for layer in range(n_boost_layers) for h in range(n_heads)]
    else:
        names = [f"{column_prefix}_h{h}" for h in range(n_cols)]
    return pl.DataFrame({name: final[:, i] for i, name in enumerate(names)})
