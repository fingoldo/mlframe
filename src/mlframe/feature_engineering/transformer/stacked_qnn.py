"""Stacked quantile-neighbours: apply iter-20 quantile-neighbours on top of the iter-20 output to expose higher-order target-distribution structure.

Iter 20 (`compute_quantile_neighbours`) exposes the LOCAL target distribution (5 quantiles per row). Stacked-qnn treats iter-20's output as a richer feature space
and runs kNN+quantile-aggregation AGAIN in (X || qnn-features) space, so the second layer's neighbours are rows whose ENTIRE target distribution shape resembles
the query's — not just rows close in raw X.

Mechanism:
1. Compute iter-20 qnn features Q_1 = (q10, q25, q50, q75, q90) per row via standard OOF / Mode-B discipline.
2. Stack features: X_stacked = (X || Q_1).
3. Compute iter-20 qnn features Q_2 from X_stacked (i.e., kNN in joint space → weighted quantiles of y).
4. Return Q_2 (5 columns per row).

Why this is non-trivial: in stage 2, the kNN is biased toward rows whose target distribution shape matches the query. This is a "what does y look like for rows
whose target distribution shape matches mine?" — analogous to second-order smoothing in time-series. For rare classes, this should sharpen the rare-positive signal.

Leakage discipline: Mode A re-fits BOTH layers per fold on train-fold rows; val-fold rows queried against fold bank.

Reference: similar to two-step kernel methods in matched-shape regression; or stacked target encoding (Pargent et al. 2022) but with quantile output.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input
from .quantile_neighbours import compute_quantile_neighbours

logger = logging.getLogger(__name__)


def compute_stacked_quantile_neighbours(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    k: int = 32,
    quantile_grid: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    softmax_temp: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "sqnn",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Stacked quantile-neighbours: apply qnn to (X || qnn_features).

    Output: ``len(quantile_grid)`` columns, prefixed with ``{column_prefix}``.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    # Layer 1: compute qnn features on standard X.
    Q1_train_df = compute_quantile_neighbours(
        X_train=X_train, y_train=y_train, X_query=None, splitter=splitter,
        seed=seed, k=k, quantile_grid=quantile_grid, softmax_temp=softmax_temp,
        standardize=standardize, column_prefix=f"{column_prefix}_l1",
        dtype=np.float32,
    )
    Q1_train = Q1_train_df.to_numpy().astype(np.float32)

    if X_query is not None:
        Q1_query_df = compute_quantile_neighbours(
            X_train=X_train, y_train=y_train, X_query=X_query, splitter=None,
            seed=seed, k=k, quantile_grid=quantile_grid, softmax_temp=softmax_temp,
            standardize=standardize, column_prefix=f"{column_prefix}_l1",
            dtype=np.float32,
        )
        Q1_query = Q1_query_df.to_numpy().astype(np.float32)
        X_stacked_train = np.concatenate([np.asarray(X_train, dtype=np.float32), Q1_train], axis=1)
        X_stacked_query = np.concatenate([np.asarray(X_query, dtype=np.float32), Q1_query], axis=1)
        Q2_df = compute_quantile_neighbours(
            X_train=X_stacked_train, y_train=y_train, X_query=X_stacked_query, splitter=None,
            seed=seed + 1, k=k, quantile_grid=quantile_grid, softmax_temp=softmax_temp,
            standardize=standardize, column_prefix=column_prefix,
            dtype=dtype,
        )
        return Q2_df

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    X_stacked_train = np.concatenate([np.asarray(X_train, dtype=np.float32), Q1_train], axis=1)
    Q2_df = compute_quantile_neighbours(
        X_train=X_stacked_train, y_train=y_train, X_query=None, splitter=splitter,
        seed=seed + 1, k=k, quantile_grid=quantile_grid, softmax_temp=softmax_temp,
        standardize=standardize, column_prefix=column_prefix,
        dtype=dtype,
    )
    logger.info("stacked_qnn: layer 1 output shape %s, layer 2 output shape %s", Q1_train.shape, Q2_df.shape)
    return Q2_df
