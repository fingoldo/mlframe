"""Multi-temperature row-attention fusion: same projection, multiple softmax temperatures concatenated.

Rationale: classical row-attention picks ONE softmax_temp (1.0 by default). Adaptive bandwidth (iter 12) picks per-row temperature from local density. Multi-temperature
fusion is a third path: run attention at SEVERAL fixed temperatures (sharp, medium, smooth) and concatenate the outputs. The downstream boosting then splits on
whichever scale of locality matters most for each region of the input space.

Mechanism:
1. Build projection (random/PLS/importance) — single set of projection matrices.
2. Project X_train (Mode A) or X_query (Mode B) once.
3. Run stage-4 row-attention at EACH ``temperature_grid`` value (default ``(0.3, 1.0, 3.0)`` — sharp / medium / smooth).
4. Concatenate the per-temperature outputs.

For boostings: a sharp attention output captures "what y looks like at exactly this point"; a smooth output captures "what y looks like in this region". Boostings can
split on either, choosing the scale that's most informative per leaf. Different from `+adaptive` which collapses to one per-row choice; multi-temp preserves all scales.

Expected gain over adaptive: when the optimal temperature is heterogeneous across X-space, multi-temp gives the boosting freedom to pick locally rather than forcing
one global decision.

Reference: kernel ridge regression with multi-bandwidth sums (Aronszajn 1950); ensembling of attention layers with different temperatures (a frozen analog of
multi-head attention's per-head bandwidth).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input
from .row_attention import compute_row_attention

logger = logging.getLogger(__name__)


def compute_multi_temperature_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    temperatures: Tuple[float, ...] = (0.3, 1.0, 3.0),
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    projection: Literal["random", "pls", "importance"] = "pls",
    column_prefix: str = "mtemp",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-temperature row-attention fusion.

    For each temperature in ``temperatures``, runs ``compute_row_attention`` with that softmax_temp and projection. Concatenates the per-temperature outputs into
    one wide feature matrix. Total output columns: ``len(temperatures) * n_heads * len(aggregate)``.

    Per-temperature ``column_prefix`` is suffixed with ``_T{temp}`` so columns are uniquely named across temperatures.

    Note: this internally calls ``compute_row_attention`` once per temperature, so it runs the projection / OOF loop ``len(temperatures)`` times. For
    ``temperatures=(0.3, 1.0, 3.0)`` that's a 3x cost over single-temperature attention. Use a smaller k if budget-constrained.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if not temperatures:
        raise ValueError("temperatures must be a non-empty tuple of positive floats.")

    pieces: list[pl.DataFrame] = []
    for temp in temperatures:
        if temp <= 0:
            raise ValueError(f"All temperatures must be positive; got {temp}.")
        piece = compute_row_attention(
            X_train=X_train, y_train=y_train, X_query=X_query, splitter=splitter,
            seed=seed, n_heads=n_heads, head_dim=head_dim, k=k,
            softmax_temp=float(temp), aggregate=aggregate, projection=projection,
            gpu_stage4=False, dedupe_threshold=None,
            column_prefix=f"{column_prefix}_T{temp:.2f}".replace(".", "p"),
        )
        pieces.append(piece)
        logger.info("multi_temperature: temp=%.2f done, output shape %s", temp, piece.shape)

    return pl.concat(pieces, how="horizontal")
