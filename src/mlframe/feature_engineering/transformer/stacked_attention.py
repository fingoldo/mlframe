"""Stacked row-attention - multi-layer transformer-style depth.

A single row-attention call gives the model a "what's the average y in my immediate neighbourhood" feature. Stacking 2 layers gives:

- Layer 1: per-row aggregate of y at neighbours in raw X space.
- Layer 2: per-row aggregate of y at neighbours in LAYER-1-OUTPUT space.

Layer 2's similarity is computed in the (y_mean_per_head)-space produced by layer 1. So layer 2 effectively asks: "which other rows have similar neighbourhood-y
patterns to mine?" Mathematically this is iterated label smoothing / label propagation, but with different similarity definitions at each layer.

Why this might work for boostings: a single row-attention adds local-mean-y features that boostings find useful when local-manifold structure dominates (kin8nm).
Stacking adds *second-order* information: clusters of rows that share similar neighbourhood patterns. Boostings can then split on "is this row in a high-mean-y
neighbourhood whose neighbours also tend to have high-mean-y" — a structural feature single-layer attention doesn't provide.

OOF discipline preserved: each layer's compute_row_attention call uses its own splitter; layer 2 is fed layer 1's OOF outputs (leak-free per the row-attention
discipline) and produces fresh OOF outputs for layer 3 onwards.

Reference: stacked transformer encoders (Vaswani et al. 2017); label propagation (Zhu & Ghahramani 2002).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Union

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input
from .row_attention import compute_row_attention

logger = logging.getLogger(__name__)


def compute_stacked_row_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    n_layers: int = 2,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    softmax_temp: float = 1.0,
    standardize: bool = True,
    projection: Literal["random", "pls"] = "random",
    gpu_stage4: Union[bool, Literal["auto"]] = False,
    return_all_layers: bool = True,
    column_prefix: str = "stack",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Run row-attention ``n_layers`` times, feeding each layer's output as the next layer's input.

    Layer 0 sees raw X (Mode A: OOF, or Mode B: full-train bank for X_query). It produces ``y_mean`` per head -> shape (N, n_heads).
    Layer 1 takes layer-0 output as its new "X" and runs row-attention again. Similarity is now computed in the y_mean-per-head space — rows that have similar
    layer-0 neighbourhood patterns will be close in layer-1's metric.
    Layer 2..n: keep going if requested. Diminishing returns past layer 2 in our experiments.

    Returns: polars DataFrame
        - If ``return_all_layers=True`` (default): concat of every layer's output, shape ``(N, n_layers * n_heads)``. Columns named ``{prefix}_L{layer}_h{head}``.
        - If ``return_all_layers=False``: only the final layer's output, shape ``(N, n_heads)``.

    Per-layer seed is ``seed + 1000 * layer`` so each layer uses independent random projections (when projection="random") or independent PLS noise (when "pls").

    ``gpu_stage4`` (default ``False``, unlike ``compute_row_attention``'s own ``"auto"`` default): every layer's ``compute_row_attention`` call forces CPU stage-4
    scoring unless the caller opts in here. Each of the ``n_layers`` calls pays its own stage-4 cost, so a GPU-equipped caller stacking several layers on a large
    ``k`` may want ``gpu_stage4="auto"`` or ``True``.

    Mode A (X_query=None): OOF features for X_train rows via the splitter. Layer i in Mode A is fed layer (i-1)'s OOF outputs as new X.
    Mode B (X_query!=None): Mode A for X_train AND Mode B for X_query at each layer. Layer i's bank is built from layer (i-1)'s OOF train outputs, and queried
    with layer (i-1)'s Mode-B X_query outputs.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1; got {n_layers}.")

    current_X_train = X_train
    current_X_query = X_query
    layer_outputs_train: list[np.ndarray] = []
    layer_outputs_query: list[np.ndarray] = []

    for layer in range(n_layers):
        layer_seed = seed + 1000 * layer
        # head_dim for this layer cannot exceed current_X_train.shape[1]; clamp adaptively (later layers have low-d input).
        layer_head_dim = min(head_dim, max(2, current_X_train.shape[1] - 1))
        # Mode A: OOF y_mean for X_train. Pass caller dtype through so .to_numpy() yields the requested dtype directly; the trailing astype(..., copy=False) is
        # then a no-op identity. Without dtype= the inner default is float32 -- callers using float64 paid a full-buffer float32->float64 copy per layer.
        out_train: np.ndarray = compute_row_attention(
            X_train=current_X_train, y_train=y_train, X_query=None, splitter=splitter,
            seed=layer_seed, n_heads=n_heads, head_dim=layer_head_dim, k=k,
            softmax_temp=softmax_temp, aggregate=("y_mean",), standardize=standardize,
            projection=projection, gpu_stage4=gpu_stage4, dedupe_threshold=None,
            allow_overcomplete=True, dtype=dtype,
        ).to_numpy().astype(dtype, copy=False)
        layer_outputs_train.append(out_train)

        if X_query is not None:
            # Mode B for X_query (single-pass with full layer-current_X_train bank).
            out_query: np.ndarray = compute_row_attention(
                X_train=current_X_train, y_train=y_train, X_query=current_X_query, splitter=splitter,
                seed=layer_seed, n_heads=n_heads, head_dim=layer_head_dim, k=k,
                softmax_temp=softmax_temp, aggregate=("y_mean",), standardize=standardize,
                projection=projection, gpu_stage4=gpu_stage4, dedupe_threshold=None,
                allow_overcomplete=True, dtype=dtype,
            ).to_numpy().astype(dtype, copy=False)
            layer_outputs_query.append(out_query)

        # Next layer sees current layer's output as input.
        # IMPORTANT: layer i+1's standardiser will refit per fold on layer i's OOF output; that's correct (the standardiser is fit on what is observed as "X" at
        # that layer, which is per-fold for OOF mode). Leakage is preserved because OOF-from-this-fold rows attend only to OOF-from-other-fold rows.
        current_X_train = out_train
        if X_query is not None:
            current_X_query = out_query
        logger.info("stacked attention: layer %d/%d done, output shape %s", layer + 1, n_layers, out_train.shape)

    if return_all_layers:
        train_concat = np.concatenate(layer_outputs_train, axis=1)
        if X_query is not None:
            query_concat = np.concatenate(layer_outputs_query, axis=1)
    else:
        train_concat = layer_outputs_train[-1]
        if X_query is not None:
            query_concat = layer_outputs_query[-1]

    # Build column names and pack into polars frame. Always emit train output (the function's contract: Mode A returns OOF for train, Mode B returns features
    # at X_query). Caller decides which to use.
    final = query_concat if X_query is not None else train_concat
    n_cols = final.shape[1]
    if return_all_layers:
        names = [f"{column_prefix}_L{layer}_h{h}" for layer in range(n_layers) for h in range(n_heads)]
    else:
        names = [f"{column_prefix}_h{h}" for h in range(n_cols)]
    return pl.DataFrame({name: final[:, i] for i, name in enumerate(names)})
