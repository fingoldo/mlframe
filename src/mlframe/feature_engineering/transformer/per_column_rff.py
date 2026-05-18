"""Per-column Random Fourier Features — each column gets its OWN random projection + cos/sin nonlinearity.

Vanilla RFF (in random_features.py) uses a single shared projection matrix ``W ∈ R^(d × m)`` so column j contributes via the j-th row of W. Per-column RFF instead
gives each column j its own random projection ``W_j ∈ R^(d_embed,)`` and bias ``b_j ∈ R^(d_embed,)``, so column j's lift is independent:

    F_ij = sqrt(2/(2*d_embed)) * [cos(x_ij * W_j + b_j), sin(x_ij * W_j + b_j)]   shape (d_embed * 2,) per column per row

Flatten across columns: ``F_i = concat_j F_ij`` of shape ``(d * 2 * d_embed,)`` per row.

Why this is "transformer-like":
- Equivalent to giving each column a learned-positional-encoding-like embedding (the per-column W_j picks a unique random direction in the embedding space).
- Captures per-column nonlinearities that vanilla shared RFF blends together.
- Downstream boostings (LGB/XGB/CB) split on individual columns: per-column RFF gives them a richer multi-frequency view of each input column without cross-column mixing dilution.

Compared to FT-Transformer (which has learned per-column embeddings): we use random embeddings (no backprop). Compared to vanilla RFF: shared projection vs per-column.

Reference: Rahimi-Recht 2007 RFF + per-column random projections (a frozen variant of FT-Transformer's column-level embeddings).
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_per_column_rff(
    X: Union[pl.DataFrame, np.ndarray],
    *,
    seed: int,
    d_embed_per_column: int = 4,
    sigma_scale: float = 1.0,
    standardize: bool = True,
    dtype: np.dtype = np.float32,
    column_prefix: str = "pcrff",
) -> pl.DataFrame:
    """Per-column Random Fourier Features. Each input column gets ``d_embed_per_column`` cos+sin features (so ``2 * d_embed_per_column`` per input column total).

    For column j with random projection ``w_j ~ N(0, sigma^-2)``, output is ``cos(x_j * w_j + b_j), sin(x_j * w_j + b_j)``.

    Output shape: ``(N, d_input * 2 * d_embed_per_column)``.

    Parameters:
        ``d_embed_per_column`` - number of (cos, sin) pairs per input column. 4 gives 8 RFF features per column → for d=8 input that's 64 output features.
        ``sigma_scale`` - bandwidth multiplier. 1.0 → unit bandwidth per column (assumes data is already roughly standardised); set to ``median_pairwise_distance`` per column for the classical heuristic.
        ``standardize`` - if True, RobustScaler-normalise each input column first (recommended for raw data).
    """
    from sklearn.preprocessing import RobustScaler
    seed = require_seed(seed)
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if X.dtype.kind in ("f", "i", "u") and X.dtype != dtype:
        X = X.astype(dtype, copy=False)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    validate_numeric_input(X, name="X", allow_fp16=True)

    if standardize:
        scaler = RobustScaler()
        X_std = scaler.fit_transform(X).astype(dtype, copy=False)
    else:
        X_std = X

    n, d_input = X_std.shape
    m = d_embed_per_column
    rng = np.random.default_rng(seed)
    # Per-column W_j and b_j.
    W = (rng.standard_normal((d_input, m)) / sigma_scale).astype(dtype, copy=False)  # (d_input, m)
    b = (rng.uniform(0, 2.0 * np.pi, size=(d_input, m))).astype(dtype, copy=False)   # (d_input, m)

    # For each column j, compute x_j * W_j (broadcasting): shape (n, m) per column.
    # Stack across columns: (n, d_input, m). Vectorise as einsum.
    # angles[n, j, i] = X_std[n, j] * W[j, i] + b[j, i]
    angles = X_std[:, :, None] * W[None, :, :] + b[None, :, :]   # (n, d_input, m)
    scale = float(np.sqrt(1.0 / m))
    cos_part = (scale * np.cos(angles)).astype(dtype, copy=False)
    sin_part = (scale * np.sin(angles)).astype(dtype, copy=False)

    # Flatten: (n, d_input * 2 * m). Order: column j cos features [0..m-1], then sin features [m..2m-1], then j+1, ...
    # This is friendly for downstream feature-importance plots (consecutive features belong to the same input column).
    out = np.empty((n, d_input * 2 * m), dtype=dtype)
    for j in range(d_input):
        out[:, j * 2 * m : j * 2 * m + m] = cos_part[:, j, :]
        out[:, j * 2 * m + m : (j + 1) * 2 * m] = sin_part[:, j, :]

    # Column names: per-input-column cos / sin pairs.
    names: list[str] = []
    for j in range(d_input):
        for i in range(m):
            names.append(f"{column_prefix}_c{j}_cos{i}")
        for i in range(m):
            names.append(f"{column_prefix}_c{j}_sin{i}")
    return pl.DataFrame({name: out[:, idx] for idx, name in enumerate(names)})
