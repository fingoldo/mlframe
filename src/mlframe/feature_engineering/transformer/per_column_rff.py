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
from typing import Union

import numba
import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


@numba.njit(parallel=True, cache=True, fastmath=False)
def _pcrff_fused_njit(X_std: np.ndarray, W: np.ndarray, b: np.ndarray, scale: np.float32) -> np.ndarray:  # pragma: no cover
    """Fused per-column RFF: compute each angle once and write scale*cos / scale*sin straight into
    the interleaved output, with no (n, d_input, m) ``angles``/``cos_part``/``sin_part`` temporaries
    and no Python per-column copy loop. Output layout matches the prior numpy path exactly
    (per column j: cos features [0..m), then sin features [m..2m)).
    """
    n, d_input = X_std.shape
    m = W.shape[1]
    out = np.empty((n, d_input * 2 * m), dtype=np.float32)
    for r in numba.prange(n):
        for j in range(d_input):
            base = j * 2 * m
            xj = X_std[r, j]
            for i in range(m):
                a = xj * W[j, i] + b[j, i]
                out[r, base + i] = scale * np.cos(a)
                out[r, base + m + i] = scale * np.sin(a)
    return out


def compute_per_column_rff(
    X: Union[pl.DataFrame, np.ndarray],
    *,
    seed: int,
    d_embed_per_column: int = 4,
    sigma_scale: float = 1.0,
    standardize: bool = True,
    dtype: type = np.float32,
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
        # bench-attempt-rejected (2026-05-24): allow_copy=False polars hint raises on any multi-column DataFrame (per-column Arrow chunk layout); fallback
        # branch adds code without speedup. The dtype gate below already short-circuits when dtypes match and astype(..., copy=False) is a 0.3 us no-op.
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

    _n, d_input = X_std.shape
    m = d_embed_per_column
    rng = np.random.default_rng(seed)
    # Per-column W_j and b_j.
    W: np.ndarray = (rng.standard_normal((d_input, m)) / sigma_scale).astype(dtype, copy=False)  # (d_input, m)
    b: np.ndarray = (rng.uniform(0, 2.0 * np.pi, size=(d_input, m))).astype(dtype, copy=False)  # (d_input, m)

    # Fused per-column RFF kernel: angle = x_j * W[j,i] + b[j,i] computed once per element and
    # written straight into the interleaved output (per column j: cos [0..m), then sin [m..2m)).
    # Replaces the prior path that materialised three full (n, d_input, m) temporaries
    # (angles / cos_part / sin_part) plus a Python per-column copy loop: ~2.3-6.8x faster across
    # n in {2k..500k} (bench_per_column_rff_fused_njit.py). The np.cos/np.sin float32 result is a
    # single-ULP (~3e-8) off the prior numpy float32 ufunc — selection-equivalent for downstream
    # boostings, not bit-identical (see that bench + test_per_column_rff_fused_njit_selection).
    scale = np.float32(np.sqrt(1.0 / m))
    out = _pcrff_fused_njit(
        np.ascontiguousarray(X_std, dtype=np.float32),
        np.ascontiguousarray(W, dtype=np.float32),
        np.ascontiguousarray(b, dtype=np.float32),
        scale,
    )
    if dtype != np.float32:
        out = out.astype(dtype, copy=False)

    # Column names: per-input-column cos / sin pairs.
    names: list[str] = []
    for j in range(d_input):
        for i in range(m):
            names.append(f"{column_prefix}_c{j}_cos{i}")
        for i in range(m):
            names.append(f"{column_prefix}_c{j}_sin{i}")
    return pl.DataFrame({name: out[:, idx] for idx, name in enumerate(names)})
