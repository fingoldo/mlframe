"""Auto-encoder latent embeddings: unsupervised nonlinear representation via MLPRegressor reconstruction.

Iter 40 mechanism. BEYOND-FROZEN unsupervised — gradient-trained autoencoder, fundamentally different from supervised mechanisms (NCA, SMOTE/MIXUP) that may be
redundant with each other.

Mechanism:
1. Train sklearn MLPRegressor with target=X (autoencoder reconstruction). Architecture: input → hidden_size → bottleneck → hidden_size → input.
2. Extract the bottleneck activation per row (the latent code).
3. Expose as ``n_bottleneck`` features.

For binary or regression — same architecture; y is not used.

Why unsupervised matters for the mammography 1.3%-positive bottleneck:
- All winning mechanisms (SMOTE, MIXUP, denrat, NCA) use y signal.
- AE uses ONLY X — gives boostings a y-independent nonlinear representation.
- Latent code captures manifold structure that linear PCA cannot.

Why CB-blind: CB cannot fit a multi-layer nonlinear encoder via gradient. Latent code is an entirely orthogonal feature class.

Leakage discipline: AE refit per fold from train-fold rows only. Val-fold rows pass through fitted encoder.

Cost: sklearn MLPRegressor (hidden=8, max_iter=200) at N=4000 trains in ~2-5 sec per fold. Total: ~10-25 sec for full OOF.

Architecture: symmetric bottleneck `[input_dim, hidden_size, bottleneck_dim, hidden_size, input_dim]` with tanh activations.

Reference: Hinton & Salakhutdinov 2006 — autoencoder dimensionality reduction. Bengio et al. 2007 — deep architectures.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_autoencoder(X: np.ndarray, *, hidden_size: int, bottleneck_dim: int, max_iter: int, seed: int):
    """Fit a 4-layer symmetric MLP autoencoder: input → hidden → bottleneck → hidden → input.

    Returns (mlp, scaler) where scaler standardises inputs.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    # MLPRegressor symmetric bottleneck: hidden -> bottleneck -> hidden -> output(input_dim).
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden_size, bottleneck_dim, hidden_size),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=0.01,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ae.fit(X_s, X_s)
    return ae, scaler


def _extract_bottleneck(ae, scaler, X: np.ndarray, bottleneck_dim: int) -> np.ndarray:
    """Extract bottleneck activation for each row.

    MLPRegressor stores per-layer weights in coefs_ / intercepts_. Walk forward through layers up to the bottleneck.
    Architecture from _fit_autoencoder: hidden_layer_sizes=(hidden, bottleneck, hidden). So:
    - layer 0 weights: input → hidden
    - layer 1 weights: hidden → bottleneck (this is the latent layer)
    - layer 2 weights: bottleneck → hidden
    - layer 3 weights: hidden → output
    The bottleneck activation = tanh(tanh(X @ W0 + b0) @ W1 + b1).
    """
    X_s = scaler.transform(X)
    h1 = np.tanh(X_s @ ae.coefs_[0] + ae.intercepts_[0])
    bottleneck = np.tanh(h1 @ ae.coefs_[1] + ae.intercepts_[1])
    if bottleneck.shape[1] != bottleneck_dim:
        # Defensive: should match by construction, but handle slight architecture drift gracefully.
        pad = np.zeros((bottleneck.shape[0], bottleneck_dim), dtype=np.float32)
        cols = min(bottleneck.shape[1], bottleneck_dim)
        pad[:, :cols] = bottleneck[:, :cols]
        bottleneck = pad
    return bottleneck.astype(np.float32)


def compute_autoencoder_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    hidden_size: int = 8,
    bottleneck_dim: int = 4,
    max_iter: int = 200,
    column_prefix: str = "ae",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Auto-encoder bottleneck latent features (unsupervised beyond-frozen).

    Output: ``bottleneck_dim`` columns per row, named ``{prefix}_z{j}``.

    y_train is NOT used (unsupervised). The y parameter is retained for API symmetry with other transformer-FE functions.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)

    def _process(Xt: np.ndarray, Xq: np.ndarray, fold_seed: int) -> np.ndarray:
        ae, scaler = _fit_autoencoder(Xt, hidden_size=hidden_size, bottleneck_dim=bottleneck_dim, max_iter=max_iter, seed=fold_seed)
        return _extract_bottleneck(ae, scaler, Xq, bottleneck_dim=bottleneck_dim)

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        return {f"{column_prefix}_z{j}": feats[:, j].astype(dtype, copy=False) for j in range(feats.shape[1])}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, bottleneck_dim), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], int(seed) + fold_idx * 23)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("autoencoder: fold %d/%d done (n_train=%d, n_val=%d, bottleneck=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), bottleneck_dim)

    return pl.DataFrame(_make_df(out))
