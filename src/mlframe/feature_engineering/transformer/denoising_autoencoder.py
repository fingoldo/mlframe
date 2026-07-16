"""Swap-noise denoising autoencoder (DAE) bottleneck features: unsupervised tabular representation learning.

Source: Porto Seguro Safe Driver Prediction 1st place -- a DAE trained on train+test features with swap
noise (see :mod:`swap_noise`) as the corruption strategy, with the hidden-layer activations fed into a
supervised NN as the main winning technique.

Distinct from :mod:`autoencoder` (:func:`compute_autoencoder_features`): that module trains a PLAIN
autoencoder (input == target, no corruption) -- a pure dimensionality-reduction / manifold-structure latent
code. This module instead trains a DENOISING autoencoder: the encoder sees a swap-noise-CORRUPTED input and
must reconstruct the CLEAN target, which forces the bottleneck to learn a representation robust to per-column
resampling noise rather than one that could simply memorize/copy-through a (near-)identity mapping. At
feature-extraction time (both OOF train rows and query rows), the CLEAN row is passed through the fitted
encoder -- corruption is a training-time-only regularizer, never applied to what the encoder actually embeds.

Leakage discipline mirrors :mod:`autoencoder`: refit per fold from train-fold rows only; val-fold / query rows
pass through the fitted encoder uncorrupted.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input
from .swap_noise import swap_noise_augment

logger = logging.getLogger(__name__)


def _fit_dae(X: np.ndarray, *, hidden_size: int, bottleneck_dim: int, max_iter: int, swap_prob: float, seed: int):
    """Fit a 4-layer symmetric MLP denoising autoencoder: swap-noise-corrupted input -> hidden -> bottleneck
    -> hidden -> CLEAN output. Returns (mlp, scaler)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    rng = np.random.default_rng(seed)
    X_corrupted = swap_noise_augment(X_s, swap_prob=swap_prob, rng=rng)

    dae = MLPRegressor(
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
        dae.fit(X_corrupted, X_s)  # corrupted -> clean: the denoising objective
    return dae, scaler


def _extract_bottleneck(dae: Any, scaler: Any, X: np.ndarray, bottleneck_dim: int) -> np.ndarray:
    """Extract the bottleneck activation for each (CLEAN, uncorrupted) row -- same forward-pass walk as
    :func:`autoencoder._extract_bottleneck` (layer 0: input->hidden, layer 1: hidden->bottleneck)."""
    X_s = scaler.transform(X)
    h1 = np.tanh(X_s @ dae.coefs_[0] + dae.intercepts_[0])
    bottleneck = np.tanh(h1 @ dae.coefs_[1] + dae.intercepts_[1])
    if bottleneck.shape[1] != bottleneck_dim:
        pad = np.zeros((bottleneck.shape[0], bottleneck_dim), dtype=np.float32)
        cols = min(bottleneck.shape[1], bottleneck_dim)
        pad[:, :cols] = bottleneck[:, :cols]
        bottleneck = pad
    return np.asarray(bottleneck.astype(np.float32))


def _extract_multilayer(dae: Any, scaler: Any, X: np.ndarray, bottleneck_dim: int) -> tuple[np.ndarray, list[str]]:
    """Extract and concatenate ALL three hidden-layer activations (encoder hidden -> bottleneck -> decoder
    hidden) for each (CLEAN, uncorrupted) row, instead of the bottleneck alone.

    Rationale: the bottleneck is a lossy compression tuned to minimize reconstruction error -- signal that
    is useful for a downstream supervised task but cheap to reconstruct (or occupies little reconstruction-
    error budget) can be squeezed out of a narrow bottleneck while still being fully present in the wider
    encoder/decoder hidden layers. Concatenating all three layers never loses what the bottleneck-only path
    captures (bottleneck columns are a strict subset) and additionally exposes the wider layers' activations.
    Returns (features, suffixes) where suffixes labels each output column's source layer for naming.
    """
    X_s = scaler.transform(X)
    h1 = np.tanh(X_s @ dae.coefs_[0] + dae.intercepts_[0])
    bottleneck = np.tanh(h1 @ dae.coefs_[1] + dae.intercepts_[1])
    if bottleneck.shape[1] != bottleneck_dim:
        pad = np.zeros((bottleneck.shape[0], bottleneck_dim), dtype=np.float32)
        cols = min(bottleneck.shape[1], bottleneck_dim)
        pad[:, :cols] = bottleneck[:, :cols]
        bottleneck = pad
    h2 = np.tanh(bottleneck @ dae.coefs_[2] + dae.intercepts_[2])

    feats = np.concatenate([h1, bottleneck, h2], axis=1).astype(np.float32)
    suffixes = [f"h1{j}" for j in range(h1.shape[1])] + [f"z{j}" for j in range(bottleneck.shape[1])] + [f"h2{j}" for j in range(h2.shape[1])]
    return feats, suffixes


def compute_denoising_autoencoder_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    hidden_size: int = 8,
    bottleneck_dim: int = 4,
    max_iter: int = 200,
    swap_prob: float = 0.15,
    column_prefix: str = "dae",
    dtype: type = np.float32,
    extract_layers: Literal["bottleneck", "multi"] = "bottleneck",
) -> pl.DataFrame:
    """Swap-noise denoising-autoencoder latent features (unsupervised).

    Output, ``extract_layers="bottleneck"`` (default, unchanged prior behavior): ``bottleneck_dim`` columns
    per row, named ``{prefix}_z{j}``.

    Output, ``extract_layers="multi"`` (opt-in): concatenates ALL three hidden-layer activations (encoder
    hidden -> bottleneck -> decoder hidden), ``2*hidden_size + bottleneck_dim`` columns per row, named
    ``{prefix}_h1{j}`` / ``{prefix}_z{j}`` / ``{prefix}_h2{j}``. Useful when some of the supervised signal is
    captured better by the wider (lower-compression) hidden layers than by the bottleneck alone -- the
    bottleneck is tuned to minimize reconstruction error, which can squeeze out signal that costs little
    reconstruction-error budget but is still predictive downstream.

    y_train is NOT used (unsupervised). The y parameter is retained for API symmetry with other
    transformer-FE functions.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)

    def _process(Xt: np.ndarray, Xq: np.ndarray, fold_seed: int) -> tuple[np.ndarray, list[str]]:
        """Fit the DAE on Xt (swap-noise corrupted -> clean) and extract latents for the CLEAN Xq."""
        dae, scaler = _fit_dae(Xt, hidden_size=hidden_size, bottleneck_dim=bottleneck_dim, max_iter=max_iter, swap_prob=swap_prob, seed=fold_seed)
        if extract_layers == "multi":
            return _extract_multilayer(dae, scaler, Xq, bottleneck_dim=bottleneck_dim)
        return _extract_bottleneck(dae, scaler, Xq, bottleneck_dim=bottleneck_dim), [f"z{j}" for j in range(bottleneck_dim)]

    def _make_df(feats: np.ndarray, suffixes: list[str]) -> dict[str, np.ndarray]:
        """Build a name-to-column mapping from the latent feature matrix, prefixing and casting each column."""
        return {f"{column_prefix}_{suffixes[j]}": feats[:, j].astype(dtype, copy=False) for j in range(feats.shape[1])}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats, suffixes = _process(X_train_f, Xq, seed)
        return pl.DataFrame(_make_df(feats, suffixes))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_out_cols = (2 * hidden_size + bottleneck_dim) if extract_layers == "multi" else bottleneck_dim
    out: np.ndarray = np.zeros((n_train, n_out_cols), dtype=dtype)
    out_suffixes: list[str] = []
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats, out_suffixes = _process(X_train_f[train_idx], X_train_f[val_idx], int(seed) + fold_idx * 23)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("denoising_autoencoder: fold %d/%d done (n_train=%d, n_val=%d, out_cols=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), n_out_cols)

    return pl.DataFrame(_make_df(out, out_suffixes))


__all__ = ["compute_denoising_autoencoder_features"]
