"""Swap-noise data augmentation for tabular denoising-autoencoder training.

Source: Porto Seguro Safe Driver Prediction 1st place -- trained a denoising autoencoder on train+test
features corrupted by "swap noise" (per row, resample a fraction of columns with the SAME-COLUMN value
taken from a randomly chosen OTHER row) rather than additive/multiplicative Gaussian noise. Swap noise keeps
every corrupted value on its column's true marginal distribution (no out-of-range values, no distributional
shift for skewed/categorical-coded numeric columns), which is why it out-performs Gaussian corruption for
tabular DAEs specifically. Reused independently (same 1st-place-writeup corroboration) by MoA prediction's
DeepInsight CNN pipeline and Santander Customer Transaction Prediction's NN augmentation.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np

ColumnSwapProbs = Union[Mapping[int, float], Sequence[float], np.ndarray]


def _resolve_column_probs(column_swap_probs: ColumnSwapProbs, d: int, default: float) -> np.ndarray:
    """Build a per-column ``(d,)`` probability array from a dict (sparse, ``default`` fills the rest) or a
    dense array-like of length ``d``."""
    if isinstance(column_swap_probs, Mapping):
        probs = np.full(d, default, dtype=float)
        for idx, p in column_swap_probs.items():
            if not 0 <= idx < d:
                raise IndexError(f"column_swap_probs key {idx} out of range for {d} columns")
            probs[idx] = p
    else:
        probs = np.asarray(column_swap_probs, dtype=float)
        if probs.shape != (d,):
            raise ValueError(f"column_swap_probs array must have shape ({d},), got {probs.shape}")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"column_swap_probs entries must be in [0, 1], got {probs}")
    return probs


def swap_noise_augment(
    X: np.ndarray,
    swap_prob: float = 0.15,
    rng: Optional[np.random.Generator] = None,
    column_swap_probs: Optional[ColumnSwapProbs] = None,
) -> np.ndarray:
    """Corrupt ``X`` by replacing a fraction of each column's cells with that column's value from a
    different, randomly permuted row.

    Parameters
    ----------
    X
        ``(n, d)`` numeric array.
    swap_prob
        Uniform fraction of cells per column selected for corruption (independently per column). Also used
        as the fallback rate for columns not named in ``column_swap_probs`` when that's a dict.
    rng
        ``np.random.Generator``; a fresh default one is used if None.
    column_swap_probs
        Opt-in per-column override of ``swap_prob``: either a ``{column_index: prob}`` dict (columns absent
        from the dict fall back to ``swap_prob``) or a dense length-``d`` array-like of per-column rates.
        Lets callers corrupt noisy/low-signal columns more aggressively than clean/informative ones when
        pretraining a denoising autoencoder -- uniform corruption wastes DAE capacity reconstructing noise
        columns exactly while under-corrupting (and thus under-learning robust structure for) the columns
        that actually carry signal. ``None`` (default) reproduces the original uniform-``swap_prob``
        behavior bit-for-bit.

    Returns
    -------
    np.ndarray
        ``(n, d)`` corrupted copy of ``X`` (input is not mutated). Each corrupted cell's value comes from a
        full-column random permutation, so it's drawn from a genuinely different row of the SAME column --
        never out-of-distribution for that column, unlike Gaussian jitter.
    """
    if not 0.0 <= swap_prob <= 1.0:
        raise ValueError(f"swap_prob must be in [0, 1], got {swap_prob}")
    rng = rng if rng is not None else np.random.default_rng()
    n, d = X.shape
    X_out = np.array(X, copy=True)

    probs = _resolve_column_probs(column_swap_probs, d, swap_prob) if column_swap_probs is not None else np.full(d, swap_prob, dtype=float)
    if n < 2 or not probs.any():
        return X_out

    mask = rng.random((n, d)) < probs
    for j in range(d):
        col_mask = mask[:, j]
        n_swap = int(col_mask.sum())
        if n_swap == 0:
            continue
        perm = rng.permutation(n)[:n_swap]
        X_out[col_mask, j] = X[perm, j]
    return X_out


__all__ = ["swap_noise_augment"]
