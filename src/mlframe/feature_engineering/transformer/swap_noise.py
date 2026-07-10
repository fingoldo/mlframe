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

from typing import Optional

import numpy as np


def swap_noise_augment(X: np.ndarray, swap_prob: float = 0.15, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Corrupt ``X`` by replacing a ``swap_prob`` fraction of each column's cells with that column's value
    from a different, randomly permuted row.

    Parameters
    ----------
    X
        ``(n, d)`` numeric array.
    swap_prob
        Fraction of cells per column selected for corruption (independently per column).
    rng
        ``np.random.Generator``; a fresh default one is used if None.

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
    if n < 2 or swap_prob == 0.0:
        return X_out

    mask = rng.random((n, d)) < swap_prob
    for j in range(d):
        col_mask = mask[:, j]
        n_swap = int(col_mask.sum())
        if n_swap == 0:
            continue
        perm = rng.permutation(n)[:n_swap]
        X_out[col_mask, j] = X[perm, j]
    return X_out


__all__ = ["swap_noise_augment"]
