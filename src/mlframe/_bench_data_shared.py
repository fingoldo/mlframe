"""Shared micro-benchmark synthetic-dataset helper used by several unrelated ``_benchmarks/``
packages (feature_selection, evaluation): independently duplicated across those scripts,
consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_regression_data(n: int, n_features: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic (X, y): ``n_features`` Gaussian columns, ``y`` a noisy sum of the first 3."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X.iloc[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).to_numpy()
    return X, y


def make_latent_projection_data(n: int, d: int, seed: int) -> np.ndarray:
    """Synthetic X: a 3-dimensional Gaussian latent projected up to ``d`` columns plus small noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, d))
    return latent @ W + rng.normal(scale=0.2, size=(n, d))


def make_binary_ensemble_pred_matrix(m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (preds, y) for ensemble-selection benches: ``m`` binary-classifier prediction rows over ``n``
    samples, noise level per row spanning ``linspace(0.4, 1.6, m)`` so models range from strong to weak signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    signal = 2 * y - 1
    noise_levels = np.linspace(0.4, 1.6, m)
    preds = np.empty((m, n))
    for i, nl in enumerate(noise_levels):
        preds[i] = np.clip(0.5 + 0.4 * signal + rng.normal(0, nl, n), 0, 1)
    return preds, y


def make_binary_noise_dataset(n_rows: int, n_cols: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic (df, y): ``n_cols`` pure-noise Gaussian columns, ``y`` an independent Bernoulli label -- every
    column has near-chance AUC by construction, for benching a drop-near-noise-univariate-AUC style screen."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_rows)
    df = pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])
    return df, y


def make_meta_stacker_oof(n_rows: int, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic OOF-prediction matrix (X) + target (y) for the lasso/elasticnet meta-stacker benches: the first two
    columns carry the signal (0.7/0.3 weighted mix), the rest are pure noise components."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    y = 0.7 * a + 0.3 * b + 0.05 * rng.normal(size=n_rows)
    extra = [rng.normal(size=n_rows) for _ in range(n_components - 2)]
    X = np.column_stack([a, b, *extra])
    return X, y
