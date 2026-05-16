"""Shared synthetic-data helpers for training tests.

These helpers replace ad-hoc local `_make_*` factories scattered across many
test modules. They use `numpy.random.default_rng(seed)` so callers never touch
the legacy global numpy RNG state (which the root conftest resets to 0 between
tests). Re-exported as fixtures from tests/training/conftest.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def make_simple_classification_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 3,
    *,
    seed: int = 42,
):
    """Return (df_with_target, feature_names, cat_features, y) for binary classification.

    Uses a deterministic local Generator; does not mutate `np.random` global state.
    The first `n_informative` features drive a logistic decision boundary; the
    remaining are pure noise. No categorical features in the simple synthetic.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coefs = np.zeros(n_features)
    coefs[:n_informative] = np.array([2.0, 3.0, -1.5][:n_informative] + [0.0] * max(0, n_informative - 3))
    logits = X @ coefs
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    cat_features: list[str] = []
    return df, feature_names, cat_features, y


def make_simple_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,
    *,
    noise_std: float = 0.5,
    seed: int = 42,
):
    """Return (df_with_target, feature_names, y) for simple regression."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - 1.5 * X[:, 2] + rng.standard_normal(n_samples) * noise_std
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, feature_names, y


def make_categorical_classification_data(
    n_samples: int = 500,
    n_numeric: int = 5,
    *,
    seed: int = 42,
):
    """Return (df_with_target, feature_names, cat_features, y) with high-card cats."""
    rng = np.random.default_rng(seed)
    X_numeric = rng.standard_normal((n_samples, n_numeric))
    cat_1 = rng.choice([f"cat_A_{i}" for i in range(100)], n_samples)
    cat_2 = rng.choice([f"cat_B_{i}" for i in range(50)], n_samples)
    cat_3 = rng.choice(["X", "Y", "Z"], n_samples)
    logits = (
        2.0 * X_numeric[:, 0]
        + 3.0 * X_numeric[:, 1]
        + (cat_1 == "cat_A_10").astype(float) * 5.0
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    df = pd.DataFrame(X_numeric, columns=[f"num_{i}" for i in range(n_numeric)])
    df["cat_1"] = cat_1
    df["cat_2"] = cat_2
    df["cat_3"] = cat_3
    df["target"] = y
    feature_names = list(df.columns[:-1])
    cat_features = ["cat_1", "cat_2", "cat_3"]
    return df, feature_names, cat_features, y


def make_outlier_regression_data(
    n_samples: int = 500,
    n_features: int = 10,
    *,
    outlier_fraction: float = 0.1,
    seed: int = 42,
):
    """Regression with 10% outliers (RANSAC/Huber tests)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - 1.5 * X[:, 2] + rng.standard_normal(n_samples) * 0.5
    n_outliers = int(outlier_fraction * n_samples)
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    y[outlier_indices] += rng.choice([-1, 1], n_outliers) * rng.uniform(10, 20, n_outliers)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, feature_names, y
