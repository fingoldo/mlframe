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
    logits = 2.0 * X_numeric[:, 0] + 3.0 * X_numeric[:, 1] + (cat_1 == "cat_A_10").astype(float) * 5.0
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


def make_sklearn_classification_df(
    n_samples: int = 200,
    n_features: int = 20,
    n_informative: int = 5,
    *,
    n_redundant: int = 0,
    n_classes: int = 2,
    n_clusters_per_class: int = 1,
    class_sep: float = 1.0,
    shuffle: bool = True,
    seed: int = 42,
    column_prefix: str = "f",
):
    """Wrap ``sklearn.datasets.make_classification`` and return a DataFrame.

    Centralises the very common pattern ``X, y = make_classification(...); X_df = pd.DataFrame(X, columns=[f"f{i}" ...])``
    that recurs 30+ times across ``tests/feature_selection``. Returns ``(X_df, y, feature_names)``; tests that need
    the raw ndarray can still pull ``X_df.to_numpy()``. Kwargs mirror the sklearn names so existing call sites
    migrate by signature renaming only. The ``column_prefix`` knob keeps the pre-existing ``f0..fN`` / ``feature_0..N``
    naming conventions intact.

    Default-kwargs divergence vs sklearn (verify before migrating asserting-on-seed-specific-values tests):
    ``n_clusters_per_class=1`` here (sklearn default is 2) and ``class_sep=1.0`` here (matches sklearn). Sites that
    were relying on sklearn's ``n_clusters_per_class=2`` default produce different (X, y) under the builder default
    and must pass ``n_clusters_per_class=2`` explicitly. Symptom: tests asserting per-feature behaviour (knockoff W
    signs, MI floors, recall on informative features) flip from pass to fail post-migration.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        shuffle=shuffle,
        random_state=seed,
    )
    feature_names = [f"{column_prefix}{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


def make_sklearn_regression_df(
    n_samples: int = 200,
    n_features: int = 20,
    n_informative: int = 5,
    *,
    noise: float = 0.1,
    seed: int = 42,
    column_prefix: str = "f",
):
    """Wrap ``sklearn.datasets.make_regression`` and return a DataFrame.

    Companion to ``make_sklearn_classification_df``. Returns ``(X_df, y, feature_names)``. Same migration ergonomics:
    kwargs mirror sklearn, ``column_prefix`` knob preserves existing column-name conventions.
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )
    feature_names = [f"{column_prefix}{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


def make_informative_noise_classification(
    n_samples: int = 200,
    n_informative: int = 5,
    n_noise: int = 15,
    *,
    seed: int = 42,
):
    """Hand-built informative+noise dataset (no sklearn dependency).

    Replicates the recurring ``simple_classification_data`` / ``high_dimensional_data`` /
    ``imbalanced_classification_data`` pattern from ``tests/feature_selection/conftest.py``: the first ``n_informative``
    cols drive a sign-of-linear-combination label, the remaining ``n_noise`` cols are pure standard-normal noise. Uses
    a local Generator -- never touches the global numpy RNG. Returns ``(X_df, y, informative_indices)``.
    """
    rng = np.random.default_rng(seed)
    X_informative = rng.standard_normal(size=(n_samples, n_informative))
    score = X_informative[:, 0] + (X_informative[:, 1] if n_informative > 1 else 0.0)
    if n_informative > 2:
        score = score - X_informative[:, 2]
    y = (score > 0).astype(int)
    X_noise = rng.standard_normal(size=(n_samples, n_noise))
    X = np.hstack([X_informative, X_noise])
    feature_names = [f"informative_{i}" for i in range(n_informative)] + [f"noise_{i}" for i in range(n_noise)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, list(range(n_informative))
