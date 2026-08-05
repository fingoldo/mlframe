"""FE_TRANSFORMER_A-5: compute_conformal_locally_adaptive_features's output column names were hardcoded
to _width_a01/_width_a02 regardless of the actual configurable alphas parameter values -- mislabels
columns for non-default alphas and IndexErrors/silently drops columns when len(alphas) != 2."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from mlframe.feature_engineering.transformer.conformal_locally_adaptive import compute_conformal_locally_adaptive_features


def _make_data(n=200, seed=0):
    """Deterministic (X_train, y_train, X_query) regression fixture."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=n).astype(np.float32)
    Xq = rng.normal(size=(20, 4)).astype(np.float32)
    return X, y, Xq


def test_single_alpha_does_not_crash_and_names_column_correctly():
    """len(alphas)==1 must not IndexError, and the single width column must be named after the actual alpha."""
    X, y, Xq = _make_data()
    out = compute_conformal_locally_adaptive_features(X, y, X_query=Xq, seed=0, alphas=(0.15,))
    assert set(out.columns) == {"cla_width_a15", "cla_sigma_hat", "cla_pred", "cla_width_ratio"}
    assert out.shape[0] == Xq.shape[0]


def test_three_alphas_all_get_named_columns_not_silently_dropped():
    """len(alphas)==3 must produce 3 correctly-named width columns, not silently drop the third."""
    X, y, Xq = _make_data(seed=1)
    out = compute_conformal_locally_adaptive_features(X, y, X_query=Xq, seed=0, alphas=(0.1, 0.2, 0.3))
    assert set(out.columns) == {"cla_width_a10", "cla_width_a20", "cla_width_a30", "cla_sigma_hat", "cla_pred", "cla_width_ratio"}
    assert out.shape[0] == Xq.shape[0]


def test_default_two_alphas_column_names_unchanged():
    """Default alphas=(0.1, 0.2) must still produce the original _width_a10/_width_a20 column names
    (renamed from the old hardcoded _width_a01/_width_a02 -- those never matched the actual default
    alpha values 0.1/0.2 in the first place)."""
    X, y, Xq = _make_data(seed=2)
    out = compute_conformal_locally_adaptive_features(X, y, X_query=Xq, seed=0)
    assert set(out.columns) == {"cla_width_a10", "cla_width_a20", "cla_sigma_hat", "cla_pred", "cla_width_ratio"}
