"""SIR solves its covariance eigenproblem in double precision even when the criterion dtype of the columns is float32."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin
from mlframe.feature_selection.filters._sliced_inverse_regression_fe import generate_sir_direction_features, sir_direction_features


def _reference_direction(X: np.ndarray, y: np.ndarray, n_slices: int = 10) -> np.ndarray:
    """Top SIR direction of the float64 columns, solved entirely in double precision."""
    X = X.astype(np.float64)
    n, p = X.shape
    ids = _quantile_bin(y.astype(np.float64), nbins=n_slices)
    mean = X.mean(0)
    Xc = X - mean
    sigma = Xc.T @ Xc / n
    m = np.zeros((p, p))
    for h in np.unique(ids):
        k = ids == h
        d = X[k].mean(0) - mean
        m += k.sum() / n * np.outer(d, d)
    w, v = scipy.linalg.eigh(m, sigma + 1e-8 * np.trace(sigma) / p * np.eye(p))
    return v[:, np.argsort(w)[::-1][0]]


def _near_collinear_case(seed: int):
    """Two columns equal up to 1e-4 noise on a large offset; the target is the amplified difference, so the signal lives in the smallest eigen-direction of Sigma."""
    r = np.random.default_rng(seed)
    n = 4000
    x1 = r.standard_normal(n) + 50.0
    x2 = x1 + 1e-4 * r.standard_normal(n)
    x3 = r.standard_normal(n)
    y = (x2 - x1) * 1e4 + 0.3 * r.standard_normal(n)
    return pd.DataFrame({"a": x1, "b": x2, "c": x3}), y


def test_sir_direction_is_solved_in_double_precision_for_near_collinear_columns():
    """The frozen direction equals the float64 reference direction on near-collinear float32 columns (cosine >= 0.99 on every seed)."""
    for seed in range(4):
        X, y = _near_collinear_case(seed)
        _enc, payload = generate_sir_direction_features(X, ["a", "b", "c"], y, n_directions=1)
        assert payload, "the near-collinear fit must not be dropped as degenerate"
        v = payload["v"][:, 0].astype(np.float64)
        ref = _reference_direction(X.to_numpy(), y)
        cos = abs(v @ ref) / (np.linalg.norm(v) * np.linalg.norm(ref))
        assert cos >= 0.99, (seed, cos)


def test_sir_direction_features_matches_generate_projection():
    """The array entry point and the frame entry point share one solve: their first projection agrees."""
    X, y = _near_collinear_case(0)
    enc, _payload = generate_sir_direction_features(X, ["a", "b", "c"], y, n_directions=1)
    got = sir_direction_features(X.to_numpy(), y, n_directions=1)
    np.testing.assert_allclose(got[:, 0], enc.iloc[:, 0].to_numpy(), rtol=1e-6, atol=1e-6)


def test_recipe_replay_equals_fit_column_exactly():
    """The single-direction recipe replay and the frame replay reproduce the fitted projection bit for bit (no matmul-vs-matvec rounding gap)."""
    from mlframe.feature_selection.filters._sliced_inverse_regression_fe import (
        apply_sir_direction,
        build_sir_direction_recipe,
        engineered_name_sir_direction,
    )
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    r = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame(r.standard_normal((n, 5)) * np.array([1.0, 3.0, 0.5, 2.0, 1.5]) + 7.0, columns=[f"x{i}" for i in range(5)])
    y = X.to_numpy() @ np.array([0.6, 0.5, 0.4, 0.3, 0.4]) + 0.2 * r.standard_normal(n)
    cols = list(X.columns)
    enc, payload = generate_sir_direction_features(X, cols, y, n_directions=2)
    np.testing.assert_array_equal(apply_sir_direction(X, payload).to_numpy(), enc.to_numpy())
    for i in range(2):
        name = engineered_name_sir_direction(cols, i)
        rec = build_sir_direction_recipe(name=name, idx=i, cols=payload["cols"], x_mean=payload["x_mean"], v=payload["v"])
        np.testing.assert_array_equal(np.asarray(apply_recipe(rec, X)), enc[name].to_numpy())
