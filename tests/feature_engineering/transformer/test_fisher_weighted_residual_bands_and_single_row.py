"""fisher_weighted_residual bands query rows on the quantity its thresholds were computed from, and accepts a single query row."""
import numpy as np

from mlframe.feature_engineering.transformer.fisher_weighted_residual import compute_fisher_weighted_residual_features


def _binary_data(n: int, seed: int = 0):
    """A binary target driven by the first feature plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.float32)
    return X, y


def test_query_rows_spread_over_the_bands_instead_of_collapsing_into_one():
    """Thresholds from cross-entropy and query rows banded on entropy put almost every query row in the lowest bands; same-quantity banding spreads them."""
    X, y = _binary_data(1500)
    Xq, _ = _binary_data(1000, seed=1)
    df = compute_fisher_weighted_residual_features(X, y, Xq, seed=3, task="binary", n_bands=5)
    counts = np.bincount(df["fishres_band"].to_numpy().astype(int), minlength=5)
    assert counts.min() >= 0.08 * 1000, counts


def test_a_single_query_row_does_not_raise():
    """A one-row query used to hit a read-only broadcast view on the in-place perturbation."""
    X, y = _binary_data(300)
    df = compute_fisher_weighted_residual_features(X, y, X[:1], seed=3, task="binary", n_bands=5)
    assert df.height == 1
