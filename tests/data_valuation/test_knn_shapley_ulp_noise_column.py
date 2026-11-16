"""A column that carries only float rounding noise must not change knn_shapley values under standardisation."""

import numpy as np

from mlframe.data_valuation._knn_shapley import knn_shapley


def test_ulp_noise_column_does_not_change_standardised_values():
    """A near-constant column whose spread is one ulp of its mean has no information; standardising it used to amplify that noise into a feature."""
    rng = np.random.default_rng(0)
    n, m = 60, 20
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    Xv = rng.normal(size=(m, 2))
    yv = (Xv[:, 0] > 0).astype(int)

    def with_noise_column(A):
        """Append a constant column perturbed by one ulp per row."""
        jitter = rng.integers(-3, 4, size=len(A)) * np.spacing(0.1)
        return np.column_stack([A, 0.1 + jitter])

    base = knn_shapley(X, y, Xv, yv, k=3, standardize=True)
    noisy = knn_shapley(with_noise_column(X), y, with_noise_column(Xv), yv, k=3, standardize=True)
    np.testing.assert_allclose(noisy, base, rtol=0, atol=1e-9)
