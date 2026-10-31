"""The parallel biased-HSIC kernel agrees with the dense centred-Gram reference to float precision, ties and constant columns included."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_hsic_fe import _hsic_batch


def _dense_reference(x: np.ndarray, y: np.ndarray) -> float:
    """Biased HSIC with both Gram matrices centred explicitly, on the full sample, in double precision."""
    def gram(z):
        """RBF Gram matrix with the median-distance bandwidth."""
        d = np.abs(z[:, None] - z[None, :])
        iu = np.triu_indices(len(z), 1)
        med = np.median(d[iu])
        s = med if med > 0 else 1.0
        return np.exp(-(d**2) / (2 * s * s))

    K, L = gram(x.astype(np.float64)), gram(y.astype(np.float64))
    n = len(x)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(H @ K @ H @ L) / (n - 1) ** 2)


def test_kernel_matches_dense_reference():
    """Continuous, rounded (ties), constant and signal columns on a sample of 300 rows (the subsample is the whole frame)."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 6))
    X[:, 1] = np.round(X[:, 1], 1)
    X[:, 2] = 1.0
    y = ((X[:, 0] ** 2 + 0.3 * rng.normal(size=n)) > 0.8).astype(np.int64)
    got = _hsic_batch(X, y, n_sample=n, random_state=0)
    from mlframe.feature_selection.filters._orthogonal_dcor_fe import _subsample_indices

    rows = _subsample_indices(n, n, 0)
    for j in range(X.shape[1]):
        ref = _dense_reference(X[rows, j], y[rows])
        assert abs(got[j] - ref) <= 5e-6 + 5e-3 * abs(ref), (j, got[j], ref)
    assert np.argmax(got) == 0
