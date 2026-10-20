"""SMOTE kernel wiring: the production minority path reaches the vectorized kernel, and the row-loop reference agrees with it.

``smote_synthesize_rowloop`` is listed in the gate's ``_SUPERSEDED`` with ``smote_synthesize_vectorized`` as its replacement, so
the parity test below is the evidence for that classification. The spy test fails if the production path stops delegating.
"""

import numpy as np

from mlframe.feature_engineering.transformer import _smote_kernels_shared as k


def _cloud(n=60, d=4, seed=1):
    """A standard-normal point cloud with a fixed seed."""
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float64)


def test_minority_vectorized_delegates_to_vectorized_kernel(monkeypatch):
    """The production minority path calls the vectorized kernel rather than repeating its loop."""
    calls = []
    real = k.smote_synthesize_vectorized

    def spy(*args, **kwargs):
        """Record the call, then delegate to the real kernel."""
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(k, "smote_synthesize_vectorized", spy)
    out = k.smote_synthesize_minority_vectorized(_cloud(), n_synthetic=20, k_neighbors=3, seed=7)
    assert len(calls) == 1
    assert out.shape == (20, 4)


def test_minority_vectorized_matches_vectorized_kernel_exactly():
    """Same seed, same inputs: the production wrapper returns the kernel's output bit for bit."""
    X = _cloud()
    a = k.smote_synthesize_minority_vectorized(X, n_synthetic=25, k_neighbors=4, seed=3)
    b = k.smote_synthesize_vectorized(X, 25, 4, 3)
    np.testing.assert_array_equal(a, b)


def test_rowloop_reference_agrees_with_vectorized_kernel():
    """The superseded row loop and its replacement produce the same synthetic rows to float32 rounding."""
    X = _cloud(n=80, d=5, seed=11)
    ref = k.smote_synthesize_rowloop(X, 40, 5, 9)
    fast = k.smote_synthesize_vectorized(X, 40, 5, 9)
    np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)


def test_minority_vectorized_fallback_for_single_row():
    """n_min < 2 keeps its explicit fallback and never reaches the kernel."""
    out = k.smote_synthesize_minority_vectorized(np.ones((1, 3)), n_synthetic=5, k_neighbors=2, seed=0)
    assert out.shape == (1, 3)
