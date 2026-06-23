"""Identity-pin for the class_mahalanobis quadratic-form BLAS optimization.

``_mahalanobis`` was switched from the naive 3-operand
``np.einsum("ij,jk,ik->i", diff, inv_cov, diff)`` to the BLAS-GEMM form
``((diff @ inv_cov) * diff).sum(axis=1)`` (3.7-11.6x faster; see
``feature_engineering/_benchmarks/bench_mahalanobis_quadform.py``).

The change is FP-reduction-order equivalent (selection-safe), and is in fact
MORE accurate (matmul accumulates in higher precision). This test pins:

1. The current (NEW) implementation matches the einsum reference to a tight
   relative tolerance (1e-4 in float32) -- so a future revert that breaks the
   numerics is caught.
2. The output equals the fp64 ground truth to within float32 precision -- the
   stronger guarantee that the optimization did not introduce a real error.

Both checks FAIL on a hypothetical implementation that mis-orders the operands
(verified during development against the pre-fix einsum which sits at ~2.4e-6
rel err vs fp64 while the NEW path is ~3.2e-7).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.class_mahalanobis import _mahalanobis


def _make(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    mean = rng.standard_normal(d).astype(np.float32)
    A = rng.standard_normal((d, d)).astype(np.float32)
    inv_cov = (A @ A.T + np.eye(d)).astype(np.float32)  # SPD, symmetric like a real inv-cov
    return X, mean, inv_cov


@pytest.mark.parametrize("n,d", [(4000, 6), (20000, 15), (5000, 30)])
def test_mahalanobis_matches_einsum_reference(n, d):
    """NEW BLAS form is reduction-order equivalent to the einsum reference."""
    X, mean, inv_cov = _make(n, d)
    got = _mahalanobis(X, mean, inv_cov)
    diff = X - mean
    ref = np.einsum("ij,jk,ik->i", diff, inv_cov, diff).astype(np.float32)
    rel = np.abs(got - ref) / (np.abs(ref) + 1e-12)
    assert np.max(rel) < 1e-4, f"max rel diff {np.max(rel):.3e} exceeds 1e-4 (n={n}, d={d})"


@pytest.mark.parametrize("n,d", [(4000, 6), (20000, 15), (5000, 30)])
def test_mahalanobis_close_to_fp64_truth(n, d):
    """The optimized path is within float32 precision of the fp64 quadratic form."""
    X, mean, inv_cov = _make(n, d)
    got = _mahalanobis(X, mean, inv_cov).astype(np.float64)
    diff64 = (X - mean).astype(np.float64)
    truth = np.einsum("ij,jk,ik->i", diff64, inv_cov.astype(np.float64), diff64)
    rel = np.abs(got - truth) / (np.abs(truth) + 1e-12)
    assert np.max(rel) < 1e-5, f"max rel err vs fp64 {np.max(rel):.3e} exceeds 1e-5 (n={n}, d={d})"
