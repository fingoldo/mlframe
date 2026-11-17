"""``_batched_gate_scalars`` (one ``pair_gram_blocks`` launch for all pairs) returns the four gate scalars of the per-pair normal-equations reference to 1e-9 relative, including a pair of the same operand twice and two identical operands (rank-deficient Gram), and the missing-operand case returns None."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_pure_form_retention_gpu_resident as m
from tests.feature_selection.gpu._gpu_support import has_device


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


def _reference(cp, V, basis, names, pairs, y):
    """The per-pair loop the batched version replaced (same ridge and solve)."""
    P = len(pairs)
    n = V.shape[0]
    G = cp.empty((P, 12, 12))
    rhs = cp.empty((P, 12))
    xy = cp.empty((P, 12))
    sfc, fyr, fstd = cp.zeros(P), cp.zeros(P), cp.zeros(P)
    for j, (a, b) in enumerate(pairs):
        fv = V[:, j]
        fstd[j] = fv.std()
        Xr = cp.concatenate([basis[names[a]], basis[names[b]]], axis=1)
        Xc = Xr - Xr.mean(axis=0, keepdims=True)
        fc = fv - fv.mean()
        G[j], rhs[j], xy[j] = Xc.T @ Xc, Xc.T @ fc, Xc.T @ y
        sfc[j], fyr[j] = cp.dot(fc, fc), cp.dot(fc, y)
    ridge = (1e-10 * cp.einsum("pii->p", G) / 12.0 + 1e-30)[:, None, None] * cp.eye(12)
    beta = cp.linalg.solve(G + ridge, rhs[:, :, None])[:, :, 0]
    ss = sfc - (beta * rhs).sum(axis=1)
    return cp.asnumpy(cp.stack([fstd, cp.sqrt(cp.maximum(ss, 0.0) / n), ss, fyr - (beta * xy).sum(axis=1)], axis=1))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("n", [1000, 9001, 30_000])
def test_scalars_equal_the_per_pair_reference(n, dtype):
    """All four scalars agree for random pairs, a repeated operand and a duplicated basis."""
    cp = _cp()
    rng = np.random.default_rng(0)
    B, P = 8, 25
    basis = [cp.asarray(rng.standard_normal((n, 6))) for _ in range(B)]
    basis[3] = basis[2].copy()
    names = {f"c{i}": i for i in range(B)}
    pairs = [(f"c{rng.integers(0, B)}", f"c{rng.integers(0, B)}") for _ in range(P)]
    pairs[0], pairs[1] = ("c2", "c3"), ("c5", "c5")
    V = cp.asarray(rng.standard_normal((n, P)))
    y = cp.asarray(rng.standard_normal(n))
    y = y - y.mean()
    if dtype == "float32":  # the production target arrives in the criterion dtype: a float32 operand must be widened, not read as double
        y, V, basis = y.astype(cp.float32), V.astype(cp.float32), [b.astype(cp.float32) for b in basis]
    got = m._batched_gate_scalars(cp, V, basis, names, pairs, y, P)
    want = _reference(cp, V.astype(cp.float64), [b.astype(cp.float64) for b in basis], names, pairs, y.astype(cp.float64))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_missing_operand_returns_none():
    """An operand outside the resident base set sends the caller to the exact CPU path."""
    cp = _cp()
    rng = np.random.default_rng(1)
    basis = [cp.asarray(rng.standard_normal((100, 6)))]
    V = cp.asarray(rng.standard_normal((100, 1)))
    assert m._batched_gate_scalars(cp, V, basis, {"c0": 0}, [("c0", "zzz")], cp.zeros(100), 1) is None


def test_result_is_reproducible():
    """Two calls return bit-identical scalars (fixed-order reductions)."""
    cp = _cp()
    rng = np.random.default_rng(2)
    n, B, P = 20_000, 5, 12
    basis = [cp.asarray(rng.standard_normal((n, 6))) for _ in range(B)]
    names = {f"c{i}": i for i in range(B)}
    pairs = [(f"c{i % B}", f"c{(i * 3 + 1) % B}") for i in range(P)]
    V = cp.asarray(rng.standard_normal((n, P)))
    y = cp.asarray(rng.standard_normal(n))
    a = m._batched_gate_scalars(cp, V, basis, names, pairs, y, P)
    b = m._batched_gate_scalars(cp, V, basis, names, pairs, y, P)
    np.testing.assert_array_equal(a, b)
