"""Regression: the fused robust-scale combine reads its inputs at the dtype they are launched in, whatever dtype the matrix arrived in."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _reference(M):
    """Per-column 1.4826*MAD with the IQR/1.349 fallback and 0 for a degenerate column, in float64 numpy."""
    med = np.median(M, axis=0)
    mad = np.median(np.abs(M - med), axis=0)
    q25, q75 = np.percentile(M, [25, 75], axis=0)
    sc = 1.4826 * mad
    iqr = (q75 - q25) / 1.349
    return np.where(sc > 1e-12, sc, np.where(iqr > 1e-12, iqr, 0.0))


def test_robust_scale_float64_matrix_matches_reference():
    """A float64 matrix under the float32 crit dtype gives the float64 reference scale, not values read through a mismatched pointer type."""
    from mlframe.feature_selection.filters._gpu_resident_basis import _batched_quantiles, _gpu_robust_scale_batched

    rng = np.random.default_rng(0)
    M = rng.standard_normal((2000, 253)) * rng.uniform(0.5, 5.0, size=253)
    M[:, 7] = 3.0  # degenerate column: zero MAD and zero IQR
    Md = cp.asarray(M)
    med = _batched_quantiles(cp, Md, [0.5])[0]
    got = cp.asnumpy(_gpu_robust_scale_batched(cp, Md, med)).astype(np.float64)
    np.testing.assert_allclose(got, _reference(M), rtol=1e-4, atol=1e-6)
    assert got[7] == 0.0
