"""The cached |corr| batch of the resident usability greedy gives the reference scores for the whole row set and a row subset, call after call."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_greedy_gpu_resident as ug


@pytest.mark.parametrize("subset", [False, True])
def test_cached_abscorr_equals_reference(subset):
    """Several residuals scored against the same candidates (with a constant and a NaN-free degenerate column) match the uncached reference."""
    rng = np.random.default_rng(0)
    n, p = 800, 12
    V = rng.standard_normal((n, p))
    V[:, 3] = 1.0  # constant column: score 0
    Vdev = cp.asarray(V)
    fn = ug._usability_greedy_gpu_step1_known_output_size(Vdev, cp, p)
    rows = cp.asarray(np.sort(rng.choice(n, 300, replace=False))) if subset else None
    m = 300 if subset else n
    for seed in range(4):
        resid = cp.asarray(np.random.default_rng(seed).standard_normal(m))
        np.testing.assert_allclose(fn(resid, rows), fn.reference(resid, rows), rtol=1e-12, atol=1e-14)
    flat = cp.zeros(m)
    np.testing.assert_array_equal(fn(flat, rows), fn.reference(flat, rows))
