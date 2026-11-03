"""The one-launch permutation null gives the per-permutation maxima of the fused MI kernel."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident, _resident_mi_candidates
from mlframe.feature_selection.filters._null_max_perm_gpu import null_max_mi_perms_gpu


def _block(n: int, rng) -> "cp.ndarray":
    """Continuous, rounded and few-level float32 candidates as a (K, n) column-major block."""
    cols = np.concatenate([rng.standard_normal((20, n)), np.round(rng.standard_normal((10, n)) * 2), rng.integers(0, 3, (6, n)).astype(float)])
    return cp.asarray(np.ascontiguousarray(cols.astype(np.float32)))


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("nbins", [5, 10])
@pytest.mark.parametrize("n_perm", [7, 70])
def test_null_maxima_equal_one_launch_per_permutation(n_classes, nbins, n_perm):
    """Bit-identical maxima for the binary bitset kernel, the generic kernel and more permutations than one launch holds (n not a multiple of 32)."""
    rng = np.random.default_rng(n_classes * 100 + nbins + n_perm)
    n = 5003
    mat = _block(n, rng)
    y = cp.asarray(rng.integers(0, n_classes, n).astype(np.int64))
    perms = np.stack([rng.permutation(n) for _ in range(n_perm)])
    cache: dict = {}
    want = np.array(
        [
            np.nanmax(_plugin_mi_classif_batch_cuda_resident(mat, y[cp.asarray(p)], nbins, y_min=0, n_classes=n_classes, relax_binning=True, keep_dtype=True, x_is_cm=True, candidate_cache=cache))
            for p in perms
        ]
    )
    _x, _cm, x_cm, interior = _resident_mi_candidates(mat, nbins, keep_dtype=True, relax_binning=True, x_is_cm=True)
    got = null_max_mi_perms_gpu(cp, x_cm, interior, y, perms, nbins, n_classes)
    assert got is not None
    assert np.array_equal(got, want)
