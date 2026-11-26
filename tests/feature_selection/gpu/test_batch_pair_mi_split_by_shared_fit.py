"""``dispatch_batch_pair_mi`` sets apart the pairs whose joint histogram does not fit the shared memory of the fast kernels: a few wide pairs no longer send the whole chunk to the strided row-chunked kernel."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_pair_mi_gpu as B


def _fixed_budget(monkeypatch, bytes_budget: int = 40_000) -> None:
    """A device-independent shared-memory budget: the joint histogram and the marginals as int32 against ``bytes_budget``."""
    monkeypatch.setattr(B, "shared_fused_kernel_fits_budget", lambda max_joint, ky: (max_joint * ky + max_joint) * 4 if (max_joint * ky + max_joint) * 4 <= bytes_budget else 0)


def _all_pairs(p: int) -> "tuple[np.ndarray, np.ndarray]":
    """Every pair ``a < b`` of ``p`` columns."""
    ia, ib = np.triu_indices(p, 1)
    return ia.astype(np.int64), ib.astype(np.int64)


def test_the_mask_separates_wide_pairs_from_the_pairs_that_fit(monkeypatch):
    """Eight 3-bin columns and two 48-bin columns against 10 classes: only the pairs of two wide columns (and the 3 x 48 ones above the budget) are set apart."""
    _fixed_budget(monkeypatch)
    nbins = np.array([3] * 8 + [48, 48], dtype=np.int32)
    pa, pb = _all_pairs(10)
    mask = B._shared_fit_mask(pa, pb, nbins, 10)
    joint = nbins[pa].astype(np.int64) * nbins[pb]
    assert mask is not None
    assert np.array_equal(mask, joint * 11 * 4 <= 40_000)
    assert int(mask.sum()) == int((joint <= 909).sum()) and int((~mask).sum()) >= 1


@pytest.mark.parametrize("nb", [3, 48])
def test_a_set_that_wholly_fits_or_wholly_does_not_is_left_alone(monkeypatch, nb):
    """No mixture, no split: ``None`` for all-narrow and for all-wide sets, and for sets of fewer than two pairs."""
    _fixed_budget(monkeypatch)
    pa, pb = _all_pairs(6)
    assert B._shared_fit_mask(pa, pb, np.full(6, nb, dtype=np.int32), 10) is None
    assert B._shared_fit_mask(pa[:1], pb[:1], np.full(6, nb, dtype=np.int32), 10) is None


@pytest.mark.skipif(not (B._CUDA_AVAIL and B._CUPY_AVAIL), reason="the routing of the device kernels needs CUDA and cupy")
@pytest.mark.parametrize("force_backend", [None, "cuda"])
def test_the_wide_pairs_alone_reach_the_row_chunked_kernel_and_every_mi_equals_the_cpu_kernel(monkeypatch, force_backend):
    """Value: the split result equals the CPU kernel pair by pair. Routing: the row-chunked kernel is asked for the wide pairs only, also when the FE step names the CUDA backend (it does)."""
    rng = np.random.default_rng(0)
    n, p, ky = 6000, 40, 10
    nbins = np.full(p, 3, dtype=np.int32)
    nbins[:3] = 48
    data = np.empty((n, p), dtype=np.int8)
    for j in range(p):
        data[:, j] = rng.integers(0, int(nbins[j]), n)
    y = rng.integers(0, ky, n).astype(np.int8)
    data[:, 0] = np.minimum(data[:, 0] + y * 4, 47)
    freqs = np.bincount(y, minlength=ky).astype(np.float64) / n
    pa, pb = _all_pairs(p)
    joint = nbins[pa].astype(np.int64) * nbins[pb]
    real = B.shared_fused_kernel_fits_budget
    wide = int((real(1, ky) > 0) and sum(1 for j in joint if real(int(j), ky) == 0))
    assert 0 < wide < pa.shape[0], "the device must fit the narrow pairs and not all the wide ones for this check"
    seen: "list[int]" = []
    original = B.batch_pair_mi_cuda_row_chunked

    def spy(factors_data, pair_a, *rest):
        """Record the number of pairs the slow kernel is asked for."""
        seen.append(int(pair_a.shape[0]))
        return original(factors_data, pair_a, *rest)

    monkeypatch.setattr(B, "batch_pair_mi_cuda_row_chunked", spy)
    got, _tag = B.dispatch_batch_pair_mi(data, pa, pb, nbins, y, freqs, force_backend)
    cpu = B.batch_pair_mi_njit_prange(data, pa, pb, nbins, y, freqs)
    assert np.allclose(got, cpu, rtol=0, atol=1e-12)
    assert sum(seen) <= wide, (seen, wide)
