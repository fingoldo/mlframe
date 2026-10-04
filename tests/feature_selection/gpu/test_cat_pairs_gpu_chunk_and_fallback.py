"""cat-FE GPU pair search: host-side chunking above the gridDim.y limit and the CPU fallback under ``backend="auto"``."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters._cat_interactions_step import _gpu_pair_search_or_none
from mlframe.feature_selection.filters.info_theory._batch_kernels import batch_pair_mi_prange


def _cuda_ok() -> bool:
    """True when cupy sees a device with at least ~300 MB free."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
        return bool(free > 300 * 1024**2)
    except Exception:
        return False


def _fixture(n=300, n_cols=6, nbins=3, seed=5):
    """Small discrete frame, a binary target and its marginal frequencies."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, nbins, size=(n, n_cols)).astype(np.int32)
    y = ((data[:, 0] + data[:, 1] + rng.integers(0, 2, n)) % 2).astype(np.int32)
    freqs_y = np.bincount(y, minlength=2).astype(np.float64) / n
    nb = np.full(n_cols, nbins, dtype=np.int32)
    return data, nb, y, freqs_y


def test_auto_backend_falls_back_to_cpu_when_gpu_pair_search_raises():
    """Under ``backend='auto'`` a GPU failure (here the old >65535-pairs ValueError) returns the CPU signal instead of escaping the fit."""

    def boom(**_kwargs):
        """Stand-in for the GPU kernel entry point."""
        raise ValueError("n_pairs exceeds the CUDA gridDim.y limit")

    data, nb, y, freqs_y = _fixture()
    pa, pb = np.array([0, 1]), np.array([2, 3])
    out = _gpu_pair_search_or_none(SimpleNamespace(backend="auto"), boom, data, pa, pb, nb, y, freqs_y, np.int32, np.zeros(6))
    assert out == (False, None, None, None)


def test_explicit_gpu_backend_still_raises():
    """An explicit ``backend='gpu'`` request surfaces the failure instead of silently running on the CPU."""

    def boom(**_kwargs):
        """Stand-in for the GPU kernel entry point."""
        raise RuntimeError("no device")

    data, nb, y, freqs_y = _fixture()
    with pytest.raises(RuntimeError):
        _gpu_pair_search_or_none(SimpleNamespace(backend="gpu"), boom, data, np.array([0]), np.array([1]), nb, y, freqs_y, np.int32, np.zeros(6))


def test_successful_gpu_pair_search_reports_interaction_information():
    """On success the helper returns joint MI, interaction information (joint - marginals) and the pair cardinalities."""
    data, nb, y, freqs_y = _fixture()
    pa, pb = np.array([0, 2]), np.array([1, 3])
    joint = np.array([0.5, 0.25])
    marginal = np.array([0.1, 0.05, 0.02, 0.01, 0.0, 0.0])
    used, ii, joint_out, n_uniq = _gpu_pair_search_or_none(
        SimpleNamespace(backend="auto"), lambda **_k: joint, data, pa, pb, nb, y, freqs_y, np.int32, marginal,
    )
    assert used is True
    np.testing.assert_allclose(ii, [0.5 - 0.1 - 0.05, 0.25 - 0.02 - 0.01])
    np.testing.assert_array_equal(joint_out, joint)
    np.testing.assert_array_equal(n_uniq, [9, 9])


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_gpu_pairs_chunks_past_grid_y_limit_and_matches_cpu():
    """70000 pairs (above the 65535 gridDim.y limit) are scored in host-side chunks and match the CPU kernel."""
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs

    data, nb, y, freqs_y = _fixture()
    base_a, base_b = np.triu_indices(6, k=1)
    reps = 70_000 // len(base_a) + 1
    pa = np.tile(base_a, reps)[:70_000].astype(np.int64)
    pb = np.tile(base_b, reps)[:70_000].astype(np.int64)
    got = mi_direct_gpu_batched_pairs(data, pa, pb, nb, y, freqs_y)
    ref = batch_pair_mi_prange(data, pa, pb, nb, y, freqs_y)
    assert got.shape == (70_000,)
    np.testing.assert_allclose(got, ref, atol=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_gpu_pairs_small_chunk_limit_gives_identical_result(monkeypatch):
    """Forcing a tiny per-launch pair limit changes nothing but the number of launches."""
    from mlframe.feature_selection.filters import _gpu_pairs
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs

    data, nb, y, freqs_y = _fixture()
    pa, pb = np.triu_indices(6, k=1)
    whole = mi_direct_gpu_batched_pairs(data, pa, pb, nb, y, freqs_y)
    monkeypatch.setattr(_gpu_pairs, "_MAX_PAIRS_PER_LAUNCH", 4)
    chunked = mi_direct_gpu_batched_pairs(data, pa, pb, nb, y, freqs_y)
    np.testing.assert_array_equal(whole, chunked)
