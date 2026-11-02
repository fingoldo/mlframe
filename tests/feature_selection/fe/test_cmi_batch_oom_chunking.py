"""The batched CMI scorer keeps the whole-batch values when the device runs out of memory and has to score the candidates in chunks, and says so."""
import logging

import numpy as np
import pytest

import mlframe.feature_selection.filters._mi_greedy_cmi_fe as cmi_mod
from tests._cuda_require import require_cuda_device


def test_oom_retries_in_chunks_with_the_same_values(monkeypatch, caplog):
    """A batch wider than 20 columns 'runs out of memory': the scorer halves the chunk until it fits and returns what an unlimited device returns, with a warning."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _fe_batched_mi as B

    rng = np.random.default_rng(0)
    n, k = 4000, 64
    codes = rng.integers(0, 6, (n, k)).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    z = rng.integers(0, 5, n).astype(np.int64)
    require_cuda_device()
    want = B.batched_cmi_gpu(codes, y, z, codes_trusted=True)
    real = B.batched_cmi_gpu
    scored = []

    def limited(x, yy, zz, **kw):
        """The device 'holds' at most 20 columns."""
        if x.shape[1] > 20:
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 1 << 30)
        scored.append(int(x.shape[1]))
        return real(x, yy, zz, **kw)

    with caplog.at_level(logging.WARNING, logger=cmi_mod.logger.name):
        got = cmi_mod._batched_cmi_in_chunks(codes, y, z, limited)
    np.testing.assert_array_equal(got, want)
    assert sum(scored) == k and max(scored) <= 20
    assert any("ran out of device memory" in r.getMessage() for r in caplog.records)


def test_a_single_column_that_does_not_fit_still_raises():
    """With one column left to try there is nothing to halve: the error reaches the caller's CPU fallback."""
    cp = pytest.importorskip("cupy")
    codes = np.zeros((10, 3), dtype=np.int64)

    def never(x, yy, zz, **kw):
        """Always out of memory."""
        raise cp.cuda.memory.OutOfMemoryError(1 << 40, 1 << 30)

    with pytest.raises(cp.cuda.memory.OutOfMemoryError):
        cmi_mod._batched_cmi_in_chunks(codes, np.zeros(10, dtype=np.int64), None, never)


def test_second_call_starts_at_the_chunk_that_fitted(monkeypatch):
    """After a width has been halved to fit, the next batch of that width does not retry the whole batch."""
    cp = pytest.importorskip("cupy")
    monkeypatch.setattr(cmi_mod, "_CMI_CHUNK_START", {})
    monkeypatch.setattr(cmi_mod, "_CMI_CHUNK_BY_SHAPE", {})
    codes = np.zeros((10, 64), dtype=np.int64)
    widths = []

    def limited(x, yy, zz, **kw):
        """The device 'holds' at most 20 columns."""
        widths.append(int(x.shape[1]))
        if x.shape[1] > 20:
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 1 << 30)
        return np.zeros(x.shape[1])

    y = np.zeros(10, dtype=np.int64)
    cmi_mod._batched_cmi_in_chunks(codes, y, None, limited)
    widths.clear()
    cmi_mod._batched_cmi_in_chunks(codes, y, None, limited)
    assert max(widths) <= 20


def test_a_different_width_starts_at_the_chunk_that_fitted_for_the_same_rows(monkeypatch):
    """A batch of another width over the same rows and conditioning starts from the reduced chunk, not from the whole batch."""
    cp = pytest.importorskip("cupy")
    monkeypatch.setattr(cmi_mod, "_CMI_CHUNK_START", {})
    monkeypatch.setattr(cmi_mod, "_CMI_CHUNK_BY_SHAPE", {})
    widths = []

    def limited(x, yy, zz, **kw):
        """The device 'holds' at most 20 columns."""
        widths.append(int(x.shape[1]))
        if x.shape[1] > 20:
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 1 << 30)
        return np.zeros(x.shape[1])

    y = np.zeros(10, dtype=np.int64)
    cmi_mod._batched_cmi_in_chunks(np.zeros((10, 64), dtype=np.int64), y, None, limited)
    widths.clear()
    cmi_mod._batched_cmi_in_chunks(np.zeros((10, 95), dtype=np.int64), y, None, limited)
    assert max(widths) <= 20


def test_an_all_singleton_conditioning_support_scores_zero_without_a_launch():
    """With a distinct conditioning value in every row the conditional MI is 0 for every column (the kernel agrees to rounding noise), and the device is not asked."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu

    rng = np.random.default_rng(3)
    n, k = 800, 5
    codes = cp.asarray(rng.integers(0, 10, (n, k)).astype(np.int64))
    y = rng.integers(0, 2, n).astype(np.int64)
    z = rng.permutation(n).astype(np.int64)
    calls = []

    def spy(x, yy, zz, **kw):
        """Fail the test if the device is asked."""
        calls.append(1)
        return batched_cmi_gpu(x, yy, zz, **kw)

    got = cmi_mod._batched_cmi_in_chunks(codes, y, z, spy)
    assert not calls
    np.testing.assert_allclose(got, np.asarray(batched_cmi_gpu(codes, y, z, codes_trusted=True)), rtol=0, atol=1e-12)
    assert not got.any()
    z_shared = np.concatenate([z[: n // 2], z[: n // 2]])  # duplicates: the device scores it
    cmi_mod._batched_cmi_in_chunks(codes, y, z_shared, spy)
    assert calls
