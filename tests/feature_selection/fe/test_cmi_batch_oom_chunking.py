"""The batched CMI scorer keeps the whole-batch values when the device runs out of memory and has to score the candidates in chunks, and says so."""
import logging

import numpy as np
import pytest

import mlframe.feature_selection.filters._mi_greedy_cmi_fe as cmi_mod


def test_oom_retries_in_chunks_with_the_same_values(monkeypatch, caplog):
    """A batch wider than 20 columns 'runs out of memory': the scorer halves the chunk until it fits and returns what an unlimited device returns, with a warning."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _fe_batched_mi as B

    rng = np.random.default_rng(0)
    n, k = 4000, 64
    codes = rng.integers(0, 6, (n, k)).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    z = rng.integers(0, 5, n).astype(np.int64)
    try:
        want = B.batched_cmi_gpu(codes, y, z, codes_trusted=True)
    except Exception as exc:
        pytest.skip(f"needs a CUDA device: {exc}")
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
