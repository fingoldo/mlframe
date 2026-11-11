"""The device sort order and tie flags of the honest null equal the host's."""

import numpy as np
import pytest


def _cupy_or_skip():
    """cupy, or a skip without a usable CUDA device."""
    cp = pytest.importorskip("cupy")
    try:  # audit-exempt: except_skip_masks_call_under_test -- the guarded call only asks the runtime for a CUDA device; nothing under test runs in the try
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    return cp


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,k", [(997, 9), (20011, 24), (50000, 12)])
def test_order_and_ties_equal_the_host(dtype, n, k):
    """Continuous, heavily tied, binary and constant columns: the same permutation (ties in the stable order) and the same flags."""
    _cupy_or_skip()
    from mlframe.feature_selection.filters._honest_null_device import _tie_flags
    from mlframe.feature_selection.filters._honest_null_presort_gpu import presort_and_ties_gpu
    from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns

    rng = np.random.default_rng(n + k)
    raw = rng.normal(size=(n, k)).astype(dtype)
    raw[:, ::3] = np.round(raw[:, ::3] * 12)
    raw[:, 1] = 2.0
    raw[:, 2] = (rng.random(n) < 0.3).astype(dtype)
    order_host = _presort_columns(raw)
    tie_host = _tie_flags(raw, order_host)
    got = presort_and_ties_gpu(raw)
    assert got is not None
    assert np.array_equal(got[0], order_host, equal_nan=True)
    assert np.array_equal(got[1], tie_host, equal_nan=True)


def test_non_finite_columns_stay_on_the_host():
    """A NaN or inf in the block makes the device path decline (the host sort handles it)."""
    _cupy_or_skip()
    from mlframe.feature_selection.filters._honest_null_presort_gpu import presort_and_ties_gpu

    raw = np.random.default_rng(0).normal(size=(100, 4)).astype(np.float32)
    raw[3, 2] = np.nan
    assert presort_and_ties_gpu(raw) is None
    raw[3, 2] = np.inf
    assert presort_and_ties_gpu(raw) is None


def test_an_out_of_memory_block_is_halved_and_retried_not_given_up(monkeypatch):
    """Concurrent stages can take the room a block was sized for: the first allocation fails, the block is halved, and the result still equals the host's (it was ``None`` before, a whole sort on the host)."""
    cp = _cupy_or_skip()
    import mlframe.feature_selection.filters._honest_null_presort_gpu as presort
    from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5000, 40)).astype(np.float64)
    real_argsort = presort.device_argsort
    calls = {"n": 0, "widths": []}

    def flaky(cols, axis):
        """The first call fails as a full pool does; later calls sort for real."""
        calls["n"] += 1
        calls["widths"].append(int(cols.shape[0]))
        if calls["n"] == 1:
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 1 << 20)
        return real_argsort(cols, axis=axis)

    monkeypatch.setattr(presort, "device_argsort", flaky)
    monkeypatch.setattr(presort, "_presort_column_block", lambda _cp, _n: 32)
    got = presort.presort_and_ties_gpu(raw)
    assert got is not None
    assert np.array_equal(got[0], _presort_columns(raw), equal_nan=True)
    assert calls["widths"][0] == 32 and calls["widths"][1] == 16, calls["widths"]


def test_an_oom_that_exhausts_the_halving_retry_notes_the_shared_cooldown(monkeypatch):
    """GPU11-13 follow-up: when the block-halving retry gives up (every attempt down to the floor OOMs), the
    final re-raised OutOfMemoryError must arm the shared cooldown before the function falls back to None."""
    cp = _cupy_or_skip()
    import mlframe.feature_selection.filters._honest_null_presort_gpu as presort
    import mlframe.feature_selection.filters._fe_gpu_vram as vram_mod

    vram_mod._reset_gpu_oom_cooldown()
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(5000, 40)).astype(np.float64)

    def always_oom(cols, axis):
        """Every attempt OOMs, regardless of block width."""
        raise cp.cuda.memory.OutOfMemoryError(1 << 30, 1 << 20)

    monkeypatch.setattr(presort, "device_argsort", always_oom)
    monkeypatch.setattr(presort, "_presort_column_block", lambda _cp, _n: 32)
    got = presort.presort_and_ties_gpu(raw)
    assert got is None
    assert vram_mod._in_oom_cooldown() is True
    vram_mod._reset_gpu_oom_cooldown()
