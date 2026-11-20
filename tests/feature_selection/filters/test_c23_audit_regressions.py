"""Regression tests of the chunk-23 audit: kernel input guards, serialised big-stack threads, uint64 dense codes and silent device fallbacks that now log."""

from __future__ import annotations

import logging
import sys
import threading
import types

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


def test_jmim_score_rejects_short_selected_column() -> None:
    """A selected column shorter than the candidate must raise instead of being read past its end by the kernel."""
    from mlframe.feature_selection.filters._jmim_scorer import jmim_score

    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 100).astype(np.int64)
    y = rng.integers(0, 2, 100).astype(np.int64)
    short = rng.integers(0, 3, 40).astype(np.int64)
    with pytest.raises(ValueError, match="same number of rows"):
        jmim_score(x, [short], y, 3, [3], 2)


def test_jmim_score_rejects_mismatched_bin_count_list() -> None:
    """The bin counts of the selected columns must match the columns."""
    from mlframe.feature_selection.filters._jmim_scorer import jmim_score

    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 100).astype(np.int64)
    y = rng.integers(0, 2, 100).astype(np.int64)
    with pytest.raises(ValueError, match="bin counts"):
        jmim_score(x, [x.copy(), x.copy()], y, 3, [3], 2)


def test_ii_null_floor_rejects_out_of_range_target_code() -> None:
    """A target code at or above the class count would index the counting kernel's table out of its row."""
    from mlframe.feature_selection.filters._interaction_information import pooled_pair_ii_null_floor

    rng = np.random.default_rng(0)
    n = 200
    data = rng.integers(0, 3, size=(n, 4)).astype(np.int32)
    nbins = np.array([5, 5, 5, 5])
    y = rng.integers(0, 2, n).astype(np.int64)
    y[0] = 2
    with pytest.raises(ValueError, match="target codes"):
        pooled_pair_ii_null_floor(
            data, nbins, np.array([0, 1]), np.array([2, 3]), np.zeros(2), np.zeros(2), y, np.array([100, 100]), n_permutations=3, random_seed=0
        )


def test_ii_null_floor_rejects_out_of_range_column_code() -> None:
    """A column code at or above its bin count must raise."""
    from mlframe.feature_selection.filters._interaction_information import pooled_pair_ii_null_floor

    rng = np.random.default_rng(0)
    n = 200
    data = rng.integers(0, 3, size=(n, 4)).astype(np.int32)
    data[3, 1] = 4
    nbins = np.array([5, 4, 5, 5])
    y = rng.integers(0, 2, n).astype(np.int64)
    with pytest.raises(ValueError, match="outside"):
        pooled_pair_ii_null_floor(
            data, nbins, np.array([0, 1]), np.array([2, 3]), np.zeros(2), np.zeros(2), y, np.array([100, 100]), n_permutations=3, random_seed=0
        )


def test_big_stack_threads_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two callers of run_in_big_stack_thread must overlap: the stack-size lock covers the thread start only, not its whole run."""
    from mlframe.feature_selection.filters import _joblib_safe

    monkeypatch.setattr(_joblib_safe, "_NEEDS_BIG_STACK", True)
    barrier = threading.Barrier(2, timeout=5)
    results: list = []

    def work() -> int:
        """Waits for the other caller: only possible when both run at once."""
        barrier.wait()
        return 1

    def caller() -> None:
        """One concurrent caller."""
        try:
            results.append(_joblib_safe.run_in_big_stack_thread(work))
        except threading.BrokenBarrierError:
            results.append(None)

    threads = [threading.Thread(target=caller) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [1, 1]


def test_unique_inverse_dense_and_dense_codes_handle_uint64_above_int64() -> None:
    """uint64 codes above the int64 range with a small span must not overflow the int64 shift."""
    from mlframe.feature_selection.filters._internals import dense_codes, unique_inverse_dense

    base = np.uint64(2**63)
    v = np.array([base + np.uint64(5), base + np.uint64(3), base + np.uint64(5), base + np.uint64(9)], dtype=np.uint64)
    uniq, inv = unique_inverse_dense(v)
    assert uniq.tolist() == [int(base) + 3, int(base) + 5, int(base) + 9]
    assert inv.tolist() == [1, 0, 1, 2]
    assert dense_codes(v).tolist() == [1, 0, 1, 2]


def test_njit_functions_dict_logs_wrap_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A registry function that cannot be wrapped stays pure Python, and that is logged at WARNING."""
    from mlframe.feature_selection.filters import _internals

    def boom(*_a, **_k):
        """njit stand-in that fails."""
        raise RuntimeError("no jit")

    monkeypatch.setattr(_internals, "njit", boom)
    reset_throttle_counts("_internals.njit_wrap_failed")
    d = {"f": lambda x: x + 1}
    with caplog.at_level(logging.WARNING, logger=_internals.logger.name):
        _internals.njit_functions_dict(d)
    assert any("njit-wrapping" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_fit_constant_memmap_failure_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A failed memmap dump returns the array and says so at WARNING."""
    from mlframe.feature_selection.filters import _joblib_safe

    def boom(_arr):
        """Content-key stand-in that fails."""
        raise OSError("disk full")

    monkeypatch.setattr(_joblib_safe, "_fit_constant_key", boom)
    reset_throttle_counts("_joblib_safe.fit_constant_memmap_failed")
    arr = np.arange(10)
    with caplog.at_level(logging.WARNING, logger=_joblib_safe.logger.name):
        assert _joblib_safe.fit_constant_memmap(arr) is arr
    assert any("memmap dump/cache failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_oj_score_block_device_failures_log_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A device fault in the winner pick or the pool step falls to the CPU kernels with a WARNING."""
    from mlframe.feature_selection.filters import _honest_oj_fused as mod
    from mlframe.feature_selection.filters import _honest_oj_fused_gpu as gpu

    class BadWinners:
        """Device winners that fail."""

        def picks(self, perms):
            """Always fails."""
            raise RuntimeError("winners fault")

    class GoodWinnersBadPool(BadWinners):
        """Winners that work next to a pool that fails."""

        def picks(self, perms):
            """Returns dummy picks."""
            return np.zeros((perms.shape[0], 1), dtype=np.int64)

    class BadPool:
        """Device pool that fails."""

        def maxima(self, *a):
            """Always fails."""
            raise RuntimeError("pool fault")

    monkeypatch.setattr(mod, "_fused_block", lambda *a: None)
    monkeypatch.setattr(gpu, "_pool_block", lambda *a: None)
    reset_throttle_counts("_honest_oj_fused.device_winners_failed")
    reset_throttle_counts("_honest_oj_fused.device_pool_failed")
    perms = np.zeros((2, 3), dtype=np.int64)
    pool_args = (None, None, None, None, 1.0)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod._score_block(BadWinners(), None, perms, None, None, None, None, None, None, None, None, pool_args)
        mod._score_block(GoodWinnersBadPool(), BadPool(), perms, None, None, None, None, None, None, None, None, pool_args)
    text = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "device winners failed" in text and "device pool failed" in text


def _fake_cupy(free: int) -> types.ModuleType:
    """A stand-in cupy module whose free device memory is ``free`` bytes."""
    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(memGetInfo=lambda: (free, 10 * free)))  # type: ignore[attr-defined]
    return fake


def test_oj_device_winners_vram_gate_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The VRAM gate that keeps the winners on the CPU says so at WARNING with the byte counts."""
    from mlframe.feature_selection.filters import _gpu_policy, _scorer_device_dispatch
    from mlframe.feature_selection.filters import _honest_oj_fused_gpu as gpu

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(10))
    monkeypatch.setattr(_gpu_policy, "cuda_available_for_run", lambda: True)
    monkeypatch.setattr(_scorer_device_dispatch, "scorer_device_choice", lambda *a, **k: True)
    reset_throttle_counts("_honest_oj_fused_gpu.winners_vram")
    xb = np.zeros((1, 1, 1, 64), dtype=np.int8)
    kx = np.ones((1, 1, 1), dtype=np.int64)
    with caplog.at_level(logging.WARNING, logger=gpu.logger.name):
        out = gpu.device_winners(xb, kx, np.array([64]), np.zeros((1, 64), dtype=np.int64), np.zeros(64), np.zeros((1, 1, 1, 1)), 64)
    assert out is None
    assert any("winners stay on the CPU" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_oj_device_pool_vram_gate_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The VRAM gate that keeps the pool on the CPU says so at WARNING with the byte counts."""
    from mlframe.feature_selection.filters import _honest_oj_fused_gpu as gpu

    dev = types.SimpleNamespace(cp=_fake_cupy(10))
    reset_throttle_counts("_honest_oj_fused_gpu.pool_vram")
    yc = np.zeros(64, dtype=np.int64)
    yc[:5] = 1
    with caplog.at_level(logging.WARNING, logger=gpu.logger.name):
        out = gpu.device_pool(dev, np.zeros(1, dtype=np.int64), np.zeros((1, 1, 64), dtype=np.int8), np.ones((1, 1), dtype=np.int64), np.ones((1, 1)), yc, np.arange(64))  # type: ignore[arg-type]
    assert out is None
    assert any("pool stays on the CPU" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_honest_null_device_prep_decline_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Too many bins for the device kernels hands the call to the host path with a WARNING."""
    from mlframe.feature_selection.filters import _honest_null_device as mod

    reset_throttle_counts("_honest_null_device.prep_declined")
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        assert mod._prep_declined({"nx": np.array([5, 40])}) is True
        assert mod._prep_declined({"nx": np.array([5, 8]), "cls": object()}) is False
    assert any("device honest null not used" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_presort_gpu_nonfinite_and_missing_cupy_log_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The host-sort fallbacks of the device presort (non-finite input, no cupy) are logged at WARNING."""
    from mlframe.feature_selection.filters import _honest_null_presort_gpu as mod

    reset_throttle_counts("_honest_null_presort_gpu.not_finite")
    reset_throttle_counts("_honest_null_presort_gpu.cupy_unavailable")
    bad = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    good = np.arange(12, dtype=np.float64).reshape(6, 2)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        assert mod.presort_and_ties_gpu(bad) is None
        monkeypatch.setitem(sys.modules, "cupy", None)
        assert mod.presort_and_ties_gpu(good) is None
    text = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "not all finite" in text and "device presort unavailable" in text
