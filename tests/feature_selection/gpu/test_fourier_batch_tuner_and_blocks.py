"""The Fourier batch detector: host-or-device by the kernel-tuning cache, and the device batch runs in column blocks."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_batch_gpu as B
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_batch as D


def test_a_tuner_entry_exists_and_the_fallback_follows_the_measured_crossover():
    """Below the crossover the host loop, from it the device batch."""
    assert D._SPEC is not None
    assert D._fallback_choice(2_000) == "host"
    assert D._fallback_choice(100_000) == "device"


def test_cmi_gpu_flag_forces_the_device_whatever_the_size(monkeypatch):
    """``MLFRAME_CMI_GPU=1`` stays an explicit override; otherwise the tuner (or its fallback) decides."""
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")
    assert D.fourier_device_chosen(100) is True
    monkeypatch.delenv("MLFRAME_CMI_GPU")
    monkeypatch.setattr(D, "_SPEC", None)
    assert D.fourier_device_chosen(2_000) is False
    assert D.fourier_device_chosen(50_000) is True


def test_the_batch_entry_asks_the_tuner_before_taking_the_device(monkeypatch):
    """With the strict flag on, a host verdict keeps the batch off the device; the lists are the host loop's."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    jobs = D._sweep_inputs({"n_samples": 6_000})[0]
    asked, ran = [], []
    monkeypatch.setattr(D, "fourier_device_chosen", lambda n: asked.append(n) or False)
    monkeypatch.setattr(B, "detect_fourier_freqs_batch_gpu", lambda *a, **k: ran.append(1))
    D.detect_fourier_freqs_batch(jobs, **D._SWEEP_PARAMS)
    assert asked == [6_000] and not ran


def test_forced_host_and_forced_device_find_the_same_lists():
    """The sweep's two callables agree on the lists, their order and their values."""
    pytest.importorskip("cupy")
    jobs = D._sweep_inputs({"n_samples": 20_000})[0]
    host, device = D._sweep_host(jobs), D._sweep_device(jobs)
    assert host.any() and np.allclose(host, device, rtol=1e-6, atol=1e-6)


def test_column_blocks_give_the_lists_of_one_block(monkeypatch):
    """Width 1, 3 and the whole group return the same lists, and the narrow width really launches one block per column."""
    pytest.importorskip("cupy")
    jobs = D._sweep_inputs({"n_samples": 8_000})[0]
    whole = D._sweep_device(jobs)
    blocks = []
    real = B._run_member_block
    monkeypatch.setattr(B, "_run_member_block", lambda cp, members, *a: blocks.append(len(members)) or real(cp, members, *a))
    for width in (1, 3):
        blocks.clear()
        monkeypatch.setattr(B, "_BLOCK_COLS", width)
        assert np.array_equal(D._sweep_device(jobs), whole)
        assert max(blocks) == width and sum(blocks) == len(jobs)


def test_the_block_width_halves_until_the_working_set_fits(monkeypatch):
    """With room for 5 columns of 1000 rows the width is the largest power of two that fits: 4."""
    import mlframe.feature_selection.filters._fe_gpu_vram as V

    monkeypatch.setattr(B, "_BLOCK_COLS", None)
    monkeypatch.setattr(V, "fe_gpu_has_vram_cushion", lambda nbytes: nbytes <= 5 * 1000 * 8 * 22)
    assert B._column_block_width(1000, 9) == 4
    assert B._column_block_width(1000, 3) == 3
