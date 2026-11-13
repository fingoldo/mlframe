"""The usability |corr| backend (CPU against the warp kernel) is chosen through the kernel-tuning cache."""
import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as M


def test_a_tuner_entry_is_registered_with_the_device_as_pre_sweep_default():
    """Without a swept region the choice is the device, the measured winner at every tested scale."""
    assert M._SPEC is not None
    assert M._fallback_choice(5000) == "device"


def test_sweep_inputs_have_the_documented_layout():
    """Operands are rows of the matrix, the pairs index them, and every form is evaluated."""
    y, operands, pair_a, pair_b, forms = M._sweep_inputs({"n_samples": 40})
    assert y.shape == (M._SWEEP_N_ROWS,) and operands.shape == (M._SWEEP_OPERANDS, M._SWEEP_N_ROWS)
    assert pair_a.shape == pair_b.shape == (40,) and int(max(pair_a.max(), pair_b.max())) < M._SWEEP_OPERANDS
    assert np.array_equal(forms, M.ALL_FORM_IDS)


def test_host_choice_keeps_the_cpu_backend(monkeypatch):
    """A tuner verdict of 'host' routes an unforced call to the CPU backend."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(400).astype(np.float32)
    ops = rng.standard_normal((6, 400)).astype(np.float32)
    a, b = np.array([0, 1, 2]), np.array([3, 4, 5])
    monkeypatch.setattr(M, "_device_wins", lambda n_pairs: False)
    _table, backend = M._dispatch_batch_pair_usability_corr_uncached(y, ops, a, b)
    assert backend == "cpu"


@pytest.mark.skipif(not M._CUDA_AVAIL, reason="no CUDA device")
def test_device_choice_runs_the_warp_backend(monkeypatch):
    """A tuner verdict of 'device' keeps the warp kernel for an unforced call."""
    rng = np.random.default_rng(2)
    y = rng.standard_normal(400).astype(np.float32)
    ops = rng.standard_normal((6, 400)).astype(np.float32)
    monkeypatch.setattr(M, "_device_wins", lambda n_pairs: True)
    _table, backend = M._dispatch_batch_pair_usability_corr_uncached(y, ops, np.array([0, 1]), np.array([2, 3]))
    assert backend == "cuda_warp"


@pytest.mark.skipif(not M._CUDA_AVAIL, reason="no CUDA device")
def test_the_registered_sweep_runs_and_the_backends_agree(monkeypatch):
    """The sweep callable the tuner entry runs completes on a reduced grid and the float32 device table agrees with the host's to 1e-6."""
    from mlframe.feature_selection.filters._ktc_fallback import host_device_sweep

    monkeypatch.setattr(M, "_SWEEP_ROWS", [200, 800])
    monkeypatch.setattr(M, "_SWEEP_N_ROWS", 800)
    regions = host_device_sweep(M.__name__, M._sweep_host, M._sweep_device, rtol=1e-6, atol=1e-6)()
    assert any(r.get("backend_choice") == "device" for r in regions), regions
