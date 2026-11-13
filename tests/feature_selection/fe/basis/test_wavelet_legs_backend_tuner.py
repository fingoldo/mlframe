"""The wavelet leg selection chooses host or device through the kernel-tuning cache on the row count."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _wavelet_basis_fe as W
from mlframe.feature_selection.filters import _wavelet_legs_tuner as T


def test_a_tuner_entry_exists_and_the_fallback_follows_the_measured_crossover():
    """Below the crossover the host body, from it the device (the sweep on the dev card put it at 20000 rows)."""
    assert T._SPEC is not None
    assert T._fallback_choice(5_000) == "host"
    assert T._fallback_choice(100_000) == "device"


def test_cmi_gpu_flag_forces_the_device_whatever_the_size(monkeypatch):
    """``MLFRAME_CMI_GPU=1`` stays an explicit override; otherwise the tuner decides."""
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")
    assert T.wavelet_device_chosen(100) is True
    monkeypatch.delenv("MLFRAME_CMI_GPU")
    monkeypatch.setattr(T, "_SPEC", None)
    assert T.wavelet_device_chosen(5_000) is False
    assert T.wavelet_device_chosen(50_000) is True


def test_policy_asks_the_tuner_only_when_the_strict_flag_admits_the_device(monkeypatch):
    """Strict flag off: the host body without consulting the tuner. Flag on: the tuner's verdict picks the route."""
    rng = np.random.default_rng(0)
    x = rng.random(6000)
    y = ((x > 0.55) & (x < 0.7)).astype(int)
    lo, span = float(x.min()), float(x.max() - x.min())
    asked = []
    monkeypatch.setattr(T, "wavelet_device_chosen", lambda n: asked.append(n) or False)
    monkeypatch.setattr(W, "_binnedmi_gpu_enabled", lambda **k: False)
    W._select_wavelet_legs(x, y, lo, span)
    assert not asked
    monkeypatch.setattr(W, "_binnedmi_gpu_enabled", lambda **k: True)
    W._select_wavelet_legs(x, y, lo, span)
    assert asked == [6000]


def test_forced_host_and_forced_device_admit_the_same_legs():
    """The sweep's two callables agree on the admitted legs and their order."""
    pytest.importorskip("cupy")
    args = T._sweep_inputs({"n_samples": 20_000})
    assert np.array_equal(T._sweep_host(*args), T._sweep_device(*args))
    assert T._sweep_host(*args).any()
