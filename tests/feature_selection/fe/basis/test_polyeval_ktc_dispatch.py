"""The default polyeval path takes the GPU when the kernel_tuning_cache crossover says so, and stays on the CPU below it or when the cache holds the no-GPU sentinel."""

import numpy as np
import pytest

import mlframe.feature_selection.filters.hermite_fe as hf


@pytest.fixture
def cuda_route(monkeypatch):
    """Pretend a GPU exists and record every call the dispatcher makes to the CUDA evaluator; the CPU evaluator is left real."""
    calls = []

    def fake_cuda(basis, x, c, device=None):
        """Record the call and return a marker array of the right length."""
        calls.append(len(x))
        return np.zeros(len(x))

    monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
    monkeypatch.setattr(hf, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(hf, "_polyeval_cuda_pick_devices", lambda n: [None])
    monkeypatch.setattr(hf, "_polyeval_cuda", fake_cuda)
    return calls


def _set_cuda_threshold(monkeypatch, threshold):
    """Make the kernel_tuning_cache lookup return the given CUDA crossover for every size."""
    monkeypatch.setattr(hf, "_lookup_polyeval_thresholds", lambda basis, n: (10**9, threshold))


def test_default_path_takes_gpu_at_and_above_the_cached_crossover(monkeypatch, cuda_route):
    """With no env override, a column at or above the cached CUDA crossover is evaluated on the device (finding for the polyeval dispatch gap)."""
    _set_cuda_threshold(monkeypatch, 1000)
    x = np.linspace(-1, 1, 1000)
    c = np.array([1.0, 0.5, 0.25])
    hf.polyeval_dispatch("hermite", x, c)
    assert cuda_route == [1000]


def test_default_path_stays_on_cpu_below_the_cached_crossover(monkeypatch, cuda_route):
    """Below the cached crossover the default path never touches the device."""
    _set_cuda_threshold(monkeypatch, 1000)
    x = np.linspace(-1, 1, 999)
    hf.polyeval_dispatch("hermite", x, np.array([1.0, 0.5]))
    assert cuda_route == []


def test_sentinel_crossover_above_the_swept_range_never_routes_to_gpu(monkeypatch, cuda_route):
    """A host where the round trip never pays stores a sentinel above the swept range; the default path must then stay on the CPU at any realistic size."""
    _set_cuda_threshold(monkeypatch, 10**12)
    x = np.linspace(-1, 1, 5000)
    hf.polyeval_dispatch("hermite", x, np.array([1.0, 0.5]))
    assert cuda_route == []


def test_forced_cpu_backend_ignores_the_gpu_crossover(monkeypatch, cuda_route):
    """``MLFRAME_POLYEVAL_BACKEND=njit`` keeps the CPU path even above the crossover."""
    _set_cuda_threshold(monkeypatch, 10)
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit")
    hf.polyeval_dispatch("hermite", np.linspace(-1, 1, 5000), np.array([1.0, 0.5]))
    assert cuda_route == []
