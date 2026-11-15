"""The pair search consults the kernel-tuning entry before taking the cupy route: with the host chosen it runs ``random_batch``, with the device chosen it runs the cupy search; the sweep callables agree on the top candidates."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _polynom_search_backend as pb


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def test_sweep_paths_select_the_same_candidates():
    """Host and device generation scorers put the same candidates at the top (the view the sweep compares)."""
    _needs_gpu()
    inputs = pb._sweep_inputs({"n_samples": 5000})
    np.testing.assert_array_equal(pb._sweep_host(*inputs), pb._sweep_device(*inputs))


def test_choice_falls_back_to_the_floor_without_a_spec(monkeypatch):
    """No tuner entry: the row floor decides."""
    monkeypatch.setattr(pb, "_SPEC", None)
    assert pb.polynom_device_chosen(pb.DEVICE_FALLBACK_ROWS) is True
    assert pb.polynom_device_chosen(pb.DEVICE_FALLBACK_ROWS - 1) is False


@pytest.mark.parametrize("device", [False, True])
def test_pair_search_route_follows_the_choice(monkeypatch, device):
    """``optimizer="cupy_kernel"`` runs the cupy search only when the tuner chooses the device; otherwise ``random_batch`` runs."""
    _needs_gpu()
    from mlframe.feature_selection.filters import _cupy_polynom_optimizer as cpo
    from mlframe.feature_selection.filters import _hermite_fe_optimise as ho
    from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair

    calls = {"cupy": 0, "random": 0}
    real_cupy, real_random = cpo.run_cupy_kernel_search, ho._run_random_batch_search

    def cupy_spy(**kw):
        """The cupy search, counting calls."""
        calls["cupy"] += 1
        return real_cupy(**kw)

    def random_spy(**kw):
        """The random_batch search, counting calls."""
        calls["random"] += 1
        return real_random(**kw)

    monkeypatch.setattr(cpo, "run_cupy_kernel_search", cupy_spy)
    monkeypatch.setattr(ho, "_run_random_batch_search", random_spy)
    monkeypatch.setattr(pb, "polynom_device_chosen", lambda n: device)  # device_search_preferred imports it at call time
    rng = np.random.default_rng(0)
    x_a = rng.standard_normal(2000)
    x_b = rng.standard_normal(2000)
    y = (x_a * x_b + 0.1 * rng.standard_normal(2000) > 0).astype(np.int64)
    optimise_hermite_pair(x_a, x_b, y, optimizer="cupy_kernel", n_trials=100, sweep_degrees=False, seed=0)
    assert (calls["cupy"] >= 1, calls["random"] >= 1) == ((True, False) if device else (False, True))
