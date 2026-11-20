"""The parallel robust-mean path agrees with the serial one (audit 2026-11-20, finding 03-F3)."""

import numpy as np

import mlframe.core.robust_location as rl


def test_parallel_and_serial_robust_mean_agree(monkeypatch):
    """Both dispatch branches return the same location to float tolerance; only the reduction order differs."""
    rng = np.random.default_rng(7)
    x = rng.standard_t(3, 60_000)
    monkeypatch.setattr(rl, "_ROBUST_MEAN_PARALLEL_MIN_N", 10**12)
    serial = rl.robust_mean_mestimator(x)
    monkeypatch.setattr(rl, "_ROBUST_MEAN_PARALLEL_MIN_N", 0)
    parallel = rl.robust_mean_mestimator(x)
    assert np.isclose(serial, parallel, rtol=1e-7, atol=1e-9), (serial, parallel)
