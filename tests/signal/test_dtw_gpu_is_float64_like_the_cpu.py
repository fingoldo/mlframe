"""The GPU DTW paths accumulate in float64, so they agree with the CPU path exactly even for series far from zero."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.signal import dtw


def _series(n=800):
    """Random-walk pair with a large offset, where float32 inputs and cost sums lose digits."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=n)) + 1000.0
    return x, x + rng.normal(0, 0.05, n)


def _need_cupy():
    """Skip when no CUDA device is usable."""
    if not dtw._HAS_CUPY:
        pytest.skip("no cupy")
    try:
        dtw.cp.zeros(1)
    except Exception as exc:
        pytest.skip(f"no CUDA device: {exc}")


@pytest.mark.parametrize("fn", ["dtw_cupy_banded", "dtw_cupy_full"])
def test_cupy_distance_and_path_equal_the_cpu_result(fn):
    """Distance equals the CPU float64 value and the warping path is identical."""
    _need_cupy()
    x, y = _series()
    d_cpu, p_cpu = dtw.dtw_cpu(x, y, window=100)
    d_gpu, p_gpu = getattr(dtw, fn)(x, y, window=100)
    assert d_gpu == pytest.approx(d_cpu, rel=1e-12, abs=0.0)
    assert p_gpu == p_cpu
