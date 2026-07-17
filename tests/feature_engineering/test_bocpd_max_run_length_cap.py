"""Tests for the BOCPD run-length cap (bayesian.bocpd_features max_run_length).

Pins two contracts:
1. On a short stream, a cap large enough to never bind leaves results bit-identical to the uncapped run.
2. The cap actually bounds the run-length posterior: capped expected/MAP run length never exceeds the cap,
   and a small cap on a long stable stream keeps memory/run-length bounded (where uncapped grows with T).
"""

import numpy as np
import pytest

from mlframe.feature_engineering.bayesian import bocpd_features


def _stream(seed, n):
    """Helper: Stream."""
    rng = np.random.default_rng(seed)
    # Two regimes to exercise change points, then a long stable tail so run length grows.
    a = rng.normal(0.0, 1.0, size=n // 2)
    b = rng.normal(5.0, 1.0, size=n - n // 2)
    return np.concatenate([a, b]).astype(np.float64)


@pytest.mark.parametrize("seed", [0, 2, 9])
def test_high_cap_matches_uncapped_short_stream(seed):
    """High cap matches uncapped short stream."""
    x = _stream(seed, 120)
    capped = bocpd_features(x, max_run_length=100000)  # >> stream length, never binds
    uncapped = bocpd_features(x, max_run_length=0)  # cap disabled
    for k in ("p_change", "expected_run_length", "max_run_length"):
        np.testing.assert_array_equal(capped[k], uncapped[k], err_msg=f"{k} diverged")


@pytest.mark.parametrize("cap", [10, 50, 200])
def test_cap_bounds_run_length(cap):
    """Cap bounds run length."""
    x = _stream(1, 2000)
    res = bocpd_features(x, max_run_length=cap)
    # The retained vector holds run lengths 0..cap (cap+1 slots), so MAP run length is bounded by cap --
    # without the cap it would grow toward T (=2000 here).
    assert np.nanmax(res["max_run_length"]) <= cap + 1e-9
    # Expected run length likewise bounded by the cap.
    assert np.nanmax(res["expected_run_length"]) <= cap + 1e-6


def test_default_cap_matches_uncapped_on_typical_stream():
    # Default cap (1000) must not change results vs uncapped on a realistic-length stream whose
    # run lengths stay well under 1000 (hazard default ~1/250).
    """Default cap matches uncapped on typical stream."""
    x = _stream(7, 800)
    default = bocpd_features(x)  # max_run_length=1000 default
    uncapped = bocpd_features(x, max_run_length=0)
    np.testing.assert_array_equal(default["p_change"], uncapped["p_change"])
