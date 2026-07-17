"""Regression: ``compute_area_profits`` indexed ``positions[end + 1]`` against ``prices.shape[0]`` instead of ``positions.shape[0]``.

Under ``@numba.njit`` bounds-checks are relaxed and the out-of-range read silently terminated the inner ``while``; under ``NUMBA_DISABLE_JIT=1`` (the nightly coverage profile) Python's strict bounds check raised ``IndexError: index N out of bounds for axis 0 with size N`` at the wrapped call site (``find_best_mps_sequence`` -> ``compute_area_profits``).
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys

import numpy as np


def test_compute_area_profits_no_oob_when_positions_shorter_than_prices():
    """Direct sensor: positions length n-1, prices length n -- accessing ``positions[end+1]`` while bounded by ``n`` is the bug."""
    from mlframe.feature_engineering.mps import compute_area_profits

    prices = np.array([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0], dtype=np.float64)
    positions = np.array([1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
    out = compute_area_profits(prices, positions)
    assert out.shape[0] >= positions.shape[0]
    assert np.all(np.isfinite(out))


def test_find_maximum_profit_system_returns_under_no_jit():
    """End-to-end sensor: must not raise IndexError when NUMBA_DISABLE_JIT=1 (or under JIT). Reuses the W10D-surfacing fixture."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system

    prices = np.array([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0])
    r = find_maximum_profit_system(prices, tc=0.0)
    assert list(r["positions"]) == [1, 1, 1, -1, -1, -1, -1]


def test_find_maximum_profit_system_under_disabled_jit_subprocess():
    """Subprocess sensor that runs the call under NUMBA_DISABLE_JIT=1; isolates the no-JIT path even when the main pytest process has already cached numba bindings."""
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    code = (
        "import numpy as np\n"
        "from mlframe.feature_engineering.mps import find_maximum_profit_system\n"
        "prices = np.array([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0])\n"
        "r = find_maximum_profit_system(prices, tc=0.0)\n"
        "assert list(r['positions']) == [1, 1, 1, -1, -1, -1, -1]\n"
    )
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=60)  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
    assert res.returncode == 0, f"subprocess failed: stdout={res.stdout!r} stderr={res.stderr!r}"
