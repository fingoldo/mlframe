"""A test must not assert on a wall-clock ratio or ceiling derived from one measurement per side.

The check is the shared ``py_ci_shared.single_shot_timing_assertion`` gate, with no baseline: the tree has no finding, so any new one fails.
A hang ceiling that only asserts the call returns carries ``@pytest.mark.hang_guard``; a deliberate timing race that runs outside the default
claim carries ``@pytest.mark.perf``; every other duration comparison is best-of-N per side (``perf_speedup_floor``/``perf_time_budget`` in
``tests/conftest.py`` scale a bound to the host).
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.single_shot_timing_assertion import assert_single_shot_timing_assertion

TESTS_DIR = Path(__file__).resolve().parents[1]


def test_tests_time_best_of_n():
    """No test asserts on a duration, or a ratio of durations, that it measured once."""
    assert_single_shot_timing_assertion(TESTS_DIR, min_files=3000)
