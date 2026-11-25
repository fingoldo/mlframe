"""Cluster 13 regression: the utilisation summary survives a fit shorter than the sample interval (audit 2026-11-20, finding 13-F33).

With no samples, summarise used to index times[-1] and crash. It now returns an explicit line; a populated summary still reports.
"""

import numpy as np

from mlframe.feature_selection._benchmarks.profiling.utilization import summarise


def test_no_samples_returns_a_plain_line():
    """Empty sample arrays give a stated message and no exception (13-F33)."""
    out = summarise(np.array([]), np.zeros((0, 4)), np.array([]))
    assert out.startswith("no utilisation samples")


def test_populated_samples_still_summarise():
    """Control: samples produce the CPU and GPU summary lines (13-F33)."""
    times = np.array([0.0, 0.5, 1.0])
    cpu = np.full((3, 4), 50.0)
    gpu = np.array([20.0, 30.0, 40.0])
    out = summarise(times, cpu, gpu)
    assert out.startswith("samples 3 over 1 s, 4 logical cores")
