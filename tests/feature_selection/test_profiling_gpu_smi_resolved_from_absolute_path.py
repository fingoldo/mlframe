"""The GPU-utilisation probe in the profiling tools never runs ``nvidia-smi`` by its bare name.

A bare name lets Windows run a planted ``nvidia-smi.exe`` from the working directory. With no absolute ``nvidia-smi`` on PATH the probe
must return NaN without starting any process.
"""

from __future__ import annotations

import math
import os

from mlframe.feature_selection._benchmarks.profiling import utilization


def test_gpu_busy_probe_never_starts_a_bare_nvidia_smi(monkeypatch, tmp_path):
    """With only a relative PATH entry, the probe reports NaN and every process it starts uses an absolute executable path."""
    started = []

    def _record(argv, *args, **kwargs):
        """Record the executable a subprocess call would start."""
        started.append(argv[0])
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(utilization.subprocess, "run", _record)
    monkeypatch.setenv("PATH", os.pathsep.join(["relative_bin_dir"]))
    monkeypatch.chdir(tmp_path)

    result = utilization._gpu_busy_percent()

    assert math.isnan(result)
    assert not [exe for exe in started if not os.path.isabs(exe)], started  # no process may start by a bare name (none starting at all is fine)
