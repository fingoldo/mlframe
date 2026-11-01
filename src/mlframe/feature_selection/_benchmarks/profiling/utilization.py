"""Idle share of the CPU cores and of the GPU during one every-family MRMR fit.

The target of the GPU work is a fit in which neither the cores nor the card wait: this tool reports how far a build is from it. It runs one fit of the ``family_scale`` frame (every ``fe_*_enable`` family except the
ones in ``ALL_EXCLUDED``) and samples, twice a second, the CPU load of every logical core (``psutil``) and the GPU utilisation (``nvidia-smi``). Output: the mean busy share of the cores and of the card, the share
of samples in which the card was below 10% and in which fewer than a quarter of the cores were busy, and a timeline in windows of ``--window`` seconds.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.utilization [n] [p]`` (defaults 100000 100).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import warnings

import numpy as np


def _gpu_busy_percent() -> float:
    """GPU utilisation (percent) of device 0 from ``nvidia-smi``, NaN when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def sample_during(fn, interval: float = 0.5) -> "tuple[object, np.ndarray, np.ndarray, np.ndarray]":
    """Run ``fn()`` while sampling; returns ``(result, times, cpu_busy_percent_per_core (samples x cores), gpu_busy_percent)``."""
    import psutil

    times: list = []
    cpu: list = []
    gpu: list = []
    stop = threading.Event()

    def _sampler() -> None:
        """Sample until stopped."""
        psutil.cpu_percent(None, percpu=True)
        t0 = time.perf_counter()
        while not stop.wait(interval):
            times.append(time.perf_counter() - t0)
            cpu.append(psutil.cpu_percent(None, percpu=True))
            gpu.append(_gpu_busy_percent())

    thread = threading.Thread(target=_sampler, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join(timeout=10)
    return result, np.asarray(times), np.asarray(cpu, dtype=float), np.asarray(gpu, dtype=float)


def summarise(times: np.ndarray, cpu: np.ndarray, gpu: np.ndarray, window: float = 20.0) -> str:
    """Mean busy shares, idle fractions and a windowed timeline as text."""
    cores = cpu.shape[1]
    cpu_total = cpu.mean(axis=1)
    lines = [
        f"samples {len(times)} over {times[-1]:.0f} s, {cores} logical cores",
        f"CPU  busy {cpu_total.mean():5.1f}% of all cores on average; samples with under 25% of the cores busy: {(cpu_total < 25).mean() * 100:4.0f}%",
        f"GPU  busy {np.nanmean(gpu):5.1f}% on average; samples with the card under 10%: {(gpu < 10).mean() * 100:4.0f}%",
        f"both idle together (CPU < 25% and GPU < 10%): {((cpu_total < 25) & (gpu < 10)).mean() * 100:4.0f}% of the samples",
        "timeline (window start s : CPU % / GPU %)",
    ]
    for w0 in np.arange(0.0, times[-1], window):
        m = (times >= w0) & (times < w0 + window)
        if m.any():
            lines.append(f"  {w0:5.0f}: {cpu_total[m].mean():5.1f} / {np.nanmean(gpu[m]):5.1f}")
    return "\n".join(lines)


def main() -> None:
    """Fit once under the sampler and print the summary."""
    warnings.simplefilter("ignore")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
        os.environ.setdefault(k, v)
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20_000, 30, 1))  # warm-up: compilation stays out of the measured fit
    data = F._make(n, p, 0)
    MRMR._FIT_CACHE.clear()
    t0 = time.perf_counter()
    _, times, cpu, gpu = sample_during(lambda: MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data))
    print(f"fit wall {time.perf_counter() - t0:.1f} s")
    print(summarise(times, cpu, gpu))


if __name__ == "__main__":
    main()
