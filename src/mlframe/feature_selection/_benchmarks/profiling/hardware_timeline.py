"""Sampler of the machine while an MRMR fit runs: CPU load per core, free RAM, and per GPU the utilisation and the free VRAM, one JSON line every ``interval`` seconds on the epoch clock.

The stage ledger (``mlframe.feature_selection.filters._stage_ledger``) records every stage on the same clock; ``stage_idle_report`` joins the two to answer, for each stage, whether a device with low load and enough
free memory stood idle before it started. Usage in a harness::

    from mlframe.feature_selection._benchmarks.profiling.hardware_timeline import TimelineSampler
    with TimelineSampler("timeline.jsonl"):
        fit()

The sample fields: ``t`` (epoch seconds), ``cpu`` (mean of all logical cores, percent), ``cores`` (percent per logical core), ``ram_free_gb``, ``gpus`` (one dict per device: ``util`` percent, ``mem_free_gb``,
``mem_used_gb``, ``temp_c``). NVML is used when importable, ``nvidia-smi`` otherwise; a machine without a GPU just has an empty ``gpus`` list. The sampler thread is a daemon and costs about one
``psutil`` call per tick (the NVML query is a few hundred microseconds)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Optional

_GB = 2**30


def _nvml_gpus() -> "Optional[list[dict[str, Any]]]":
    """Per-device readings through NVML, or ``None`` when NVML is not usable here."""
    try:
        import pynvml

        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out.append(
                {
                    "index": i,
                    "util": float(util.gpu),
                    "mem_free_gb": mem.free / _GB,
                    "mem_used_gb": mem.used / _GB,
                    "temp_c": float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                }
            )
        return out
    except Exception:
        return None


def _smi_gpus() -> "list[dict[str, Any]]":
    """Per-device readings through ``nvidia-smi`` (the fallback), empty when it cannot be run."""
    try:
        from mlframe.training._gpu_state_probe import _resolve_smi

        text = subprocess.run(
            [str(_resolve_smi()), "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        out = []
        for line in text.strip().splitlines():
            idx, util, used, total, temp = [x.strip() for x in line.split(",")]
            out.append({"index": int(idx), "util": float(util), "mem_free_gb": (float(total) - float(used)) / 1024.0, "mem_used_gb": float(used) / 1024.0, "temp_c": float(temp)})
        return out
    except Exception:
        return []


def read_sample() -> "dict[str, Any]":
    """One reading of the machine."""
    import psutil

    cores = psutil.cpu_percent(interval=None, percpu=True)
    gpus = _nvml_gpus()
    return {
        "t": time.time(),
        "cpu": float(sum(cores) / max(len(cores), 1)),
        "cores": [float(c) for c in cores],
        "ram_free_gb": psutil.virtual_memory().available / _GB,
        "gpus": gpus if gpus is not None else _smi_gpus(),
    }


class TimelineSampler:
    """Context manager that samples the machine every ``interval`` seconds into ``path`` (JSON lines; ``None`` keeps the samples in memory only) and keeps the samples in ``samples``."""

    def __init__(self, path: Optional[str] = None, interval: float = 0.5) -> None:
        self.path = path
        self.interval = float(interval)
        self.samples: "list[dict[str, Any]]" = []
        self._stop = threading.Event()
        self._thread: "Optional[threading.Thread]" = None

    def __getstate__(self) -> "dict[str, Any]":
        """The state without the sampler thread and the stop event (neither can be pickled); a copy is a stopped sampler."""
        state = dict(self.__dict__)
        state["_thread"] = None
        state.pop("_stop", None)
        return state

    def __setstate__(self, state: "dict[str, Any]") -> None:
        """Restore a stopped sampler with a fresh stop event."""
        self.__dict__.update(state)
        self._stop = threading.Event()

    def _run(self) -> None:
        """Sample until stopped."""
        import psutil

        psutil.cpu_percent(interval=None, percpu=True)  # the first call only arms the counters
        fh = open(self.path, "a", encoding="utf-8") if self.path else None  # closed in the finally below
        try:
            while not self._stop.wait(self.interval):
                sample = read_sample()
                self.samples.append(sample)
                if fh is not None:
                    fh.write(json.dumps(sample) + "\n")
                    fh.flush()
        finally:
            if fh is not None:
                fh.close()

    def __enter__(self) -> "TimelineSampler":
        """Start the sampler thread."""
        self._thread = threading.Thread(target=self._run, name="hardware-timeline", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Stop the sampler and wait for the last line to be written."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
