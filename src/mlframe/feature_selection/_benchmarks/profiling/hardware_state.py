"""Thermal and clock state of the GPU and the CPU, for timing runs.

A card and a CPU that throttle make the same code time differently: the same all-families fit ran in 52 s early in a session and in 65 to 96 s after hours of GPU work, with the GPU at 82 C idling at
139 MHz of 1911 (CUDA.md section 6, "Thermal state and clocks"). Every timing that is quoted or compared should carry this state, taken right before and right after the run.

``snapshot()`` is one reading, ``line()`` renders it on one line, ``wait_until_cool()`` blocks until the GPU is at or below a temperature (``MLFRAME_BENCH_WAIT_COOL_C`` makes the all-families harnesses do
it before the warm-up), ``summarise_samples()`` renders the per-sample readings a running sampler collected (mean and maximum temperature, mean SM clock against the maximum, the throttle reasons seen).
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

#: Throttle-reason bits of ``clocks_throttle_reasons.active`` that mean the clock is held down (idle and application-clock bits are not throttling).
_THROTTLE_BITS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}

_QUERY = "temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,utilization.gpu"


def _smi() -> str:
    """The ``nvidia-smi`` executable path (the project's own resolver, which knows the Windows install)."""
    from mlframe.training._gpu_state_probe import _resolve_smi

    return str(_resolve_smi())


def _gpu_reading() -> dict:
    """One ``nvidia-smi`` reading of device 0 as a dict; empty when it cannot be read."""
    try:
        out = subprocess.run([_smi(), f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False).stdout
        t, sm, sm_max, mem, mask, util = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {
            "gpu_temp_c": float(t),
            "gpu_sm_mhz": float(sm),
            "gpu_sm_max_mhz": float(sm_max),
            "gpu_mem_mhz": float(mem),
            "gpu_throttle_mask": int(mask, 16) if mask.lower().startswith("0x") else int(float(mask)),
            "gpu_util_pct": float(util),
        }
    except Exception:
        return {}


def _cpu_reading() -> dict:
    """The CPU load and current / maximum clock from ``psutil``; empty when it is not importable."""
    try:
        import psutil

        freq = psutil.cpu_freq()
        out = {"cpu_load_pct": float(psutil.cpu_percent(interval=None))}
        if freq is not None:
            out.update({"cpu_mhz": float(freq.current), "cpu_max_mhz": float(freq.max or 0.0)})
        return out
    except Exception:
        return {}


def snapshot() -> dict:
    """One reading of the GPU (temperature, SM / memory clock, throttle mask, utilisation) and the CPU (load, clock)."""
    return {**_gpu_reading(), **_cpu_reading(), "time": time.time()}


def throttle_reasons(mask: int) -> list:
    """Names of the throttling reasons set in the ``clocks_throttle_reasons.active`` bit mask."""
    return [name for bit, name in _THROTTLE_BITS.items() if mask & bit]


def line(snap: Optional[dict] = None) -> str:
    """One-line rendering of a snapshot (taken now when none is given)."""
    s = snapshot() if snap is None else snap
    if "gpu_temp_c" not in s:
        gpu = "GPU n/a"
    else:
        why = throttle_reasons(int(s["gpu_throttle_mask"]))
        gpu = f"GPU {s['gpu_temp_c']:.0f} C, SM {s['gpu_sm_mhz']:.0f}/{s['gpu_sm_max_mhz']:.0f} MHz, mem {s['gpu_mem_mhz']:.0f} MHz, util {s['gpu_util_pct']:.0f}%, throttle {','.join(why) or 'none'}"
    cpu = f"CPU load {s['cpu_load_pct']:.0f}%" + (f", {s['cpu_mhz']:.0f}/{s['cpu_max_mhz']:.0f} MHz" if "cpu_mhz" in s else "") if "cpu_load_pct" in s else "CPU n/a"
    return f"{gpu}; {cpu}"


def wait_until_cool(max_temp_c: float = 62.0, timeout_s: float = 900.0, poll_s: float = 10.0) -> dict:
    """Block until the GPU is at or below ``max_temp_c`` (or ``timeout_s`` passed); returns the last snapshot. A card that cannot be read returns at once."""
    deadline = time.time() + timeout_s
    snap = snapshot()
    while "gpu_temp_c" in snap and snap["gpu_temp_c"] > max_temp_c and time.time() < deadline:
        time.sleep(poll_s)
        snap = snapshot()
    return snap


def summarise_samples(samples: list) -> str:
    """Mean and maximum GPU temperature, mean SM clock against the maximum clock, the mean CPU clock and the throttle reasons seen, over the readings a sampler collected."""
    gpu = [s for s in samples if "gpu_temp_c" in s]
    if not gpu:
        return "hardware: no GPU readings"
    temps = [s["gpu_temp_c"] for s in gpu]
    sm = [s["gpu_sm_mhz"] for s in gpu]
    sm_max = max(s["gpu_sm_max_mhz"] for s in gpu)
    seen = sorted({name for s in gpu for name in throttle_reasons(int(s["gpu_throttle_mask"]))})
    cpu = [s["cpu_mhz"] for s in samples if "cpu_mhz" in s]
    cpu_max = max((s["cpu_max_mhz"] for s in samples if "cpu_max_mhz" in s), default=0.0)
    cpu_txt = f", CPU mean clock {sum(cpu) / len(cpu):.0f}/{cpu_max:.0f} MHz" if cpu else ""
    return f"hardware over the run: GPU temperature mean {sum(temps) / len(temps):.0f} C, max {max(temps):.0f} C; SM clock mean {sum(sm) / len(sm):.0f}/{sm_max:.0f} MHz; throttle seen: {', '.join(seen) or 'none'}{cpu_txt}"
