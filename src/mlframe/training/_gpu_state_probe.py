"""Best-effort, never-raising GPU state probe for log lines (utilisation, memory, other processes on the GPU).

Used by the crash-diagnostics heartbeat and the CatBoost GPU fit monitor. pynvml (nvidia-ml-py) is preferred because it is
in-process and cheap; ``nvidia-smi`` is the fallback (a short subprocess, only called at heartbeat cadence). When neither
works the probe remembers that and returns ``None`` from then on, so a box without NVIDIA tooling pays the cost once.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404 - fixed argv to nvidia-smi, no shell, no user input
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_BACKEND: Optional[str] = None  # "nvml" | "smi" | "none"; resolved lazily on first probe
_NVML: Any = None
_SMI_TIMEOUT_S = 5.0


def _resolve_backend() -> str:
    """Pick the probe backend once per process: ``"nvml"`` when pynvml initialises, else ``"smi"`` when nvidia-smi is on PATH, else ``"none"``."""
    global _BACKEND, _NVML
    with _LOCK:
        if _BACKEND is not None:
            return _BACKEND
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            _NVML = pynvml
            _BACKEND = "nvml"
        except Exception as e:
            logger.debug("pynvml unavailable, falling back to nvidia-smi if present: %s", e)
            _BACKEND = "smi" if shutil.which("nvidia-smi") else "none"
        return _BACKEND


def _probe_nvml() -> Dict[str, Any]:
    """Per-GPU utilisation / memory and the compute processes on each GPU, read in-process through NVML."""
    nv = _NVML
    gpus: List[Dict[str, Any]] = []
    procs: List[Dict[str, Any]] = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        gpus.append({"index": i, "util_pct": float(util.gpu), "mem_used_mb": mem.used / 2**20, "mem_total_mb": mem.total / 2**20})
        try:
            plist = nv.nvmlDeviceGetComputeRunningProcesses(h)
        except Exception as e:
            logger.debug("nvmlDeviceGetComputeRunningProcesses failed on gpu%d: %s", i, e)
            plist = []
        for p in plist:
            used = getattr(p, "usedGpuMemory", None)
            procs.append({"gpu": i, "pid": int(p.pid), "name": _proc_name(int(p.pid)), "mem_mb": (used / 2**20) if used else None})
    return {"gpus": gpus, "processes": procs}


def _proc_name(pid: int) -> str:
    """Executable name of ``pid`` via psutil, ``"?"`` when it cannot be read (process gone, no permission, psutil missing)."""
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception as e:
        logger.debug("process name lookup failed for pid %s: %s", pid, e)
        return "?"


def _num(s: str) -> Optional[float]:
    """``float(s)``, or ``None`` for nvidia-smi placeholders such as ``[N/A]``."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _probe_smi() -> Dict[str, Any]:
    """Same snapshot as :func:`_probe_nvml`, parsed from two ``nvidia-smi --query-*`` CSV calls (the process query is optional)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        exe = "nvidia-smi"
    out = subprocess.run(  # nosec B603
        [exe, "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": int(_num(parts[0]) or 0), "util_pct": _num(parts[1]), "mem_used_mb": _num(parts[2]), "mem_total_mb": _num(parts[3])})
    procs = []
    try:
        out2 = subprocess.run(  # nosec B603
            [exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=True,
        ).stdout
        for line in out2.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and _num(parts[0]) is not None:
                procs.append({"gpu": None, "pid": int(_num(parts[0])), "name": os.path.basename(parts[1]), "mem_mb": _num(parts[2])})
    except Exception as e:
        logger.debug("nvidia-smi compute-apps query failed: %s", e)
    return {"gpus": gpus, "processes": procs}


def gpu_snapshot() -> Optional[Dict[str, Any]]:
    """``{"gpus": [...], "processes": [...]}`` or ``None`` when no probe works. Never raises."""
    global _BACKEND
    try:
        backend = _resolve_backend()
        if backend == "nvml":
            return _probe_nvml()
        if backend == "smi":
            return _probe_smi()
    except Exception as e:
        logger.warning("GPU probe failed (%s); disabling it for this process: %s", _BACKEND, e)
        _BACKEND = "none"
    return None


def format_gpu_snapshot(snap: Optional[Dict[str, Any]], *, exclude_pid: Optional[int] = None, max_procs: int = 8) -> str:
    """One-line human summary; ``"gpu=n/a"`` when unavailable. ``exclude_pid`` drops the current process from the 'other processes' list."""
    if not snap:
        return "gpu=n/a"
    try:
        parts = []
        for g in snap.get("gpus", []):
            util = g.get("util_pct")
            used, total = g.get("mem_used_mb"), g.get("mem_total_mb")
            parts.append(
                f"gpu{g.get('index')}: util={util:.0f}%" if util is not None else f"gpu{g.get('index')}: util=?"
            )
            if used is not None and total:
                parts[-1] += f" mem={used:.0f}/{total:.0f}MB"
        others = other_gpu_processes(snap, exclude_pid=exclude_pid)
        if others and max_procs > 0:
            shown = ", ".join(
                f"{p.get('name')}[{p.get('pid')}]" + (f"={p['mem_mb']:.0f}MB" if p.get("mem_mb") else "") for p in others[:max_procs]
            )
            more = f" (+{len(others) - max_procs} more)" if len(others) > max_procs else ""
            parts.append(f"other GPU processes: {shown}{more}")
        return "; ".join(parts) if parts else "gpu=n/a"
    except Exception as e:
        logger.warning("format_gpu_snapshot failed on %r: %s", snap, e)
        return "gpu=?"


# On Windows (WDDM) nvidia-smi lists every desktop process holding a graphics context, without per-process VRAM. Those
# are not what slows a CUDA fit, and listing them buries the one process that matters (another trainer, a browser tab doing WebGL).
_DESKTOP_NOISE = frozenset(
    n.lower() for n in (
        "explorer.exe", "dwm.exe", "textinputhost.exe", "shellexperiencehost.exe", "searchapp.exe", "searchhost.exe",
        "startmenuexperiencehost.exe", "systemsettings.exe", "applicationframehost.exe", "lockapp.exe", "widgets.exe",
        "[insufficient permissions]", "csrss.exe", "runtimebroker.exe",
    )
)


def other_gpu_processes(snap: Optional[Dict[str, Any]], exclude_pid: Optional[int] = None) -> List[Dict[str, Any]]:
    """GPU processes other than ``exclude_pid``, without Windows desktop-shell noise, largest VRAM user first."""
    if not snap:
        return []
    out = [
        p for p in snap.get("processes", [])
        if (exclude_pid is None or p.get("pid") != exclude_pid) and str(p.get("name", "")).lower() not in _DESKTOP_NOISE
    ]
    return sorted(out, key=lambda p: -(p.get("mem_mb") or 0.0))
