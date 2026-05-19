"""Stable short fingerprint of the live host for the kernel-tuning cache.

The fingerprint must be:
* Short -- it ends up as a filename, so 60 chars max.
* Stable -- the same machine yields the same key across runs (no
  timestamps, no PIDs, no driver version unless it materially changes
  kernel performance).
* Discriminating -- different machines with measurable kernel-perf
  differences MUST get different keys. cc-level (e.g. cc 6.1 vs cc 8.6)
  matters most; CPU model matters less since the kernels run on GPU.

Current schema::

    cpu_<CPU_MODEL_SLUG>_gpu_<GPU_NAME_SLUG>_cc<MAJOR>.<MINOR>
    cpu_unknown_no_gpu   (CPU-only host -- the GPU cache is irrelevant
                          but we still want a key for CPU-only kernel
                          tuning, e.g. numba thread count)
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


def _slug(s: str, maxlen: int = 40) -> str:
    """Snake-case + lowercase + strip vendor noise + truncate."""
    s = re.sub(r"\(R\)|\(TM\)|\bCPU\b|\bGPU\b|@.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.strip("-._").lower()[:maxlen] or "unknown"


def _cpu_model_slug() -> str:
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return _slug(info.get("brand_raw", "unknown"))
    except Exception:
        return "unknown"


def _gpu_slug_and_cc() -> tuple[str, str]:
    """Returns (gpu_name_slug, cc_str) or ("no-gpu", "") on CPU-only hosts."""
    try:
        from pyutilz.system.gpu_dispatch import gpu_capability_summary
        s = gpu_capability_summary(0)
        if s is None:
            return ("no-gpu", "")
        name = s.get("name") or "unknown"
        cc = f"{int(s.get('cc_major', 0))}.{int(s.get('cc_minor', 0))}"
        return (_slug(name), cc)
    except Exception as e:
        logger.debug("gpu_capability_summary failed: %s", e)
        return ("no-gpu", "")


@lru_cache(maxsize=1)
def hw_fingerprint() -> str:
    """Stable per-host kernel-tuning cache key.

    Format: ``cpu_<slug>_gpu_<slug>_cc<major>.<minor>``. The result is
    cached per-process (host doesn't change mid-run).
    """
    cpu = _cpu_model_slug()
    gpu, cc = _gpu_slug_and_cc()
    if gpu == "no-gpu":
        return f"cpu_{cpu}_no-gpu"
    return f"cpu_{cpu}_gpu_{gpu}_cc{cc}"


def cache_dir() -> str:
    """Resolve the on-disk cache directory.

    Honours ``$MLFRAME_KERNEL_CACHE_DIR`` if set; else falls back to
    ``~/.mlframe/kernel_tuning``. Creates the directory on first call.
    """
    override = os.environ.get("MLFRAME_KERNEL_CACHE_DIR", "").strip()
    if override:
        path = override
    else:
        path = os.path.join(os.path.expanduser("~"), ".mlframe", "kernel_tuning")
    os.makedirs(path, exist_ok=True)
    return path


def cache_path() -> str:
    """Full path to the JSON file for the live host."""
    return os.path.join(cache_dir(), f"{hw_fingerprint()}.json")
