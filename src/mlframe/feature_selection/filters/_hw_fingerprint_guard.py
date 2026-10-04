"""Keeps GPU-less runs from poisoning the persisted hardware fingerprint that keys every kernel-tuning lookup.

pyutilz computes ``hw_fingerprint()`` once, persists it to ``<cache_dir>/.hw_fingerprint.json`` and reads it back in every later process for a week.
Its GPU probe turns any failure into ``no-gpu``, so one run with ``CUDA_VISIBLE_DEVICES=""``, ``MLFRAME_DISABLE_GPU=1`` or a transient driver fault
wrote a CPU-only fingerprint that later GPU runs then resolved, selecting backends (and re-tuning) under the CPU-only tuning directory.

The guard wraps three pyutilz functions:

* the GPU probe reports ``no-gpu`` for a run that opted out, so an opted-out run keys the CPU tuning directory in memory only;
* the disk write refuses to persist a ``no-gpu`` fingerprint unless the host genuinely has no usable CUDA device and the run did not opt out;
* the disk read ignores a persisted ``no-gpu`` fingerprint when this run can use a GPU, and rewrites a persisted GPU fingerprint to ``no-gpu`` for an
  opted-out run, so each run resolves the fingerprint of the hardware it will actually use.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_NO_GPU = "no-gpu"
_NO_GPU_SUFFIX = "_" + _NO_GPU
_GPU_MARK = "_gpu_"
_GUARD_ATTR = "_mlframe_hw_fp_guard"
_FACADE = "pyutilz.performance.kernel_tuning.cache"
_BASE = "pyutilz.performance.kernel_tuning.cache.cache_base"


def _opted_out() -> bool:
    """True when this run declared it must not use the GPU."""
    from ._gpu_policy import gpu_globally_disabled

    return gpu_globally_disabled()


def _gpu_usable() -> bool:
    """True when a CUDA device is present and this run may use it."""
    from ._gpu_policy import cuda_available_for_run

    return cuda_available_for_run()


def fingerprint_for_opted_out_run(fingerprint: str) -> str:
    """``cpu_<cpu>_gpu_<gpu>_cc<x>`` -> ``cpu_<cpu>_no-gpu``; any other fingerprint is returned unchanged."""
    head, sep, _ = fingerprint.partition(_GPU_MARK)
    return head + _NO_GPU_SUFFIX if sep else fingerprint


def guard_gpu_probe(probe: Callable[[], tuple]) -> Callable[[], tuple]:
    """Wrap ``_gpu_slug_and_cc`` so an opted-out run reports ``("no-gpu", "")``."""

    def guarded() -> tuple:
        """Report no GPU for an opted-out run, else the real probe."""
        if _opted_out():
            return (_NO_GPU, "")
        return probe()

    setattr(guarded, _GUARD_ATTR, True)
    return guarded


def guard_disk_write(write: Callable[[str], None]) -> Callable[[str], None]:
    """Wrap ``_write_hw_fingerprint_to_disk`` so only a genuine CPU-only host persists a ``no-gpu`` fingerprint."""

    def guarded(fingerprint: str) -> None:
        """Skip persisting a no-gpu fingerprint produced by an opt-out or a failed probe."""
        if fingerprint.endswith(_NO_GPU_SUFFIX) and (_opted_out() or _gpu_usable()):
            logger.debug("hw_fingerprint: not persisting %s (opt-out or a GPU is usable, so it would mislead later runs)", fingerprint)
            return
        write(fingerprint)

    setattr(guarded, _GUARD_ATTR, True)
    return guarded


def guard_disk_read(read: Callable[[], Optional[str]]) -> Callable[[], Optional[str]]:
    """Wrap ``_read_hw_fingerprint_from_disk`` so a persisted fingerprint is only trusted when it matches what this run can use."""

    def guarded() -> Optional[str]:
        """Return the persisted fingerprint adjusted to this run's GPU eligibility, or ``None`` to force a re-probe."""
        fp = read()
        if fp is None:
            return None
        if _opted_out():
            return fingerprint_for_opted_out_run(fp)
        if fp.endswith(_NO_GPU_SUFFIX) and _gpu_usable():
            return None
        return fp

    setattr(guarded, _GUARD_ATTR, True)
    return guarded


def _wrap(module: Any, name: str, guard: Callable[[Any], Any]) -> bool:
    """Replace ``module.name`` with ``guard(original)`` once; returns whether a wrapper was installed."""
    original = getattr(module, name, None)
    if original is None or getattr(original, _GUARD_ATTR, False):
        return False
    setattr(module, name, guard(original))
    return True


def install_hw_fingerprint_guard() -> bool:
    """Install the guard on pyutilz (idempotent, best-effort); returns True when pyutilz's kernel-tuning cache is importable.

    The wrappers are installed on both the ``cache`` facade and ``cache_base``: ``hw_fingerprint`` resolves the GPU probe through the facade and the disk
    helpers through its own module globals.
    """
    try:
        import pyutilz.performance.kernel_tuning.cache  # noqa: F401
    except ImportError:
        return False
    facade, base = sys.modules.get(_FACADE), sys.modules.get(_BASE)
    changed = False
    for module in (facade, base):
        if module is None:
            continue
        changed |= _wrap(module, "_gpu_slug_and_cc", guard_gpu_probe)
        changed |= _wrap(module, "_write_hw_fingerprint_to_disk", guard_disk_write)
        changed |= _wrap(module, "_read_hw_fingerprint_from_disk", guard_disk_read)
    if changed and base is not None:
        cached = getattr(getattr(base, "hw_fingerprint", None), "cache_clear", None)
        if cached is not None:
            cached()
    return True
