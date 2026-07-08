"""Centralized ABSOLUTE VRAM-cushion guard + memory-pool cap for the MRMR / FE GPU (cupy) paths.

Motivation (confirmed on a live 1M wellbore run, 2026-07-05): the 4 GB GTX 1050 Ti sat at ~322 MiB
free / 4094 during a fit. MRMR's cupy pool had NO ``set_limit`` cap so it grew unbounded and ate the
card, and every existing VRAM guard only computed a RELATIVE ``min(1536MB, free_b * 0.5)`` cushion
AFTER the pool had already consumed the device. On a near-full / SHARED card (concurrent pytest,
autocad-mcp also on the GPU) the next kernel launch then faulted (``cudaErrorLaunchFailure``), which
the GPU-FE try/excepts silently fell back to CPU -- the root cause of CMI / pair-MI running on CPU.

Two mechanisms, both a pure ADD (they only TIGHTEN when the GPU may be used, never loosen):

* ``fe_gpu_has_vram_cushion(bytes_needed)`` -- cheap absolute-cushion gate to call per-dispatch. Returns
  ``False`` (route CPU) whenever ``free - bytes_needed`` would drop below an ABSOLUTE cushion, so no
  kernel is launched on a near-full card. Permissive (``True``) when cupy / memGetInfo is unavailable,
  so non-GPU hosts are entirely unaffected.
* ``ensure_fe_gpu_pool_limit()`` -- ONCE per process, cap MRMR's OWN default memory pool to a fraction
  of total VRAM (default 0.6) so it cannot consume the whole device and starve concurrent processes /
  the next launch. On exhaustion cupy raises ``OutOfMemoryError`` which the existing GPU-FE try/excepts
  already catch -> graceful CPU fallback.

STRICT-vs-cushion decision (documented, deliberate): ``MLFRAME_FE_GPU_STRICT`` may force PAST the
conservative SIZE / crossover gates (a size threshold being conservative does not imply OOM), but it
does NOT bypass this cushion. A cushion violation means REAL out-of-memory risk on a contended card;
forcing a launch there would fault, not "diagnose". So the cushion is checked UNCONDITIONALLY, before
and independent of any STRICT override.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Require at least this many MB free before touching the GPU (absolute floor, NOT relative to the
# already-consumed pool). Default 1024 = >=1 GB free. Env-overridable per host / per contention level.
_DEFAULT_MIN_FREE_MB = 1024

# Cap MRMR's OWN cupy default pool to this fraction of TOTAL device memory so it leaves headroom for
# concurrent processes / the model / the next kernel launch. Env-overridable per host.
_DEFAULT_POOL_FRACTION = 0.6

# On a hypothetical very small device a fixed 1 GB cushion could be most/all of the card. Take the MIN of
# the absolute floor and HALF the TOTAL so the cushion never demands more headroom than a small card can
# offer -- while still using the FULL absolute floor on any normal card. At 0.5 the clamp only bites for
# cards <= 2*floor (i.e. <= ~2 GB at the 1 GB default); a 4 GB GTX 1050 Ti keeps the full 1 GB cushion.
_TINY_CARD_CUSHION_FRACTION = 0.5

# Set once ``ensure_fe_gpu_pool_limit`` has (attempted to) install the pool cap. Idempotent guard.
_POOL_LIMIT_DONE = False


def _min_free_mb() -> int:
    """Absolute free-VRAM floor in MB from ``MLFRAME_FE_GPU_MIN_FREE_MB`` (default 1024)."""
    try:
        return int(os.environ.get("MLFRAME_FE_GPU_MIN_FREE_MB", _DEFAULT_MIN_FREE_MB))
    except (ValueError, TypeError):
        return _DEFAULT_MIN_FREE_MB


def _pool_fraction() -> float:
    """Own-pool cap fraction from ``MLFRAME_FE_GPU_POOL_FRACTION`` (default 0.6)."""
    try:
        frac = float(os.environ.get("MLFRAME_FE_GPU_POOL_FRACTION", _DEFAULT_POOL_FRACTION))
    except (ValueError, TypeError):
        return _DEFAULT_POOL_FRACTION
    # Clamp to a sane (0, 1] range; a nonsense value must not disable the cap or over-cap.
    if not (0.0 < frac <= 1.0):
        return _DEFAULT_POOL_FRACTION
    return frac


def fe_gpu_has_vram_cushion(bytes_needed: int = 0) -> bool:
    """Is there enough FREE VRAM to safely launch a GPU-FE kernel needing ``bytes_needed`` extra bytes?

    Returns ``free_b - bytes_needed >= cushion`` where ``cushion = min(MLFRAME_FE_GPU_MIN_FREE_MB,
    _TINY_CARD_CUSHION_FRACTION * total)`` -- an ABSOLUTE floor (default >=1 GB free), unlike the
    existing relative ``free_b * 0.5`` caps which are computed only AFTER the pool already ate the card.

    Cheap (one ``memGetInfo``); safe to call per-dispatch. PERMISSIVE (returns ``True``) whenever cupy
    or ``memGetInfo`` is unavailable / raises, so non-GPU hosts and probe failures are unaffected: the
    cushion can only DECLINE the GPU on a genuinely near-full card, never block a host that has no cupy.

    Also lazily installs the own-pool cap on first call (see ``ensure_fe_gpu_pool_limit``)."""
    try:
        import cupy as cp
    except Exception:  # noqa: BLE001  -- no cupy: non-GPU host, stay permissive (caller's other gates decide)
        return True
    # Lazily cap our own pool so even the first cushion probe benefits from headroom.
    ensure_fe_gpu_pool_limit()
    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
    except Exception as exc:  # noqa: BLE001  -- probe failed: permissive, do not block the GPU on a probe error
        logger.debug("fe_gpu_has_vram_cushion: memGetInfo failed (%s); permissive", exc)
        return True
    cushion_b = _min_free_mb() * 1024 * 1024
    tiny = int(total_b * _TINY_CARD_CUSHION_FRACTION)
    if tiny > 0:
        cushion_b = min(cushion_b, tiny)
    return bool((free_b - int(bytes_needed)) >= cushion_b)


def ensure_fe_gpu_pool_limit() -> bool:
    """ONCE per process, cap the cupy default memory pool to ``MLFRAME_FE_GPU_POOL_FRACTION`` of total VRAM.

    So MRMR's OWN pool cannot exceed ~60% of the device and always leaves headroom for concurrent
    processes / the model / the next launch. Idempotent (module-level ``_POOL_LIMIT_DONE`` flag); logs
    once at INFO. On exhaustion cupy raises ``OutOfMemoryError`` which the existing GPU-FE try/excepts
    catch -> graceful CPU. No-op (returns ``False``) when cupy is unavailable or any step raises."""
    global _POOL_LIMIT_DONE
    if _POOL_LIMIT_DONE:
        return False
    _POOL_LIMIT_DONE = True  # set first: a failed attempt must not retry every dispatch
    try:
        import cupy as cp
    except Exception:  # noqa: BLE001  -- no cupy: nothing to cap
        return False
    try:
        frac = _pool_fraction()
        cp.get_default_memory_pool().set_limit(fraction=frac)
        try:
            _, total_b = cp.cuda.runtime.memGetInfo()
            logger.info(
                "fe_gpu_pool: capped cupy default pool at fraction=%.2f (~%d MiB of %d MiB total)",
                frac, int(total_b * frac) // (1024 * 1024), total_b // (1024 * 1024),
            )
        except Exception:  # noqa: BLE001  -- logging detail only
            logger.info("fe_gpu_pool: capped cupy default pool at fraction=%.2f", frac)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("fe_gpu_pool: set_limit failed (%s); pool uncapped", exc)
        return False


def _reset_fe_gpu_pool_limit_flag() -> None:
    """Test hook: re-arm the once-per-process pool-cap guard so a fresh ``ensure_...`` call runs again."""
    global _POOL_LIMIT_DONE
    _POOL_LIMIT_DONE = False


__all__ = [
    "fe_gpu_has_vram_cushion",
    "ensure_fe_gpu_pool_limit",
]
