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


def _cushion_bytes(total_b: int) -> int:
    """The absolute cushion floor in bytes for a device with ``total_b`` total VRAM -- shared by
    ``fe_gpu_has_vram_cushion`` and ``ensure_fe_gpu_pool_limit`` (2026-07-09 fix) so both mechanisms agree
    on ONE definition of "the reserved headroom" instead of each independently choosing a fraction that
    could, in combination, exceed total device memory on a small card (see ``ensure_fe_gpu_pool_limit``'s
    docstring for why the two must be computed jointly)."""
    cushion_b = _min_free_mb() * 1024 * 1024
    tiny = int(total_b * _TINY_CARD_CUSHION_FRACTION)
    if tiny > 0:
        cushion_b = min(cushion_b, tiny)
    return cushion_b


def fe_gpu_has_vram_cushion(bytes_needed: int = 0, *, free_b: "int | None" = None, total_b: "int | None" = None) -> bool:
    """Is there enough FREE VRAM to safely launch a GPU-FE kernel needing ``bytes_needed`` extra bytes?

    Returns ``free_b - bytes_needed >= cushion`` where ``cushion = min(MLFRAME_FE_GPU_MIN_FREE_MB,
    _TINY_CARD_CUSHION_FRACTION * total)`` -- an ABSOLUTE floor (default >=1 GB free), unlike the
    existing relative ``free_b * 0.5`` caps which are computed only AFTER the pool already ate the card.

    Cheap (one ``memGetInfo``); safe to call per-dispatch. PERMISSIVE (returns ``True``) whenever cupy
    or ``memGetInfo`` is unavailable / raises, so non-GPU hosts and probe failures are unaffected: the
    cushion can only DECLINE the GPU on a genuinely near-full card, never block a host that has no cupy.

    ``free_b``/``total_b``: pass the caller's OWN already-probed ``memGetInfo()`` result to skip this
    function's internal probe entirely (e.g. ``_cmi_cuda._should_use_cuda`` already queries ``memGetInfo``
    for its relative cap just above this call -- probing twice per dispatch is redundant). Either both or
    neither must be given; a partial pair falls back to probing (never silently mixes a stale value).

    Also lazily installs the own-pool cap on first call (see ``ensure_fe_gpu_pool_limit``)."""
    if free_b is None or total_b is None:
        try:
            import cupy as cp
        except Exception:  # -- no cupy: non-GPU host, stay permissive (caller's other gates decide)
            return True
        # Lazily cap our own pool so even the first cushion probe benefits from headroom.
        ensure_fe_gpu_pool_limit()
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
        except Exception as exc:  # -- probe failed: permissive, do not block the GPU on a probe error
            logger.debug("fe_gpu_has_vram_cushion: memGetInfo failed (%s); permissive", exc)
            return True
    else:
        # Caller already probed; still ensure the pool cap is installed (idempotent, no extra memGetInfo).
        ensure_fe_gpu_pool_limit()
    cushion_b = _cushion_bytes(total_b)
    if (free_b - int(bytes_needed)) >= cushion_b:
        return True
    # ``memGetInfo``'s free counts blocks RETAINED by our own cupy pool as used -- after a few FE stages the
    # pool can hold most of the card in internally-FREE blocks (instantly reusable by the next cupy alloc,
    # invisible to memGetInfo). Observed live (2026-07-14 wellbore 100k): free=0.52GB of 4GB with the pool
    # holding the rest, causing a 0.16GB batch_pair_mi upload to be REJECTED and the whole batched pair-MI
    # to fall off the full-resident path. Before declining, release the pool's free blocks back to the
    # device (a no-op for blocks actually in use) and re-probe -- the check is then exact, with no
    # fragmentation assumptions. The re-cudaMalloc cost this trades away only occurs where the alternative
    # was rejecting the GPU path outright.
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        if int(pool.free_bytes()) <= 0:
            return False
        pool.free_all_blocks()
        free_b2, total_b2 = cp.cuda.runtime.memGetInfo()
    except Exception as exc:
        logger.debug("fe_gpu_has_vram_cushion: pool free_all_blocks/re-probe failed (%s); declining", exc)
        return False
    return bool((int(free_b2) - int(bytes_needed)) >= _cushion_bytes(int(total_b2)))


def ensure_fe_gpu_pool_limit() -> bool:
    """ONCE per process, cap the cupy default memory pool to a fraction of total VRAM.

    So MRMR's OWN pool cannot exceed the device and always leaves headroom for concurrent processes /
    the model / the next launch. The requested ``MLFRAME_FE_GPU_POOL_FRACTION`` (default 0.6) is JOINTLY
    bounded against the absolute cushion floor (2026-07-09 fix): the effective cap is
    ``min(pool_fraction, (total - cushion) / total)`` -- computed from the SAME ``_cushion_bytes`` the
    per-dispatch cushion check uses -- so ``pool_cap_bytes + cushion_bytes`` never exceeds total device
    memory BY CONSTRUCTION. Before this fix the two fractions were chosen independently (0.6 pool +
    up-to-0.5-of-total cushion): on any card at/below ~2x the absolute cushion floor (~2 GB at the 1 GB
    default) their SUM exceeded 100% of total VRAM, i.e. the policy's own arithmetic guaranteed self-
    conflict regardless of the actual workload -- not just empirically fine on the 4 GB reference card.

    Idempotent (module-level ``_POOL_LIMIT_DONE`` flag); logs once at INFO. On exhaustion cupy raises
    ``OutOfMemoryError`` which the existing GPU-FE try/excepts catch -> graceful CPU. No-op (returns
    ``False``) when cupy is unavailable or any step raises."""
    global _POOL_LIMIT_DONE
    if _POOL_LIMIT_DONE:
        return False
    _POOL_LIMIT_DONE = True  # set first: a failed attempt must not retry every dispatch
    try:
        import cupy as cp
    except Exception:  # -- no cupy: nothing to cap
        return False
    try:
        requested_frac = _pool_fraction()
        try:
            _, total_b = cp.cuda.runtime.memGetInfo()
        except Exception:
            total_b = 0
        frac = requested_frac
        if total_b > 0:
            cushion_frac = _cushion_bytes(total_b) / total_b
            frac = max(0.0, min(requested_frac, 1.0 - cushion_frac))
        cp.get_default_memory_pool().set_limit(fraction=frac)
        if total_b > 0:
            logger.info(
                "fe_gpu_pool: capped cupy default pool at fraction=%.2f (~%d MiB of %d MiB total; "
                "jointly bounded against the VRAM cushion, requested fraction was %.2f)",
                frac, int(total_b * frac) // (1024 * 1024), total_b // (1024 * 1024), requested_frac,
            )
        else:  # -- logging detail only
            logger.info("fe_gpu_pool: capped cupy default pool at fraction=%.2f", frac)
        return True
    except Exception as exc:
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
