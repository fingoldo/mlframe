"""GPU backend gates for the FE pair search: memoized size-and-hardware decisions for GPU candidate binning and pair-MI.

Carved verbatim out of ``_pairs_core.py`` to keep it under the 1k-line ceiling; ``_pairs_core`` re-exports every name so the existing
``from ._pairs_core import _fe_gpu_discretize_enabled`` call sites keep working.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# _fe_gpu_discretize_enabled/_fe_gpu_binning_enabled are
# called up to 3x per pair, once per chunk, and once per ext-val tied-leader-set (unlike their sibling
# resolve_fe_dispatch_env_gate(), which IS hoisted once per check_prospective_fe_pairs call via
# _fe_env_gate) - each call re-reads env vars, imports+calls cuda_available_for_run(), and (on the default
# "auto" setting) a kernel_tuning_cache-backed backend-choice lookup, violating the repo's own "hoist the
# dispatch decision out of hot loops" rule. Threading a hoisted bool through the 4-file call graph
# (_pairs_core/_pairs_score/_pairs_chunks/_pairs_emit) would be the "proper" fix but touches a lot of the
# 60+-param hot-loop signature surface for a purely efficiency (not correctness) finding; memoizing each
# function's own result by its actual inputs is equivalent and self-contained (same inputs -> same
# output, and the env vars this reads are not supported to change mid-fit - the already-hoisted
# _fe_env_gate makes the identical assumption for its own env reads).
_GPU_GATE_CACHE: dict[tuple, bool] = {}
_GPU_GATE_CACHE_MAX = 64
# Guards every mutate/iterate of the bounded LRU below: concurrent fits share this module-level dict,
# and next(iter(...))/pop/insert from two threads corrupts the OrderedDict or drops the wrong entry.
_GPU_GATE_CACHE_LOCK = threading.Lock()


def _gpu_gate_env_signature() -> str:
    """Env inputs the gate decision depends on, folded into the cache key so a mid-process strict-mode flip
    (MLFRAME_FE_GPU_STRICT / MLFRAME_DISABLE_GPU) invalidates the memo instead of returning a stale verdict.

    The per-gate tri-state kill-switches (MLFRAME_FE_GPU_DISCRETIZE / MLFRAME_FE_GPU_BINNING) are read LIVE by
    the uncached gates, so they belong in the key too: without them, flipping the discretize switch mid-process
    is silently defeated by a memo entry computed under the old value."""
    return "|".join(
        os.environ.get(name, "") for name in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_DISABLE_GPU", "MLFRAME_FE_GPU_DISCRETIZE", "MLFRAME_FE_GPU_BINNING")
    )


def _gpu_gate_cached(kind: str, n_rows: int, n_cands: int, compute) -> bool:
    """Thread-safe bounded-LRU memo shared by the discretize/binning gates."""
    key = (kind, _gpu_gate_env_signature(), int(n_rows), int(n_cands))
    with _GPU_GATE_CACHE_LOCK:
        if key in _GPU_GATE_CACHE:
            return _GPU_GATE_CACHE[key]
        result = bool(compute(n_rows, n_cands))
        if key not in _GPU_GATE_CACHE and len(_GPU_GATE_CACHE) >= _GPU_GATE_CACHE_MAX:
            # evict-ok: memo; a miss recomputes the value
            _GPU_GATE_CACHE.pop(next(iter(_GPU_GATE_CACHE)))
        _GPU_GATE_CACHE[key] = result
    return result


def _fe_gpu_discretize_enabled(n_rows: int, n_cands: int) -> bool:
    """Memoized wrapper over :func:`_fe_gpu_discretize_enabled_uncached` - see the note above."""
    return _gpu_gate_cached("discretize", n_rows, n_cands, _fe_gpu_discretize_enabled_uncached)


def _fe_gpu_discretize_enabled_uncached(n_rows: int, n_cands: int) -> bool:
    """Whether to run the per-pair candidate MI (binning + observed-MI) on the GPU. The GPU path is
    BIT-IDENTICAL to the CPU analytic dispatch (verified maxdiff 0 on binning + observed MI), so the FE
    selection is unchanged either way - this only chooses the faster backend for the size.

    ``MLFRAME_FE_GPU_DISCRETIZE`` tri-state: ``0/false`` forces CPU; ``1/true`` forces GPU when CUDA is
    present; unset/``auto`` (the default) routes per-host via kernel_tuning_cache - GPU only above the
    measured n*K crossover, CPU below - so a small fit is never regressed and a slow-H2D host that loses
    on GPU is routed to CPU. Requires CUDA; any GPU failure falls back to the CPU dispatcher downstream."""
    _env = os.environ.get("MLFRAME_FE_GPU_DISCRETIZE", "auto").strip().lower()
    if _env in ("0", "false", "no", "off"):
        return False
    try:
        from .._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return False
    except Exception as e:
        # fallback: CUDA absent is the normal CPU-only case; the GPU discretize gate then stays on the CPU path.
        logger.debug("_fe_gpu_discretize_enabled_uncached: cuda_available_for_run() check failed, defaulting to CPU: %s", e)
        return False
    if _env in ("1", "true", "yes", "on"):
        return True
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, diagnostic, default OFF): force GPU past the KTC crossover.
    # The GPU pair-MI path is bit-identical to the CPU analytic dispatch (verified maxdiff 0) -> selection-equivalent.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled

        if fe_gpu_strict_enabled(n=int(n_rows), p=int(n_cands)):
            return True
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("fe_gpu_strict_enabled() check failed, continuing to the crossover-based decision: %s", e)
    try:  # auto: per-host crossover from kernel_tuning_cache (measurement-backed fallback)
        from .._gpu_resident_fe import fe_gpu_pairs_mi_backend_choice

        return bool(fe_gpu_pairs_mi_backend_choice(int(n_rows), int(n_cands)) == "gpu")
    except Exception as e:
        # fallback: the KTC backend choice is an optional probe; any failure keeps the CPU discretize path.
        logger.debug("_fe_gpu_discretize_enabled_uncached: KTC backend-choice lookup failed, defaulting to CPU: %s", e)
        return False


def _fe_gpu_binning_enabled(n_rows: int, n_cands: int) -> bool:
    """Memoized wrapper over :func:`_fe_gpu_binning_enabled_uncached` - see the note above."""
    return _gpu_gate_cached("binning", n_rows, n_cands, _fe_gpu_binning_enabled_uncached)


def _fe_gpu_binning_enabled_uncached(n_rows: int, n_cands: int) -> bool:
    """Whether to run the FE candidate BINNING (``discretize_2d_quantile_batch``) on the GPU.

    DECOUPLED from ``_fe_gpu_discretize_enabled`` / the full ``fe_gpu_pairs_mi`` analytic
    path. The binning alone (``gpu_discretize_codes_host``) is BIT-IDENTICAL to the CPU njit binning
    (verified maxdiff 0) and 17-24x faster at n=100k - but the FULL pair-MI sweep cached "cpu" for the
    n_rows<=100000 region (its extra GPU MI/chi2 overhead lost a noisy A/B there), which wrongly forced
    the cheap binning back onto the CPU njit path (the #1 wall hotspot: 116s of a 228s GPU-mode F2 100k
    fit). The binning is a strictly simpler op with its own crossover, so it routes through its own
    dedicated KTC backend choice here. Selection is unchanged either way (bit-identical codes).

    ``MLFRAME_FE_GPU_BINNING`` tri-state: ``0/false`` forces CPU; ``1/true`` forces GPU when CUDA is
    present; unset/``auto`` (default) routes per-host via kernel_tuning_cache. ``MLFRAME_FE_GPU_DISCRETIZE=0``
    still forces CPU for BOTH (a global FE-GPU kill switch), so existing CPU-only configs are unchanged."""
    _env_all = os.environ.get("MLFRAME_FE_GPU_DISCRETIZE", "auto").strip().lower()
    if _env_all in ("0", "false", "no", "off"):
        return False  # global FE-GPU kill switch also disables binning
    _env = os.environ.get("MLFRAME_FE_GPU_BINNING", "auto").strip().lower()
    if _env in ("0", "false", "no", "off"):
        return False
    try:
        from .._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return False
    except Exception as e:
        # fallback: CUDA absent is the normal CPU-only case; the GPU binning gate then stays on the CPU path.
        logger.debug("_fe_gpu_binning_enabled_uncached: cuda_available_for_run() check failed, defaulting to CPU: %s", e)
        return False
    if _env in ("1", "true", "yes", "on"):
        return True
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, diagnostic, default OFF): force GPU binning past the KTC
    # crossover. The GPU binning is bit-identical to the CPU njit binning (verified maxdiff 0) -> selection-equivalent.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled

        if fe_gpu_strict_enabled(n=int(n_rows), p=int(n_cands)):
            return True
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("fe_gpu_strict_enabled() check failed, continuing to the crossover-based decision: %s", e)
    try:  # auto: per-host binning crossover from kernel_tuning_cache (measurement-backed fallback)
        from .._gpu_resident_fe import fe_gpu_binning_backend_choice

        return bool(fe_gpu_binning_backend_choice(int(n_rows), int(n_cands)) == "gpu")
    except Exception as e:
        # fallback: the KTC binning choice is an optional probe; any failure keeps the CPU binning path.
        logger.debug("_fe_gpu_binning_enabled_uncached: KTC binning backend-choice lookup failed, defaulting to CPU: %s", e)
        return False
