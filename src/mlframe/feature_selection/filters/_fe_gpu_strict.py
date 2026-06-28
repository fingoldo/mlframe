"""STRICT GPU-mode diagnostic flag for the MRMR / FE GPU dispatch (2026-06-24).

DIAGNOSTIC-ONLY, default OFF, purely additive. ``MLFRAME_FE_GPU_STRICT=1`` forces EVERY FE GPU-vs-CPU
dispatch decision that HAS a bit-identical (selection-equivalent) GPU twin to choose the GPU, bypassing the
per-host KTC crossover / size threshold / default-OFF gate. It exists to answer ONE question: when the GPU
busy% looks low (~8-15%) on this card, is the FE "GPU path" CPU-bound because the KTC crossover gates many
kernels BACK to CPU (sub-crossover on the weak GTX 1050 Ti -- i.e. MIS-GATED / movable on a stronger card),
or because the remaining work is GENUINELY CPU-only (Python greedy-selection orchestration, the sequential
``_combine_factorize_njit``, scipy.special ops -- the irreducible residual)?

WHAT STRICT FORCES (each has a proven selection-equivalent GPU twin -- ~1e-15 / argmax-identical):
  * ``_cmi_cuda._should_use_cuda``           -- batched conditional-MI CPU<->CUDA crossover
  * ``_cmi_cuda_ktc.cmi_use_cuda``           -- the swept crossover backing the above
  * ``_resident_candidate_mi_ktc``           -- resident GPU candidate-gen + plug-in MI
  * ``_permutation_null_resident_ktc``       -- resident GPU maxT permutation-null floor
  * ``_usability_pool_resident_ktc``         -- resident GPU batched pair-combo MI table
  * ``_pairs_core._fe_gpu_discretize_enabled`` / ``_fe_gpu_binning_enabled`` -- FE candidate binning + MI

WHAT STRICT CANNOT FORCE (no bit-identical GPU twin -- these stay CPU and ARE the "truly CPU-only" residual):
  * the Python greedy-selection orchestration loop itself
  * the sequential ``_combine_factorize_njit`` factorize
  * ``scipy.special`` analytic-MI ops, fingerprint / recipe-replay / validation glue
The radix / histgate threads-per-block + kernel-variant KTC specs are GPU-vs-GPU tuning (the work is ALREADY
on the GPU); strict is a no-op for them -- they never route to CPU.

STRICT IS A NO-OP WHEN CUDA IS ABSENT: with no usable device the gate returns False and the dispatch is the
exact CPU path -- byte-for-byte the legacy no-GPU behavior. STRICT NEVER CHANGES THE DEFAULT (flag unset =
current KTC-gated behavior, unchanged): forcing GPU on these gateable kernels is selection-equivalent, so the
SAME compound + recipes are recovered under both flag states.
"""
from __future__ import annotations

import os

# Quiet the intermittent cupy<->numba illegal-address race at interpreter teardown (cosmetic; suppressed ONLY
# during finalization, never mid-fit -- see _gpu_teardown_guard). Cheap import (no cupy), idempotent install.
from ._gpu_teardown_guard import install_cuda_teardown_guard as _install_cuda_teardown_guard

_install_cuda_teardown_guard()

# Only the device-availability probe is cached (CUDA presence is immutable for the process); the ENV FLAG is
# read LIVE on every call. Caching the flag's first-seen value was a latent bug: a process that toggled
# MLFRAME_FE_GPU_STRICT mid-run (or a test suite where one test sets/unsets it before another) would freeze on
# the stale first value -> order-dependent dispatch. The live env read is ~1.5us/call (measured), negligible
# next to a greedy round; the expensive ~17us pyutilz/numba CUDA probe stays memoised below.
_CUDA_USABLE_CACHE: bool | None = None


def _cuda_usable() -> bool:
    """Best-effort CUDA-availability probe (mirrors ``_gpu_resident_fe._cuda_present``); any failure -> False.

    Memoised for the process lifetime: device presence does not change mid-run, and this probe shells into
    pyutilz/numba (~17us/call) which the per-greedy-round dispatch must not pay repeatedly. The CUDA_VISIBLE_DEVICES
    / MLFRAME_DISABLE_GPU short-circuits are part of the cached result -- those are start-of-process device gates,
    not the runtime STRICT toggle which is read live in ``fe_gpu_strict_enabled``."""
    global _CUDA_USABLE_CACHE
    if _CUDA_USABLE_CACHE is None:
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if (_cvd is not None and _cvd.strip() == "") or os.environ.get("MLFRAME_DISABLE_GPU", "") == "1":
            _CUDA_USABLE_CACHE = False
        else:
            try:
                from pyutilz.core.pythonlib import is_cuda_available
                _CUDA_USABLE_CACHE = bool(is_cuda_available())
            except Exception:
                try:
                    from numba import cuda as _c
                    _CUDA_USABLE_CACHE = bool(getattr(_c, "is_available", lambda: False)())
                except Exception:
                    _CUDA_USABLE_CACHE = False
    return _CUDA_USABLE_CACHE


def fe_gpu_strict_enabled() -> bool:
    """Whether STRICT GPU mode is active: ``MLFRAME_FE_GPU_STRICT`` truthy AND a CUDA device is usable.

    The env flag is read LIVE on every call so a mid-process toggle (or per-test set/unset) is observed
    immediately -- the only cached part is the immutable CUDA-device probe (see ``_cuda_usable``). Default OFF ->
    returns False and every gate keeps its current KTC-crossover behavior, so this is purely additive. No-op
    without CUDA."""
    _on = os.environ.get("MLFRAME_FE_GPU_STRICT", "").strip().lower() in ("1", "true", "on", "yes")
    return bool(_on and _cuda_usable())
