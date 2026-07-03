"""STRICT GPU-mode flag for the MRMR / FE GPU dispatch (2026-06-24; AUTO size-gated default 2026-07-03).

``MLFRAME_FE_GPU_STRICT=1`` forces EVERY FE GPU-vs-CPU dispatch decision that HAS a bit-identical
(selection-equivalent) GPU twin to choose the GPU, bypassing the per-host KTC crossover / size threshold. ``=0``
forces the exact CPU path. When UNSET (or ``auto``) the flag is now AUTO size-gated (``fe_gpu_strict_enabled``): it
engages STRICT automatically on fits at/above ``MLFRAME_FE_GPU_STRICT_AUTO_MIN_N`` (default 100k) with a usable CUDA
device -- the regime where STRICT is measured selection-equivalent to CPU (small-n divergence is finite-sample MI
variance that fades by ~50k) and ~2.5x faster; below the threshold, or with no GPU, the exact CPU path runs
(byte-identical legacy). It originally existed to answer ONE diagnostic question: when the GPU
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
exact CPU path -- byte-for-byte the legacy no-GPU behavior. The AUTO default only engages STRICT where it is
selection-equivalent to CPU (large n, GPU present), so the SAME compound + recipes are recovered; small-n / no-GPU
fits are byte-identical to the legacy default. Force ``MLFRAME_FE_GPU_STRICT=0`` to pin the exact CPU path at any n.
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

# AUTO size-gated default: the STRICT resident FE path is measured selection-equivalent to the CPU path once n is
# large enough (the small-n divergence is finite-sample MI-estimation variance that fades as n grows -- convergence
# across scenarios by ~50k) AND it is ~2.5x faster there. So when MLFRAME_FE_GPU_STRICT is UNSET (or "auto"), STRICT
# engages automatically on fits at or above the threshold (and with a usable CUDA device); below it, or explicitly
# "0", the exact CPU path runs. ``_AUTO_FIT_N`` is the current fit's row count, set by the MRMR entry around fit()
# (None => AUTO stays off, so any non-MRMR caller is unchanged). Threshold is env-overridable per host.
_AUTO_FIT_N: int | None = None
# Conservative default: STRICT is measured selection-equivalent to CPU by ~50k, but the AUTO threshold is set at
# 100k -- the mlframe production regime (100k-100M rows) where the win lands -- so it sits comfortably ABOVE the
# convergence point AND above every existing test fit (all <=60k stay on the exact CPU path). Env-overridable per host.
_DEFAULT_AUTO_MIN_N = 100_000


def set_auto_fit_n(n: int | None) -> None:
    """Record the current fit's row count so the AUTO (unset-env) STRICT default can size-gate on it. MRMR.fit sets
    this at entry and clears it in a finally; None disables the AUTO path (leaving STRICT off unless env-forced)."""
    global _AUTO_FIT_N
    _AUTO_FIT_N = int(n) if n is not None else None


def clear_auto_fit_n() -> None:
    global _AUTO_FIT_N
    _AUTO_FIT_N = None


def _auto_min_n() -> int:
    try:
        return int(os.environ.get("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", _DEFAULT_AUTO_MIN_N))
    except (ValueError, TypeError):
        return _DEFAULT_AUTO_MIN_N


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
    """Whether STRICT GPU mode is active. Three-state ``MLFRAME_FE_GPU_STRICT``:

    * ``1``/``true``/``on``/``yes`` -- force STRICT (subject to a usable CUDA device).
    * ``0``/``false``/``off``/``no`` -- force OFF (exact CPU path; byte-identical legacy).
    * unset / ``auto`` -- AUTO size-gated default: STRICT engages when the current fit's row count is at/above
      ``MLFRAME_FE_GPU_STRICT_AUTO_MIN_N`` (default 100k -- the production regime, above the ~50k convergence point)
      and a CUDA device is usable; below the threshold (or when no fit-n context is set) it is OFF.

    The env is read LIVE every call (mid-process toggles observed immediately); only the immutable CUDA-device probe
    is cached. No-op without CUDA. Small-n and no-GPU behavior is byte-identical to the pre-AUTO default."""
    raw = os.environ.get("MLFRAME_FE_GPU_STRICT", "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return bool(_cuda_usable())
    if raw in ("", "auto"):
        n = _AUTO_FIT_N
        if n is not None and n >= _auto_min_n():
            return bool(_cuda_usable())
    return False
