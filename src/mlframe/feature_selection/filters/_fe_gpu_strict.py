"""STRICT GPU-mode flag for the MRMR / FE GPU dispatch (2026-06-24; AUTO size-gated default 2026-07-03).

``MLFRAME_FE_GPU_STRICT=1`` forces EVERY FE GPU-vs-CPU dispatch decision that HAS a bit-identical
(selection-equivalent) GPU twin to choose the GPU, bypassing the per-host KTC crossover / size threshold. ``=0``
forces the exact CPU path. When UNSET (or ``auto``) the flag is now AUTO size-gated (``fe_gpu_strict_enabled``): it
engages STRICT automatically on fits that clear ``_auto_gate_passes`` -- EITHER row count at/above
``MLFRAME_FE_GPU_STRICT_AUTO_MIN_N`` (default 100k) regardless of column count, OR (2026-07-11: the row-only
threshold ignored column count entirely, so a wide-but-under-100k-rows fit like a 79,237-row x 544-column
production fold -- ~43M total elements, far past the point resident FE is worthwhile -- stayed on CPU purely for
sitting under an arbitrary row mark) row count at/above the ~50k VALIDATED convergence floor AND the fit's total
(row, column) work clearing the same ``n*p`` / ``p`` floor a per-call dispatch already needs -- with a usable CUDA
device in either case; otherwise the exact CPU path runs (byte-identical legacy). The regime where STRICT is
measured selection-equivalent to CPU: small-n divergence is finite-sample MI variance that fades by ~50k, and it
is ~2.5x faster there. It originally existed to answer ONE diagnostic question: when the GPU
busy% looks low (~8-15%) on this card, is the FE "GPU path" CPU-bound because the KTC crossover gates many
kernels BACK to CPU (sub-crossover on the weak GTX 1050 Ti -- i.e. MIS-GATED / movable on a stronger card),
or because the remaining work is GENUINELY CPU-only (Python greedy-selection orchestration, the sequential
``_combine_factorize_njit``, scipy.special ops -- the irreducible residual)?

WHAT STRICT FORCES (each has a proven selection-equivalent GPU twin -- ~1e-15 / argmax-identical):
  * ``_cmi_cuda._should_use_cuda``  -- batched conditional-MI CPU<->CUDA crossover
  * ``_cmi_cuda_ktc.cmi_use_cuda``  -- the swept crossover backing the above
  * ``_resident_candidate_mi_ktc``  -- resident GPU candidate-gen + plug-in MI
  * ``_permutation_null_resident_ktc``  -- resident GPU maxT permutation-null floor
  * ``_usability_pool_resident_ktc``  -- resident GPU batched pair-combo MI table
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

RESIDENCY GAP FOUND AND CLOSED (found 2026-07-16, fixed 2026-07-17; ``_mi_greedy_cmi_fe.greedy_cmi_fe_construct``,
called ONLY from the standalone benchmark script ``_benchmarks/bench_cmi_greedy_noisefloor_marginal_hoist.py``,
never from ``MRMR.fit``): its STRICT branch used to bin every candidate through the host-materializing
``_quantile_bin`` (D2H per candidate for the ``.tobytes()`` fingerprint) and fold each selected winner into the
conditioning support Z via the host-only ``_renumber_joint`` -- 212 bulk D2H events / fit on an 8000x11 synthetic
frame (``test_cmi_residency_traffic.py``). Closed by: (1) binning every candidate once via
``_quantile_bin_gpu_resident`` and fingerprinting the resident codes with a device-side reduction hash instead of
host bytes; (2) folding the Z conditioning support fully on-device via ``_renumber_joint_gpu`` (the existing
device twin of ``_renumber_joint``) instead of materializing each winner's codes host-side; (3) making the
y/z-invariant CMI precompute (``precompute_cmi_yz_terms``, still host-only -- no resident twin) LAZY, computed
only on the rare batched-CMI-call exception fallback instead of unconditionally every round. Verified: 0 bulk D2H
on the same fixture (down from 212), selection-equivalent (bit-identical fold partition via ``_renumber_joint_gpu``
-- see its own docstring), full CMI-greedy test suite green. The host ``_quantile_bin`` / ``_renumber_joint`` /
eager-``cand_bins`` code paths are kept intact as the non-resident fallback (GPU off, or a resident call faults
mid-fit) -- see ``_host_bins`` / the ``cand_bins_dev`` truthiness checks throughout ``greedy_cmi_fe_construct``.
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
# "0", the exact CPU path runs. ``_AUTO_FIT_N``/``_AUTO_FIT_P`` are the current fit's row/column counts, set by the
# MRMR entry around fit() (None => AUTO stays off, so any non-MRMR caller is unchanged). Threshold is
# env-overridable per host.
_AUTO_FIT_N: int | None = None
_AUTO_FIT_P: int | None = None
# Conservative default: STRICT is measured selection-equivalent to CPU by ~50k, but the AUTO threshold is set at
# 100k -- the mlframe production regime (100k-100M rows) where the win lands -- so it sits comfortably ABOVE the
# convergence point AND above every existing test fit (all <=60k stay on the exact CPU path). Env-overridable per host.
_DEFAULT_AUTO_MIN_N = 100_000

# PER-CALL work floor (2026-07-09 fix). Before this, AUTO gated ONLY on the FIT's total row count: once a fit
# crossed ``_DEFAULT_AUTO_MIN_N`` rows, EVERY subsequent dispatch was force-routed to STRICT for the REST of the
# fit, including late-round calls whose OWN (n, p) shape is tiny (e.g. a handful of remaining candidates in the
# greedy loop) -- exactly the small-shape regime the per-host KTC crossover exists to route back to CPU (kernel
# launch + host<->device round-trip overhead dominates at that size). When a caller passes ``n``/``p`` (its own
# call shape, not the fit-level row count), STRICT additionally requires BOTH ``n * p >= _STRICT_MIN_CALL_WORK``
# AND ``p >= _STRICT_MIN_CALL_P`` -- mirroring, exactly, the hand-tuned CPU/CUDA bootstrap crossover
# (``_cmi_cuda._should_use_cuda``'s ``n*p>=1_000_000 and p>=64`` fallback) so STRICT cannot force a trivially
# small dispatch onto the GPU just because the overall fit happened to be large (the ``p`` floor alone matters:
# a huge ``n`` with a tiny ``p``, e.g. 2 remaining late-round candidates, must still decline). Callers that do
# not pass shape (the other STRICT-forced dispatch points listed in this module's docstring) keep the
# fit-level-only gate unchanged.
_STRICT_MIN_CALL_WORK = 1_000_000
_STRICT_MIN_CALL_P = 64

# FIT-LEVEL column-aware AUTO relaxation (2026-07-11 fix). ``_DEFAULT_AUTO_MIN_N`` (100k) was a pure ROW-count
# threshold with no column term at all -- a fit with hundreds of columns (real production shape: mlframe's own
# wellbore benchmark runs a 79,237-row x 544-col fold, ~43M total elements, WAY past the point resident FE is
# worthwhile) still declined AUTO-STRICT purely for sitting under the 100k row mark, even though the SAME
# n*p/p-based work floor already used for per-call decisions (above) would clear by 40x+. The 100k row figure
# was never itself the validated safety boundary -- the safety claim is convergence "by ~50k" rows (see the
# module docstring); 100k was an EXTRA conservative margin on top of that, independent of column count. So a
# fit at/above the validated ~50k convergence floor that ALSO clears the SAME work floor a per-call dispatch
# would need is just as safe as one that happens to cross 100k rows outright -- this only ADDS an alternate,
# monotonic path to auto-engage (any fit that already qualified under the old n>=100k rule still qualifies
# unchanged; this never narrows engagement, only widens it for the wide-but-under-100k-rows case).
_MIN_N_FOR_CONVERGENCE = 50_000


def set_auto_fit_n(n: int | None, p: int | None = None) -> None:
    """Record the current fit's row/column counts so the AUTO (unset-env) STRICT default can size-gate on both.
    MRMR.fit sets this at entry and clears it in a finally; ``n=None`` disables the AUTO path entirely (leaving
    STRICT off unless env-forced). ``p`` is optional for backward compatibility with any other caller that only
    ever tracked row count -- omitting it just means the AUTO gate falls back to the pure row-count rule."""
    global _AUTO_FIT_N, _AUTO_FIT_P
    _AUTO_FIT_N = int(n) if n is not None else None
    _AUTO_FIT_P = int(p) if p is not None else None


def clear_auto_fit_n() -> None:
    """Reset the AUTO fit-row/column-count gate; called in MRMR.fit's finally so a subsequent non-MRMR caller doesn't inherit a stale shape."""
    global _AUTO_FIT_N, _AUTO_FIT_P
    _AUTO_FIT_N = None
    _AUTO_FIT_P = None


def _auto_min_n() -> int:
    """Row-count threshold above which the AUTO (unset-env) path engages STRICT GPU-resident FE regardless of column
    count; env-overridable via ``MLFRAME_FE_GPU_STRICT_AUTO_MIN_N``, falling back to ``_DEFAULT_AUTO_MIN_N`` on an
    unparsable value. See ``_auto_gate_passes`` for the column-aware alternate path below this floor."""
    try:
        return int(os.environ.get("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", _DEFAULT_AUTO_MIN_N))
    except (ValueError, TypeError):
        return _DEFAULT_AUTO_MIN_N


def _auto_gate_passes(fit_n: int | None, fit_p: int | None) -> bool:
    """Whether the fit-level AUTO gate engages: EITHER the pure row-count rule (n >= _auto_min_n(), unchanged from
    before this fix -- column count irrelevant once n alone is this large), OR -- new -- the fit is at/above the
    validated ~50k convergence floor AND its total (n, p) work already clears the SAME work floor a per-call
    dispatch would need (n*p >= _STRICT_MIN_CALL_WORK and p >= _STRICT_MIN_CALL_P). ``fit_p`` unavailable (older
    caller, or genuinely 1-D) falls back to the row-only rule -- never engages MORE eagerly for missing data."""
    if fit_n is None:
        return False
    if fit_n >= _auto_min_n():
        return True
    if fit_p is None:
        return False
    return fit_n >= _MIN_N_FOR_CONVERGENCE and (fit_n * fit_p) >= _STRICT_MIN_CALL_WORK and fit_p >= _STRICT_MIN_CALL_P


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


def fe_gpu_strict_enabled(*, n: int | None = None, p: int | None = None, min_p: int | None = None) -> bool:
    """Whether STRICT GPU mode is active. Three-state ``MLFRAME_FE_GPU_STRICT``:

    * ``1``/``true``/``on``/``yes`` -- force STRICT (subject to a usable CUDA device AND, when the caller passes
      ``n``/``p``, the per-call work floor below -- explicit force still should not blow past a trivially tiny call).
    * ``0``/``false``/``off``/``no`` -- force OFF (exact CPU path; byte-identical legacy).
    * unset / ``auto`` -- AUTO size-gated default: STRICT engages when the current fit clears ``_auto_gate_passes``
      (row count at/above ``MLFRAME_FE_GPU_STRICT_AUTO_MIN_N``, default 100k -- OR, since 2026-07-11, row count
      at/above the ~50k validated convergence floor AND the fit's total (row, column) work already clears the
      SAME work floor a per-call dispatch would need -- see ``_auto_gate_passes``) and a CUDA device is usable;
      otherwise it is OFF.

    ``n``/``p`` (optional): the CALLING dispatch's own shape (not the fit-level row count). When provided, STRICT
    additionally requires ``n * p >= _STRICT_MIN_CALL_WORK`` -- see that constant's docstring. Omit to preserve the
    legacy fit-level-only gate (existing call sites that have not been updated to pass shape).

    The env is read LIVE every call (mid-process toggles observed immediately); only the immutable CUDA-device probe
    is cached. No-op without CUDA. Small-n and no-GPU behavior is byte-identical to the pre-AUTO default."""
    def _passes_call_work_floor() -> bool:
        """Whether this call's (n, p) shape clears the total-work and per-column floors gating STRICT GPU use."""
        if n is None or p is None:
            return True  # no per-call shape given -> caller not yet shape-aware; fit-level gate alone decides
        # ``min_p`` overrides ONLY the p-floor leg: the default 64 mirrors the CANDIDATE-batch CPU/CUDA
        # crossover (p = candidate columns, each its own reduction). A PERMUTATION-null workload (p = nperm,
        # every permutation re-reading the SAME resident candidate) does not share that crossover -- holding
        # it to p >= 64 forced every nperm<=24 null onto the CPU for the whole fit (wellbore-100k GPU-strict
        # profile: 2232 _conditional_perm_null calls / ~261s on CPU). Such callers pass min_p=2; the n*p
        # total-work leg still binds either way.
        p_floor = _STRICT_MIN_CALL_P if min_p is None else int(min_p)
        return (int(n) * int(p)) >= _STRICT_MIN_CALL_WORK and int(p) >= p_floor

    raw = os.environ.get("MLFRAME_FE_GPU_STRICT", "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return bool(_cuda_usable()) and _passes_call_work_floor()
    if raw in ("", "auto"):
        if _auto_gate_passes(_AUTO_FIT_N, _AUTO_FIT_P):
            return bool(_cuda_usable()) and _passes_call_work_floor()
    return False
