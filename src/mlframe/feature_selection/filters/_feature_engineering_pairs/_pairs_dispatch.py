"""CPU/GPU dispatch for the batched FE-candidate MI + permutation noise-gate."""
from __future__ import annotations

import os

import numpy as np

from ._pairs_common import _module_logger


# Code-version for the per-host CPU-vs-GPU split of the batched FE-candidate
# MI+permutation noise-gate, keyed on the work size (n_rows x n_cols). The CPU
# njit-prange kernel wins for small/medium batches (its joint-hist pass is tiny and
# the H2D copy of the discretized frame + per-shuffle launch overhead would dominate
# on GPU); GPU pays off only for very large K at large n. The CPU/GPU backend is
# resolved per-host via the canonical ``KernelTuningCache.get_or_tune`` orchestrator
# (no hardcoded threshold). This is computed over the participating CPU + GPU fns via
# ``compute_code_version`` so a kernel-numerics edit invalidates stale per-host cache
# entries automatically; falls back to a static string when the GPU module / pyutilz
# code-versioning is unavailable.
try:  # GPU twin owns the canonical code_version (covers CPU + cupy + cuda bodies).
    from ..batch_mi_noise_gate_gpu import _batch_mi_noise_gate_code_version as _bming_code_version
    _BATCH_MI_NOISE_GATE_CODE_VERSION = _bming_code_version() or "batch_mi_noise_gate-v2"
except Exception:
    _BATCH_MI_NOISE_GATE_CODE_VERSION = "batch_mi_noise_gate-v2"


def _dispatch_batch_mi_with_noise_gate(
    disc_2d: np.ndarray,
    quantization_nbins: int,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    min_nonzero_confidence: float,
    use_su: bool,
    batch_mi_kernel,
) -> np.ndarray:
    """Route the batched FE-candidate MI + permutation noise-gate to the CPU njit kernel
    or (best-effort) a GPU batched path, by work size ``n * K`` via the per-host
    kernel_tuning_cache. The CPU kernel is the required win and the always-correct
    fallback; the GPU branch is gated behind ``is_cuda_available`` + try/except so any
    failure transparently falls back to CPU (mirrors ``mi_direct``'s GPU fastpath).

    Returns ``fe_mi[K]`` -- the per-column observed MI, zeroed where the permutation gate
    rejects. Bit-identical to a per-candidate ``mi_direct`` loop on the default FE path.
    """
    n = disc_2d.shape[0]
    K = disc_2d.shape[1]
    factors_nbins = np.full(K, int(quantization_nbins), dtype=np.int64)

    # F2 (2026-06-22): route the CPU njit observed-MI kernel to the fused v2 (one n-row pass per column
    # instead of two -- bit-identical, measured 1.18-1.21x at the canonical 30k chunk) via the per-host
    # kernel_tuning_cache. The caller passes the v1 kernel as ``batch_mi_kernel``; the selector swaps it
    # for v2 when the KTC (or the v2-default fallback) picks it. Every CPU njit call below (analytic
    # npermutations=0 observed-MI, GPU-opt-out, the always-correct fallback) uses ``_cpu_kernel``. The
    # GPU twin is unaffected (it has its own device kernels). Failure -> original ``batch_mi_kernel``.
    _cpu_kernel = batch_mi_kernel
    try:
        from ..info_theory._batch_kernels import select_batch_mi_kernel
        _cpu_kernel = select_batch_mi_kernel(int(n), int(K))
    except Exception:
        _cpu_kernel = batch_mi_kernel

    # RESIDENT-CODES HANDOFF (gated, default OFF): if the FE chunk binned these codes ON the GPU and kept
    # them resident, pop the device codes array keyed on THIS ``disc_2d`` (same object flows producer ->
    # here). When the resident-CUDA gate is the chosen consumer it reads them IN PLACE, skipping the codes'
    # H2D re-upload (a pointless GPU->host->GPU round-trip). ALWAYS pop (single-use, self-clearing) so a
    # stale entry can never satisfy a later dispatch; non-resident branches just ignore it and read host
    # ``disc_2d``. ``device_codes`` is None unless the gate is on AND the producer stashed a match.
    device_codes = None
    _ensure_host = None  # callable: materialise the (deferred) host codes into disc_2d, idempotent
    try:
        from .._gpu_resident_fe import (
            fe_gpu_resident_codes_enabled, take_resident_codes, ensure_host_codes_filled,
        )
        _ensure_host = ensure_host_codes_filled
        if fe_gpu_resident_codes_enabled():
            device_codes = take_resident_codes(disc_2d)
    except Exception:
        device_codes = None
        _ensure_host = None

    def _need_host_codes():
        """Materialise the host codes into ``disc_2d`` before any host-codes read (analytic gate / CPU
        njit / non-resident GPU path). No-op unless the producer DEFERRED the codes D2H (resident handoff
        on AND no host consumer has read yet) -- in which case it D2Hs the resident device codes into
        ``disc_2d`` now (the EXACT bytes the eager fill produced -> selection unchanged). NOT swallowed: a
        fill failure must NOT let a host consumer read the unfilled (garbage) ``disc_2d`` -- it propagates
        so the GPU branch's try/except routes to the always-correct CPU kernel (which re-calls this; if the
        D2H is genuinely broken the whole GPU stack is, and a loud error beats silent wrong selection)."""
        if _ensure_host is not None:
            _ensure_host(disc_2d)

    # ---- Analytic large-n noise gate (2026-06-16) -------------------------------------------------
    # The per-candidate permutation null -- CPU prange shuffles AND the GPU cupy-argsort twin below --
    # is the dominant large-n FE-scan cost. At large n the gate's permutation p-value is the analytic
    # G-test tail (2N*MI ~ chi2), so the shuffles are unnecessary: compute the ungated observed MI once
    # (the CPU kernel with npermutations=0) and apply the keep/reject decision analytically. Only when
    # MI is raw (not SU-normalised -- the chi2 identity requires it) and n >= threshold; below it the
    # permutation path (CPU/GPU) runs byte-for-byte unchanged. Bypasses the GPU branch entirely.
    if int(npermutations) > 0 and not use_su:
        try:
            from .._analytic_mi_null import (
                analytic_batch_noise_gate, analytic_null_enabled, analytic_null_applicable,
            )
            # Candidates are quantised to ``quantization_nbins`` (low, fixed cardinality); each column's
            # OCCUPIED bins are <= that, so checking applicability with the declared count + occupied y
            # bins is conservative (worst-case sparsest table) -- if it passes here it passes per column.
            # NOTE (P1-6): this APPLICABILITY check uses the DECLARED nbins (worst-case densest df ->
            # smallest n/cells -> hardest to pass, i.e. safe direction), while ``analytic_batch_noise_gate``
            # later uses each column's OCCUPIED bx for the actual chi2 df. The two intentionally differ:
            # the gate-on check is the conservative bound, the per-column df is the exact one.
            _by_occ = int(np.unique(np.asarray(classes_y)).size)
            _an_ok = analytic_null_enabled() and analytic_null_applicable(
                int(n), int(quantization_nbins), _by_occ,
            )
        except Exception:
            _an_ok = False
        if _an_ok:
            try:
                _need_host_codes()  # analytic gate reads host codes -> materialise the deferred D2H now
                _observed = _cpu_kernel(
                    disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
                    classes_y_safe=classes_y_safe, freqs_y=freqs_y, npermutations=0,
                    base_seed=np.uint64(0), min_nonzero_confidence=float(min_nonzero_confidence),
                    use_su=False, dtype=np.int32,
                    classes_dtype=disc_2d.dtype if disc_2d.dtype.itemsize <= 2 else np.int16,
                )
                return analytic_batch_noise_gate(
                    disc_2d, _observed, classes_y, int(n), float(min_nonzero_confidence),
                )
            except Exception as _an_exc:  # any failure -> fall through to the permutation path
                _module_logger.debug(
                    "analytic noise gate failed (%s: %s); permutation fallback",
                    type(_an_exc).__name__, _an_exc,
                )

    # Per-host CPU/GPU backend choice via the canonical ``get_or_tune`` orchestrator
    # (per-host cache, code-version checked -> on-miss tuner -> measurement-backed
    # fallback; no hardcoded threshold). The GPU twin (``batch_mi_noise_gate_gpu``)
    # is bit-identical (GPU does only the integer counting; entropy stays on the CPU
    # bit-exact path), so the tuner runs a real CPU-vs-GPU sweep and the cache routes
    # large (n_rows x n_cols) batches to GPU where it measurably wins on this host.
    backend = "cpu"
    # Explicit GPU OPT-OUT honoured BEFORE the KTC lookup. ``CUDA_VISIBLE_DEVICES=""`` (empty string -- the documented mlframe
    # convention for "no GPU on this run") must force CPU here: cupy ignores that env for device enumeration on some builds, so a
    # cached GPU ``backend_choice`` would route to the cupy path whose device->host copy then HANGS indefinitely (not an exception,
    # so the GPU try/except below never catches it). Skip the GPU route entirely when the user asked for no CUDA device.
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    _gpu_opted_out = (_cvd is not None and _cvd.strip() == "") or os.environ.get("MLFRAME_DISABLE_GPU", "") == "1"
    if _gpu_opted_out:
        _need_host_codes()  # CPU njit kernel reads host codes -> materialise the deferred D2H now
        return _cpu_kernel(
            disc_2d=disc_2d,
            factors_nbins=factors_nbins,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            npermutations=int(npermutations),
            base_seed=np.uint64(0),
            min_nonzero_confidence=float(min_nonzero_confidence),
            use_su=bool(use_su),
            dtype=np.int32,
            classes_dtype=disc_2d.dtype if disc_2d.dtype.itemsize <= 2 else np.int16,
        )
    # Under an explicit max_runtime_mins budget, skip the CPU-vs-GPU crossover sweep (blocking on first use, tens of
    # seconds at large n) and use the measurement-backed fallback; the sweep still runs on a normal no-budget fit so
    # per-host tuning is unaffected. Checked BEFORE the get_or_tune so the budgeted fit never pays the sweep.
    _budget_active = False
    try:
        from .._fe_deadline import fe_budget_active
        _budget_active = fe_budget_active()
    except Exception:
        _budget_active = False
    if _budget_active:
        try:
            from ..batch_mi_noise_gate_gpu import _batch_mi_noise_gate_fallback_choice
            _fb = _batch_mi_noise_gate_fallback_choice(int(n), int(K))
            backend = str(_fb.get("backend_choice", "cpu")) if isinstance(_fb, dict) else "cpu"
        except Exception:
            backend = "cpu"
    else:
        try:
            from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

            from ..batch_mi_noise_gate_gpu import (
                _run_batch_mi_noise_gate_sweep,
                _batch_mi_noise_gate_fallback_choice,
            )

            # load_or_create() returns the MEMOIZED per-host cache singleton (~0ms/call); a bare
            # KernelTuningCache() ctor re-loads cache state from disk EVERY call (~0.75ms), which
            # on a wide near-saturated frame (thousands of FE raw-pairs, one dispatch each) added
            # seconds of pure overhead. This dispatcher is on the per-raw-pair hot path -> singleton.
            _res = KernelTuningCache.load_or_create().get_or_tune(
                "batch_mi_noise_gate",
                dims={"n_rows": int(n), "n_cols": int(K)},
                tuner=_run_batch_mi_noise_gate_sweep,  # real CPU-vs-GPU sweep (bit-identical GPU)
                axes=["n_rows", "n_cols"],
                fallback={"backend_choice": _batch_mi_noise_gate_fallback_choice(int(n), int(K))},
                code_version=_BATCH_MI_NOISE_GATE_CODE_VERSION,
                async_sweep=True,  # FIT-TIME: never block the FE pair-search on the sweep; measure in the background
            )
            if isinstance(_res, str):
                backend = _res
            elif _res:
                backend = str(_res.get("backend_choice", "cpu"))
        except Exception:  # pyutilz missing / cache error -> CPU (always correct)
            backend = "cpu"

    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, diagnostic, default OFF): force the noise-gate MI onto the
    # bit-identical GPU twin (GPU does the integer counting, entropy stays on the CPU bit-exact path ->
    # selection-equivalent), past the KTC crossover. Honoured AFTER the explicit GPU opt-out above (which
    # still wins); a no-op without CUDA. Any GPU failure still falls through to the CPU njit kernel below.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            backend = "gpu"
    except Exception:
        pass

    # GPU region: route to the bit-identical GPU twin. Any failure (cupy/cuda
    # unavailable, OOM, shape edge) returns None / raises and falls through to the
    # always-correct CPU njit kernel below (mirrors mi_direct's GPU fastpath).
    if backend in ("gpu", "cupy", "cuda"):
        try:
            _res = _batch_mi_with_noise_gate_gpu(
                disc_2d=disc_2d,
                factors_nbins=factors_nbins,
                classes_y=classes_y,
                classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                npermutations=npermutations,
                min_nonzero_confidence=min_nonzero_confidence,
                use_su=use_su,
                force_backend=(backend if backend in ("cupy", "cuda") else None),
                device_codes=device_codes,
                ensure_host_codes=_need_host_codes,
            )
            if _res is not None:
                return _res
        except Exception as _exc:  # pragma: no cover - GPU optional
            _module_logger.debug(
                "batch_mi_with_noise_gate GPU path failed (%s: %s); CPU fallback",
                type(_exc).__name__, _exc,
            )

    # CPU njit-prange kernel (the required win and always-correct fallback).
    _need_host_codes()  # CPU kernel reads host codes -> materialise the deferred D2H now
    return _cpu_kernel(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=int(npermutations),
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=bool(use_su),
        dtype=np.int32,
        # OPT-B: size the (n, K) densified-code buffer to disc_2d's (now narrow) width -- the
        # dense codes live in the SAME [0, n_bins) range, so int8/int16 is value-identical and
        # cuts both the alloc (the 589MiB->147MiB classes_dense that OOM'd RAM-tight hosts) and
        # the per-permutation strided gather bandwidth. joint_counts (the real counter) stays int32.
        classes_dtype=disc_2d.dtype if disc_2d.dtype.itemsize <= 2 else np.int16,
    )


def _batch_mi_with_noise_gate_gpu(
    disc_2d,
    factors_nbins,
    classes_y,
    classes_y_safe,
    freqs_y,
    npermutations,
    min_nonzero_confidence,
    use_su,
    force_backend=None,
    device_codes=None,
    ensure_host_codes=None,
):
    """Bit-identical GPU twin of ``batch_mi_with_noise_gate``; returns ``fe_mi[K]``
    when a GPU backend is available + chosen, else ``None`` (-> CPU fallback).

    Delegates to ``batch_mi_noise_gate_gpu.dispatch_batch_mi_with_noise_gate_gpu``,
    which runs the cupy / numba.cuda joint-histogram counting on the GPU and the
    entropy reduction on the bit-exact CPU path (the y-shuffles use the IDENTICAL
    CPU LCG/Fisher-Yates stream, ``base_seed*2654435761 + (i+1)`` then the PCG
    step). ``base_seed`` is 0 on the default FE path -- matching the CPU kernel call
    below -- so the GPU and CPU shuffle streams (and thus the noise-gate rejection)
    are identical to the bit.
    """
    # FULL-GPU-RESIDENT gate (2026-06-21): once the cache has chosen GPU for this size, run the WHOLE
    # noise gate on-device (batched histogram + GPU entropy, only the (P,K) MI matrix D2H) -- bit-
    # identical to the GPU-hist+CPU-entropy path (verified maxdiff ~1e-18, FE pins green) and ~1.14x
    # faster at the canonical K. SU has no GPU-entropy form -> use the standard path. Any failure falls
    # through. Opt-out: MLFRAME_FE_GPU_RESIDENT_GATE=0.
    if not bool(use_su) and os.environ.get("MLFRAME_FE_GPU_RESIDENT_GATE", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            from ..batch_mi_noise_gate_gpu import batch_mi_with_noise_gate_cuda_resident, _CUDA_AVAIL as _CA
            # Use the resident CUDA gate whenever GPU is chosen -- INCLUDING when the cache/fallback said
            # "cupy": the cupy noise gate OOMs on a 4 GB consumer GPU (its (rows, n*K) tiled buffer), which
            # silently falls back to the CPU njit gate (~3x slower); the resident CUDA path is bit-
            # identical, never builds that buffer, and doesn't OOM. The cache's backend pick is GPU-vs-CPU;
            # the GPU SUB-backend is ours to choose, and resident-cuda dominates cupy here.
            if _CA:
                # RESIDENT-CODES HANDOFF (gated): feed the on-device codes (when the producer kept them
                # resident) straight to the histogram kernel -- skips the codes' H2D re-upload, bit-
                # identical (same int codes). ``device_codes=None`` keeps the H2D-from-host path.
                # When the device codes are consumed in place, host ``disc_2d`` is never read -> the
                # producer's DEFERRED host-codes D2H is skipped entirely (the canonical win). If
                # ``device_codes`` is None the resident gate H2Ds host ``disc_2d`` instead, so the deferred
                # host buffer MUST be materialised first (no-op when not deferred).
                if device_codes is None and ensure_host_codes is not None:
                    ensure_host_codes()
                return batch_mi_with_noise_gate_cuda_resident(
                    disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
                    classes_y_safe=classes_y_safe, freqs_y=freqs_y, npermutations=int(npermutations),
                    base_seed=np.uint64(0), min_nonzero_confidence=float(min_nonzero_confidence),
                    use_su=False, d_disc_resident=device_codes,
                )
        except Exception as _exc:
            _module_logger.debug("resident gate failed (%s: %s); standard GPU path", type(_exc).__name__, _exc)
    # Standard (non-resident) GPU path H2Ds host ``disc_2d`` -> materialise the deferred host buffer first.
    if ensure_host_codes is not None:
        ensure_host_codes()
    try:
        from ..batch_mi_noise_gate_gpu import dispatch_batch_mi_with_noise_gate_gpu
    except Exception:
        return None
    _out = dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=int(npermutations),
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=bool(use_su),
        dtype=np.int32,
        force_backend=force_backend,
    )
    if _out is None:
        return None
    fe_mi, _backend_name = _out
    return fe_mi
