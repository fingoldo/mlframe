"""CPU/GPU dispatch for the batched FE-candidate MI + permutation noise-gate."""
from __future__ import annotations

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

    # Per-host CPU/GPU backend choice via the canonical ``get_or_tune`` orchestrator
    # (per-host cache, code-version checked -> on-miss tuner -> measurement-backed
    # fallback; no hardcoded threshold). The GPU twin (``batch_mi_noise_gate_gpu``)
    # is bit-identical (GPU does only the integer counting; entropy stays on the CPU
    # bit-exact path), so the tuner runs a real CPU-vs-GPU sweep and the cache routes
    # large (n_rows x n_cols) batches to GPU where it measurably wins on this host.
    backend = "cpu"
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
            )
            if _res is not None:
                return _res
        except Exception as _exc:  # pragma: no cover - GPU optional
            _module_logger.debug(
                "batch_mi_with_noise_gate GPU path failed (%s: %s); CPU fallback",
                type(_exc).__name__, _exc,
            )

    # CPU njit-prange kernel (the required win and always-correct fallback).
    return batch_mi_kernel(
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
        classes_dtype=disc_2d.dtype if disc_2d.dtype.itemsize <= 4 else np.int32,
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
