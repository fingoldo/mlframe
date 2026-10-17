"""Helper blocks extracted from the over-long functions of ``_gpu_resident_pair_mi`` (function-length gate); behaviour is unchanged."""
from __future__ import annotations


import os
from typing import TYPE_CHECKING, Optional, Any

from mlframe.utils.log_throttle import log_throttle
import numpy as np

if TYPE_CHECKING:
    pass

# Parent-defined names these blocks consume. Imported at module top: the PARENT does
# ``from ._gpu_resident_pair_mi import *`` at its BOTTOM (after all these names are defined), so re-entering
# the partially-initialised parent here always finds them - no circular-import hazard.
from ._gpu_resident_shared import (
    _COMBOS,
    _candidate_names,
    _fused_generate_block,
    _gpu_k_chunk,
    _quantile_levels_dev,
    _unary_stack_cm,
)


def _grand_fused_pair_mi_part1(
    a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, classes_y_safe: np.ndarray, freqs_y: np.ndarray, nbins: int, use_su: bool, npermutations: int,
    random_seed: int, min_nonzero_confidence: float,
) -> tuple[Any, ...]:
    """Extracted block of ``grand_fused_pair_mi`` (part 1); returns the names the caller still uses."""
    import cupy as cp

    from ._gpu_resident_select import _gpu_resident_discretize_codes  # type: ignore[attr-defined]  # dynamically re-exported via globals(); lazy: cross-sibling, avoids cycle
    from . import hermite_fe as _hf  # noqa: F401 - full-init parent before the GPU MI import cycle
    from .batch_mi_noise_gate_gpu import (
        batch_mi_with_noise_gate_cuda_resident,
        dispatch_batch_mi_with_noise_gate_gpu,
    )

    # Intentional float64: operands at numpy's float64 op width, so device candidates and codes match the host bin_func path
    a_gpu = cp.asarray(a, dtype=cp.float64)
    # Intentional float64: operands at numpy's float64 op width, so device candidates and codes match the host bin_func path
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    csafe = np.ascontiguousarray(classes_y_safe)
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)  # Intentional float64: class/bin counts and frequencies feeding the MI formula
    k_chunk = _gpu_k_chunk(n)
    parts: list[np.ndarray] = []

    def _ensure_disc_host(cache: list[Optional[np.ndarray]], dev: np.ndarray) -> np.ndarray:
        """LAZY host materialisation of the resident disc codes ``dev`` - D2H'd (once, cached in
        ``cache``) only if a host-reading path actually needs them; the resident GPU gate (the common
        case) never triggers this, so it never touches the host for this data. ``cache`` is a fresh
        ``[None]`` per block (passed in, not captured, to keep this def loop-variable-safe)."""
        val = cache[0]
        if val is None:
            val = np.asarray(cp.asnumpy(dev))
            cache[0] = val
        return val

    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start : start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)  # GPU gen (resident)
        disc_dev = _gpu_resident_discretize_codes(cand, nbins).astype(cp.int8)  # GPU disc - KEPT resident
        del cand
        fnb = np.full(len(block), int(nbins), dtype=np.int64)
        _disc_host_cache: list[Optional[np.ndarray]] = [None]

        out = None
        if not bool(use_su):
            # RESIDENT noise-gate: feed the on-device disc codes straight into the resident
            # CUDA kernel via ``d_disc_resident`` - it reads them in place, so the codes never round-trip
            # GPU -> host -> GPU (the D2H-then-H2D this fix eliminates). ``disc_2d`` below is a metadata-
            # only placeholder (matching shape/dtype, uninitialised) - the resident kernel only reads its
            # ``.shape``/``.dtype`` when ``d_disc_resident`` is given (its own OOB-screen skips reading the
            # host bytes in that branch), so no data ever needs to be D2H'd for the common (successful) case.
            # ``batch_mi_with_noise_gate_cuda_resident`` does not support ``use_su=True`` (its own docstring
            # - SU has no GPU-entropy form), so that case is excluded here and falls through unchanged below.
            try:
                _placeholder = np.empty(disc_dev.shape, dtype=disc_dev.dtype)
                out = batch_mi_with_noise_gate_cuda_resident(
                    disc_2d=_placeholder,
                    factors_nbins=fnb,
                    classes_y=y_i64,
                    classes_y_safe=csafe,
                    freqs_y=fy,
                    npermutations=int(npermutations),
                    base_seed=np.uint64(random_seed),
                    min_nonzero_confidence=float(min_nonzero_confidence),
                    use_su=False,
                    dtype=np.int32,
                    d_disc_resident=disc_dev,
                )
            except Exception as _res_exc:
                import logging

                log_throttle(
                    logging.getLogger("mlframe.feature_selection.filters._gpu_resident_pair_mi"),
                    "gpu_pairs_mi.grand_fused_noise_gate",
                    logging.WARNING,
                    "grand_fused_pair_mi resident noise-gate failed (%s: %s); D2H fallback",
                    type(_res_exc).__name__,
                    _res_exc,
                )
                out = None
        # FORCE the GPU noise-gate: measured 15x faster than CPU here (K=384 n=200k: cupy 3093ms vs
        # CPU njit 46176ms - the noise-gate is the REAL bottleneck, not gen/discretize). The default
        # chooser (_batch_mi_noise_gate_backend_choice) picks CPU on this host - a tuner mis-calibration
        # (it under-rates the GPU gate, same class as the MI-dispatch issue); force cupy/cuda so the
        # grand-fused path actually runs the gate on the GPU. Falls back to CPU only if no GPU backend.
        # Reached when use_su=True (the resident kernel's excluded case, eager D2H below) or when the
        # resident attempt above failed/was skipped.
        if out is None:
            disc_host = _ensure_disc_host(_disc_host_cache, disc_dev)
            for _fb in ("cupy", "cuda"):
                gpu_res = dispatch_batch_mi_with_noise_gate_gpu(
                    disc_2d=disc_host,
                    factors_nbins=fnb,
                    classes_y=y_i64,
                    classes_y_safe=csafe,
                    freqs_y=fy,
                    npermutations=int(npermutations),
                    base_seed=np.uint64(random_seed),
                    min_nonzero_confidence=float(min_nonzero_confidence),
                    use_su=bool(use_su),
                    dtype=np.int32,
                    force_backend=_fb,
                )
                if gpu_res is not None:
                    out = gpu_res[0]
                    break
        if out is None:  # GPU noise-gate unavailable -> the always-correct CPU kernel on the same disc
            from ._feature_engineering_pairs._pairs_dispatch import _fe_classes_dtype
            from .info_theory import batch_mi_with_noise_gate as _cpu_gate

            disc_host = _ensure_disc_host(_disc_host_cache, disc_dev)
            fe_mi = _cpu_gate(
                disc_2d=disc_host,
                factors_nbins=fnb,
                classes_y=y_i64,
                classes_y_safe=csafe,
                freqs_y=fy,
                npermutations=int(npermutations),
                base_seed=np.uint64(random_seed),
                min_nonzero_confidence=float(min_nonzero_confidence),
                use_su=bool(use_su),
                # Widen rather than hardcode int16: a nbins > 32767 column would otherwise trip the kernel's
                # capacity clamp (or, before it existed, wrap silently into a negative bucket).
                dtype=np.int32,
                classes_dtype=_fe_classes_dtype(np.dtype(disc_host.dtype), fnb),
            )
        else:
            fe_mi = out[0] if isinstance(out, tuple) else out
        parts.append(np.asarray(fe_mi, dtype=np.float64))  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    return True, (_candidate_names(), np.concatenate(parts) if parts else np.empty(0))


def _grand_fused_pair_mi_fused_part1(
    cp: Any, a_gpu: Any, b_gpu: Any, y_codes: np.ndarray, n: Any, freqs_y: np.ndarray, npermutations: int, _build_shuffle_matrix: Any,
    classes_y_safe: np.ndarray, random_seed: int, nbins: int, use_su: bool, _mi_columns_from_counts_cpu: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``grand_fused_pair_mi_fused`` (part 1); returns the names the caller still uses."""
    from ._gpu_resident_pair_mi import _device_mi_kernel, _grand_fusion_block_counts
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)

    # y-vectors: row 0 = original y, rows 1.. = the Fisher-Yates shuffles (SAME host LCG the noise-gate
    # uses -> bit-identical permutation stream). Uploaded ONCE as (P1, n) int32 and shared by every chunk.
    y_orig = np.ascontiguousarray(y_codes, dtype=np.int64).reshape(1, n)
    K_y = int(np.asarray(freqs_y).shape[0])
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)  # Intentional float64: class/bin counts and frequencies feeding the MI formula
    nperm = int(npermutations) if npermutations and npermutations > 0 else 0
    if nperm > 0:
        shuf = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(random_seed), nperm)
        y_all_host = np.empty((nperm + 1, n), dtype=np.int64)
        y_all_host[0, :] = y_orig[0, :]
        y_all_host[1:, :] = shuf.astype(np.int64)
    else:
        y_all_host = y_orig
    P1 = int(y_all_host.shape[0])
    y_all_dev = cp.asarray(np.ascontiguousarray(y_all_host, dtype=np.int32))

    # The shared-mem histogram is (P1, nbins, K_y) int32 per block; it must fit this device's per-block
    # shared-memory limit. If not (very high nperm / many y-classes / large nbins), raise so the caller
    # falls back to the non-fused exact path - correctness over fusion. Reserve a little headroom.
    hist_bytes = P1 * int(nbins) * K_y * 4
    try:
        sm_limit = int(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["sharedMemPerBlock"])
    except Exception as e:
        import logging

        log_throttle(
            logging.getLogger("mlframe.feature_selection.filters._gpu_resident_pair_mi"),
            "gpu_pairs_mi.shared_mem_probe",
            logging.WARNING,
            "device sharedMemPerBlock probe failed, using the conservative 48KiB default: %s",
            e,
        )
        sm_limit = 48 * 1024
    if hist_bytes > sm_limit - 256:
        raise RuntimeError(
            f"grand-fusion shared histogram {hist_bytes}B exceeds device limit {sm_limit}B "
            f"(P1={P1}, nbins={nbins}, K_y={K_y}); caller falls back to the non-fused path"
        )

    # Binning working dtype: mirror _gpu_resident_discretize_codes (native f64 here -> bit-identical edges
    # to the non-fused grand-fusion path; MLFRAME_FE_GPU_BINNING_DTYPE=float32 forces f32 percentile).
    forced = os.environ.get("MLFRAME_FE_GPU_BINNING_DTYPE", "").strip().lower()
    # Intentional float64: bin-edge / cut vector, compared against values at full precision
    work = cp.float32 if forced in ("float32", "f32", "single") else cp.float64
    qs = _quantile_levels_dev(cp, nbins, work)

    k_chunk = _gpu_k_chunk(n)
    original_mi_parts: list[np.ndarray] = []
    perm_mi_parts: list[list[np.ndarray]] = [[] for _ in range(nperm)]
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start : start + k_chunk]
        blk = len(block)
        # PASS 1: transient generate -> exact percentile edges -> discard the float matrix. Generate directly
        # COLUMN-MAJOR (blk, n): the ONLY consumer here is the per-candidate percentile, and cp.percentile over
        # a (blk, n) C-contiguous buffer along the CONTIGUOUS axis=1 sorts each candidate's n values coalesced
        # - vs axis=0 over the (n, blk) row-major matrix, which sorts the STRIDED axis (cupy copies/transposes
        # it internally). Same value set per candidate -> BIT-IDENTICAL edges (and the (blk, nbins-1) edges_int
        # below is unchanged). cand is discarded straight after, so no (n, K) consumer sees the flipped layout.
        cand = _fused_generate_block(ua_cm, ub_cm, block, column_major=True)  # (blk, n) f64, transient
        if cand.dtype != work:
            cand = cand.astype(work, copy=False)
        if blk == 1:
            # cupy single-column percentile bug guard (mirror _gpu_resident_discretize_codes).
            bin_edges = cp.percentile(cand.ravel(), qs).reshape(-1, 1)  # (nbins+1, 1)
        else:
            bin_edges = cp.percentile(cand, qs, axis=1)  # (nbins+1, blk) over the (blk, n) plane
        del cand  # (n,blk) float GONE before the hist pass
        # interior edges, transposed to (blk, nbins-1) row-major f64 for the kernel's per-candidate scan.
        # Intentional float64: bin-edge / cut vector, compared against values at full precision
        edges_int = cp.ascontiguousarray(bin_edges[1:-1, :].T.astype(cp.float64))
        del bin_edges
        total_size = blk * int(nbins) * K_y
        mi_kernel = _device_mi_kernel() if not use_su else None
        if mi_kernel is not None:
            # Device path: the joint histograms stay on the GPU, the per-(p, column) MI comes from the CUDA kernel, and only the
            # (P1, blk) MI matrix returns to the host (the host CPU reduction below moved every histogram across the bus).
            counts_dev = _grand_fusion_block_counts(ua_cm, ub_cm, block, edges_int, y_all_dev, nbins, K_y, total_size, device_out=True)
            del edges_int
            P1 = nperm + 1
            col_off_d = cp.asarray(np.arange(blk, dtype=np.int64) * (int(nbins) * K_y))
            nbins_d = cp.full(blk, int(nbins), dtype=cp.int32)
            fy_d = cp.asarray(fy, dtype=cp.float64)  # Intentional float64: class/bin counts and frequencies feeding the MI formula
            counts_flat = counts_dev.reshape(-1)
            out_mi = cp.empty((P1, blk), dtype=cp.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
            threads = 128
            grid = (P1 * blk + threads - 1) // threads
            ones_d = cp.ones(blk, dtype=cp.float64)  # Intentional float64: class/bin counts and frequencies feeding the MI formula
            mi_kernel[grid, threads](counts_flat, col_off_d, nbins_d, fy_d, np.int64(n), np.int32(K_y), ones_d, np.int64(total_size), np.int32(P1), out_mi)
            om_d = out_mi[0].copy()
            mi_kernel[grid, threads](counts_flat, col_off_d, nbins_d, fy_d, np.int64(n), np.int32(K_y), om_d, np.int64(total_size), np.int32(P1), out_mi)
            del counts_dev, counts_flat
            om = cp.asnumpy(om_d)
            perm_host = cp.asnumpy(out_mi)
            original_mi_parts.append(om)
            for p in range(nperm):
                perm_mi_parts[p].append(perm_host[p + 1])
            continue
        counts = _grand_fusion_block_counts(ua_cm, ub_cm, block, edges_int, y_all_dev, nbins, K_y, total_size)
        del edges_int
        # CPU bit-exact reduction (identical to batch_mi_with_noise_gate_cupy). Use the BATCHED njit
        # _mi_columns_from_counts_cpu (one compiled call over all blk columns) instead of the per-(perm,k)
        # Python->njit dispatch - bit-identical (same _mi_from_counts_cpu body per column) but removes the
        # blk*(nperm+1) Python-call overhead. ref_mi reproduces the perm-skip: all-positive for the original
        # row (compute every column), then `om` for each perm row (compute only where om>0; else stays 0).
        nb_ky = int(nbins) * K_y
        _col_off = np.arange(blk, dtype=np.int64) * nb_ky
        _nbins_arr = np.full(blk, int(nbins), dtype=np.int64)
        _all_pos = np.ones(blk, dtype=np.float64)  # Intentional float64: class/bin counts and frequencies feeding the MI formula
        om = _mi_columns_from_counts_cpu(
            np.ascontiguousarray(counts[0]),
            _col_off,
            _nbins_arr,
            K_y,
            fy,
            n,
            bool(use_su),
            _all_pos,
        )
        original_mi_parts.append(om)
        for p in range(nperm):
            mp = _mi_columns_from_counts_cpu(
                np.ascontiguousarray(counts[p + 1]),
                _col_off,
                _nbins_arr,
                K_y,
                fy,
                n,
                bool(use_su),
                om,
            )
            perm_mi_parts[p].append(mp)
        del counts
    return nperm, original_mi_parts, perm_mi_parts
