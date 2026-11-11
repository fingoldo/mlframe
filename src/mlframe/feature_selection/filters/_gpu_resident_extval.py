"""Device-born external-validation candidate materialise + discretise (Phase-1 residency).

The external-validation MI sweep in ``_emit_pair_features`` builds ``out[:, e*n_ops+o] = op_o(param_a, ext_e)``
for every (external_factor x binary_op) candidate on the HOST (``_materialise_extval_njit``, double) and then
uploads the whole ``(n, K)`` float buffer to ``gpu_discretize_codes_host`` for binning - a bulk H2D of the
candidate VALUES (measured ~46 MB on the F2 1M/30k STRICT-resident fit). This module builds the SAME candidate
matrix RESIDENT on the device (double, matching the njit arithmetic op-for-op) and quantile-bins it resident,
so ONLY ``param_a`` + the external-factor columns cross H2D (once, small), never the ``(n, K)`` matrix.

SELECTION-EQUIVALENCE: the device ops are the exact double numpy bin_func semantics (see ``_extval_op_gpu``);
NaN/inf are NOT scrubbed - they flow into the resident binner exactly as the CPU path's ``nanpercentile``
(NaN-ignoring edges) + ``searchsorted`` (NaN/inf -> rightmost bin) handle them (``_gpu_resident_discretize_codes``
is the same binner ``gpu_discretize_codes_host`` uses, verified bit-identical to ``discretize_2d_quantile_batch``).
So the codes match the host path except for GPU FP reduction order in the percentile edges (~1e-15, below the
equi-frequency binner's boundary tolerance). Any cupy/device fault returns ``None`` -> the caller keeps the exact
host njit + upload path (correctness first; the device path is a residency optimisation, never a correctness risk).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

import numpy as np

from ._fe_usability_signal import _crit_np_dtype

from mlframe.utils.env_flags import env_flag


def _extval_op_gpu(cp, op: int, a, b):
    """One binary op on device in double, byte-faithful to ``_materialise_extval_njit`` (``_NJIT_BINARY_OP_CODES``:
    0=mul 1=add 2=sub 3=div 4=max 5=min 6=abs_diff 7=signed 8=ratio_abs). No NaN scrub: NaN/inf propagate exactly
    as the numpy bin_funcs produce them (the resident binner routes them to the rightmost bin, as searchsorted
    does on the CPU)."""
    if op == 0:  # mul
        return a * b
    if op == 1:  # add
        return a + b
    if op == 2:  # sub
        return a - b
    if op == 3:  # div = _safe_div: exact x/y for y != 0, eps floor only on an exact-zero denominator
        return a / cp.where(b == 0.0, cp.asarray(1e-9, dtype=a.dtype), b)
    if op == 4:  # max = np.maximum (nan-propagating)
        return cp.maximum(a, b)
    if op == 5:  # min = np.minimum (nan-propagating)
        return cp.minimum(a, b)
    if op == 6:  # abs_diff = |a - b|
        return cp.abs(a - b)
    if op == 7:  # signed = sign(a) * |b| (nan-propagating; sign(0)=0)
        return cp.sign(a) * cp.abs(b)
    # op == 8: ratio_abs = a / (|b| + 1)
    return a / (cp.abs(b) + 1.0)


def _candidate_matrix(cp, pa, param_b_list, ops, n, n_ops, resident_operand):
    """The ``(n, n_ext * n_ops)`` device matrix (crit dtype) of ``op_o(pa, ext_e)`` (ext-outer / op-inner), or ``None`` when an external column has the wrong length.

    The fused kernel computes all cells in one launch (``MLFRAME_EXTVAL_FUSED=0`` selects the per-column cupy loop).

    bench-attempt-rejected (2026-08-03): batching across ext factors via one (n, n_ext) stacked matrix + broadcasting (pa[:, None] op B) per op -- reducing
    n_ext*n_ops single-column kernel launches to n_ops broadcast launches -- was measured 0.85x (SLOWER) on n=1.62M/n_ext=20/n_ops=9 (2529ms -> 2967ms, warm,
    best-of-10): the stack itself costs n_ext column-assignment launches and the broadcast op does more elementwise work per launch. The fused kernel avoids both.
    """
    bs = []
    for _pb in param_b_list:
        b = resident_operand(_pb, ("extval_ext",), dtype=pa.dtype).ravel()  # the crit dtype of the candidate matrix (float32 by default)
        if b.shape[0] != n:
            return None
        bs.append(b)
    if env_flag("MLFRAME_EXTVAL_FUSED", True):
        from ._gpu_resident_extval_fused import fused_extval_candidates

        return fused_extval_candidates(cp, pa, cp.stack(bs), cp.asarray(ops, dtype=cp.int32))
    out_dev = cp.empty((n, len(bs) * n_ops), dtype=pa.dtype)
    for _e, b in enumerate(bs):
        _base = _e * n_ops
        for _o in range(n_ops):
            out_dev[:, _base + _o] = _extval_op_gpu(cp, int(ops[_o]), pa, b)
    return out_dev


def gpu_materialise_extval_codes_host(
    param_a: np.ndarray, param_b_list: Sequence[np.ndarray], op_codes: np.ndarray, nbins: int,
    *, dtype: Any = np.int8, defer_host_fill: bool = False, param_a_dev: Any = None,
) -> Optional[np.ndarray]:
    """Device twin of ``_materialise_extval_njit`` + discretise. Materialises ``out[:, e*n_ops+o] =
    op_o(param_a, param_b_list[e])`` RESIDENT in double (ext-outer/op-inner order, matching the njit column
    layout), quantile-bins it resident, and returns the ``(n, K)`` host codes of ``dtype``. The candidate VALUES
    never cross H2D - only ``param_a`` + the external-factor columns upload (once, small). Returns ``None`` on
    any cupy/device fault (caller keeps the host path). ``op_codes`` are the ``_NJIT_BINARY_OP_CODES`` in
    bin_func-registry order (the SAME array the njit kernel walks). ``defer_host_fill=True`` (the caller feeds the codes straight to the resident
    noise gate) returns an unfilled host buffer with the device codes registered for in-place consumption instead of copying them back. ``param_a_dev`` is ``param_a`` already on the device (``param_a`` itself may then be ``None``), which skips the copy back
    and re-upload of the survivor column."""
    # Codes live in [0, nbins-1]; the default int8 is C signed char (-128..127) so nbins>128 wraps the top
    # bins negative. A dtype/nbins contract violation is a programming error, not a device fault - raise it
    # BEFORE the cupy try (which swallows to None) so it surfaces instead of masquerading as a GPU-miss.
    _cd = np.dtype(dtype)
    if np.issubdtype(_cd, np.integer) and np.iinfo(_cd).max < int(nbins) - 1:
        raise ValueError(f"dtype {_cd} cannot represent codes up to nbins-1={int(nbins) - 1} (max={np.iinfo(_cd).max})")
    try:
        import cupy as cp
        from ._gpu_resident_discretize import _gpu_resident_discretize_codes  # same binner as gpu_discretize_codes_host
        from ._fe_resident_operands import resident_operand  # content-keyed device cache (dedups repeated operands)
        from ._gpu_resident_fe import (
            _stash_deferred_host_fill,
            _stash_resident_codes,
            fe_gpu_defer_host_codes_enabled,
            fe_gpu_resident_codes_enabled,
        )
    except ImportError as _exc:
        logger.debug("gpu_materialise_extval_codes_host fell back to its default after %s", type(_exc).__name__)
        return None
    try:
        # param_a + each external-factor column ride the content-keyed resident cache: the SAME ext-factor
        # content reused across pairs uploads ONCE (dedup), and each is a f64 device array here.
        _real = _crit_np_dtype()
        pa = resident_operand(param_a, ("extval_pa",), dtype=_real).ravel() if param_a_dev is None else param_a_dev.astype(_real, copy=False).ravel()
        n = int(pa.shape[0])
        _ops = np.asarray(op_codes).ravel()
        n_ops = int(_ops.shape[0])
        n_ext = len(param_b_list)
        if n == 0 or n_ops == 0 or n_ext == 0:
            return None
        K = n_ext * n_ops
        # Cushion-check BEFORE the (n, K) candidate matrix: without it, a device pool already pinned at its
        # limit (e.g. by the resident-operand cache) fails this call's allocation on EVERY (base, external-
        # factor-set) pair with the SAME OutOfMemoryError, each attempt paying cupy's own allocator work for a
        # request that cannot succeed until something else frees VRAM (the exact pattern fixed in
        # perm_null_residue_mis_resident, GPU11-13). One cheap memGetInfo here skips straight to the host path.
        from ._fe_gpu_vram import fe_gpu_has_vram_cushion

        if not fe_gpu_has_vram_cushion(n * K * np.dtype(_real).itemsize):
            return None
        out_dev = _candidate_matrix(cp, pa, param_b_list, _ops, n, n_ops, resident_operand)
        if out_dev is None:
            return None
        if bool(cp.isfinite(out_dev).all()):
            codes_dev = _gpu_resident_discretize_codes(out_dev, int(nbins))
        else:
            # The host binner takes the quantile edges over the non-NaN rows of a column; the finite-path device binner takes them over all rows (NaN sorted last), which turns upper edges NaN and gives a different contingency
            # table. A NaN-aware device binner reproduces the host codes exactly (a missing external factor is the usual cause); an infinite value or a column with fewer than two numbers still goes to the host binner.
            from ._gpu_nan_quantile_codes import quantile_codes_with_nan_columns

            codes_dev = quantile_codes_with_nan_columns(out_dev, int(nbins))
            if codes_dev is None:
                log_throttle(logger, "extval_nonfinite_candidates", logging.INFO, "gpu_materialise_extval_codes_host: infinite or near-empty candidate columns (n=%d, K=%d); using the host binner", n, K)
                return None
        codes_dev = codes_dev.astype(cp.dtype(_cd), copy=False) if codes_dev.dtype != _cd else codes_dev
        if defer_host_fill and fe_gpu_resident_codes_enabled() and fe_gpu_defer_host_codes_enabled():
            # The codes stay RESIDENT: the noise gate consumes the device copy in place (take_resident_codes) and never reads the host array, so the
            # (n, K) D2H is skipped. ``out`` is UNFILLED; a host-reading consumer (analytic / CPU / non-resident gate) fills it on demand via
            # ensure_host_codes_filled, with exactly the bytes the eager D2H below would have produced.
            out = np.empty((n, K), dtype=_cd)
            _stash_resident_codes(out, codes_dev)
            _stash_deferred_host_fill(out, codes_dev)
            return out
        return np.asarray(cp.asnumpy(codes_dev))
    except Exception as e:
        logger.debug("device-resident extval materialise+discretize failed, caller falls back to the host njit + upload path: %s", e)
        try:
            import cupy as _cp

            if isinstance(e, _cp.cuda.memory.OutOfMemoryError):
                from ._fe_gpu_vram import note_gpu_oom

                note_gpu_oom()  # GPU11-13: arm the shared cooldown even though the cushion check above passed - a race against another allocation can still OOM here.
        except Exception:  # nosec B110 - best-effort cooldown note, never let it mask the original failure
            pass
        return None


__all__ = ["gpu_materialise_extval_codes_host"]
