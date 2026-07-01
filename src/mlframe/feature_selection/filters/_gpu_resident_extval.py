"""Device-born external-validation candidate materialise + discretise (Phase-1 residency).

The external-validation MI sweep in ``_emit_pair_features`` builds ``out[:, e*n_ops+o] = op_o(param_a, ext_e)``
for every (external_factor x binary_op) candidate on the HOST (``_materialise_extval_njit``, float64) and then
uploads the whole ``(n, K)`` float buffer to ``gpu_discretize_codes_host`` for binning -- a bulk H2D of the
candidate VALUES (measured ~46 MB on the F2 1M/30k STRICT-resident fit). This module builds the SAME candidate
matrix RESIDENT on the device (float64, matching the njit arithmetic op-for-op) and quantile-bins it resident,
so ONLY ``param_a`` + the external-factor columns cross H2D (once, small), never the ``(n, K)`` matrix.

SELECTION-EQUIVALENCE: the device ops are the exact float64 numpy bin_func semantics (see ``_extval_op_gpu``);
NaN/inf are NOT scrubbed -- they flow into the resident binner exactly as the CPU path's ``nanpercentile``
(NaN-ignoring edges) + ``searchsorted`` (NaN/inf -> rightmost bin) handle them (``_gpu_resident_discretize_codes``
is the same binner ``gpu_discretize_codes_host`` uses, verified bit-identical to ``discretize_2d_quantile_batch``).
So the codes match the host path except for GPU FP reduction order in the percentile edges (~1e-15, below the
equi-frequency binner's boundary tolerance). Any cupy/device fault returns ``None`` -> the caller keeps the exact
host njit + upload path (correctness first; the device path is a residency optimisation, never a correctness risk).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _extval_op_gpu(cp, op: int, a, b):
    """One binary op on device in float64, byte-faithful to ``_materialise_extval_njit`` (``_NJIT_BINARY_OP_CODES``:
    0=mul 1=add 2=sub 3=div 4=max 5=min 6=abs_diff 7=signed 8=ratio_abs). No NaN scrub: NaN/inf propagate exactly
    as the numpy bin_funcs produce them (the resident binner routes them to the rightmost bin, as searchsorted
    does on the CPU)."""
    if op == 0:            # mul
        return a * b
    if op == 1:            # add
        return a + b
    if op == 2:            # sub
        return a - b
    if op == 3:            # div = _safe_div: exact x/y for y != 0, eps floor only on an exact-zero denominator
        return a / cp.where(b == 0.0, cp.asarray(1e-9, dtype=cp.float64), b)
    if op == 4:            # max = np.maximum (nan-propagating)
        return cp.maximum(a, b)
    if op == 5:            # min = np.minimum (nan-propagating)
        return cp.minimum(a, b)
    if op == 6:            # abs_diff = |a - b|
        return cp.abs(a - b)
    if op == 7:            # signed = sign(a) * |b| (nan-propagating; sign(0)=0)
        return cp.sign(a) * cp.abs(b)
    # op == 8: ratio_abs = a / (|b| + 1)
    return a / (cp.abs(b) + 1.0)


def gpu_materialise_extval_codes_host(
    param_a: np.ndarray, param_b_list: Sequence[np.ndarray], op_codes: np.ndarray, nbins: int,
    *, dtype: Any = np.int8,
) -> Optional[np.ndarray]:
    """Device twin of ``_materialise_extval_njit`` + discretise. Materialises ``out[:, e*n_ops+o] =
    op_o(param_a, param_b_list[e])`` RESIDENT in float64 (ext-outer/op-inner order, matching the njit column
    layout), quantile-bins it resident, and returns the ``(n, K)`` host codes of ``dtype``. The candidate VALUES
    never cross H2D -- only ``param_a`` + the external-factor columns upload (once, small). Returns ``None`` on
    any cupy/device fault (caller keeps the host path). ``op_codes`` are the ``_NJIT_BINARY_OP_CODES`` in
    bin_func-registry order (the SAME array the njit kernel walks)."""
    try:
        import cupy as cp
        from ._gpu_resident_discretize import _gpu_resident_discretize_codes  # same binner as gpu_discretize_codes_host
        from ._fe_resident_operands import resident_operand  # content-keyed device cache (dedups repeated operands)
    except Exception:
        return None
    try:
        # param_a + each external-factor column ride the content-keyed resident cache: the SAME ext-factor
        # content reused across pairs uploads ONCE (dedup), and each is a f64 device array here.
        pa = resident_operand(param_a, ("extval_pa",), dtype=np.float64).ravel()
        n = int(pa.shape[0])
        _ops = np.asarray(op_codes).ravel()
        n_ops = int(_ops.shape[0])
        n_ext = len(param_b_list)
        if n == 0 or n_ops == 0 or n_ext == 0:
            return None
        K = n_ext * n_ops
        out_dev = cp.empty((n, K), dtype=cp.float64)
        for _e, _pb in enumerate(param_b_list):
            b = resident_operand(_pb, ("extval_ext",), dtype=np.float64).ravel()
            if b.shape[0] != n:
                return None
            _base = _e * n_ops
            for _o in range(n_ops):
                out_dev[:, _base + _o] = _extval_op_gpu(cp, int(_ops[_o]), pa, b)
        codes_dev = _gpu_resident_discretize_codes(out_dev, int(nbins))  # NaN/inf -> rightmost bin (searchsorted)
        _cd = np.dtype(dtype)
        codes_dev = codes_dev.astype(cp.dtype(_cd), copy=False) if codes_dev.dtype != _cd else codes_dev
        return cp.asnumpy(codes_dev)
    except Exception:
        return None


__all__ = ["gpu_materialise_extval_codes_host"]
