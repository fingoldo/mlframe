"""GPU-resident FE: batched plain-unary op-code kernel + ``build_resident_operand_table`` (carve sibling of ``_gpu_resident_materialise``).

Moved VERBATIM; ``_gpu_resident_materialise`` re-exports every name here. ``_gpu_apply_prewarp`` stays lazy-imported
from ``_gpu_resident_fe`` inside the function body to avoid an import cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from mlframe.utils.log_throttle import log_throttle
from ._gpu_resident_shared import _unary_apply

logger = logging.getLogger(__name__)

# BATCHED plain-unary op-code kernel (launch-reduction). build_resident_operand_table applied
# each GPU-built operand column with a separate _unary_apply (a cupy elementwise op) + .astype(f32) + strided
# slice-assign (~3 cuLaunchKernel/col) - the measured #2 launch source (282). For the columns that share one
# float64 raw-operand group and use a pure-elementwise unary, ONE kernel now applies the per-column op code
# (libdevice math = the SAME functions cupy's elementwise calls -> bit-identical) and writes f32 straight into
# the operand table. log (smart_log full-column shift), erf/gammaln (special), prewarp, mixed-dtype groups,
# and any per-op parity miss stay on the per-column path. _BATCH_UNARY_OPS maps the bit-verified ops to codes.
# sinc (code 16) is intentionally EXCLUDED: its sin(pi x)/(pi x) form has a sub-ulp mismatch vs cupy's
# xp.sinc, so it stays on the bit-exact per-column path. Every op below is verified maxdiff-0 vs _unary_apply.
_BATCH_UNARY_OPS = {
    "identity": 0,
    "neg": 1,
    "abs": 2,
    "sqr": 3,
    "reciproc": 4,
    "sqrt": 5,
    "sin": 6,
    "sign": 7,
    "rint": 8,
    "qubed": 9,
    "invsquared": 10,
    "invqubed": 11,
    "cbrt": 12,
    "invcbrt": 13,
    "invsqrt": 14,
    "exp": 15,
    "cos": 17,
    "tan": 18,
    "arcsin": 19,
    "arccos": 20,
    "arctan": 21,
    "sinh": 22,
    "cosh": 23,
    "tanh": 24,
    "arcsinh": 25,
    "arccosh": 26,
    "arctanh": 27,
}
_BATCH_UNARY_SRC = r"""
extern "C" __global__
void batch_unary(const double* __restrict__ G, const long long* __restrict__ slot,
                 const int* __restrict__ opc, const long long* __restrict__ out_col,
                 const long long n, const int m, const int ncols, const int n_operands,
                 float* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)ncols;
    if (t >= total) return;
    int c = (int)(t % (long long)ncols);
    long long row = t / (long long)ncols;
    double x = G[row * (long long)m + slot[c]];
    double r;
    switch (opc[c]) {
        case 0:  r = x; break;
        case 1:  r = -x; break;
        case 2:  r = fabs(x); break;
        case 3:  r = x * x; break;
        case 4:  r = pow(x, -1.0); break;
        case 5:  r = sqrt(fabs(x)); break;
        case 6:  r = sin(x); break;
        case 7:  r = isnan(x) ? x : (x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)); break;
        case 8:  r = rint(x); break;
        case 9:  r = x * x * x; break;
        case 10: r = 1.0 / (x * x); break;
        case 11: r = 1.0 / (x * x * x); break;
        case 12: r = cbrt(x); break;
        case 13: r = pow(x, -1.0 / 3.0); break;
        case 14: r = pow(x, -1.0 / 2.0); break;
        case 15: r = exp(x); break;
        case 16: { double pix = 3.141592653589793 * x; r = (x == 0.0) ? 1.0 : sin(pix) / pix; break; }
        case 17: r = cos(x); break;
        case 18: r = tan(x); break;
        case 19: r = asin(x); break;
        case 20: r = acos(x); break;
        case 21: r = atan(x); break;
        case 22: r = sinh(x); break;
        case 23: r = cosh(x); break;
        case 24: r = tanh(x); break;
        case 25: r = asinh(x); break;
        case 26: r = acosh(x); break;
        case 27: r = atanh(x); break;
        default: r = x; break;
    }
    out[row * (long long)n_operands + out_col[c]] = (float)r;
}
"""
_BATCH_UNARY_KERNEL = None


def _get_batch_unary_kernel():
    """Lazily compile (once) and return the module-level batched plain-unary op-code ``batch_unary`` RawKernel singleton."""
    global _BATCH_UNARY_KERNEL
    if _BATCH_UNARY_KERNEL is None:
        import cupy as cp

        _BATCH_UNARY_KERNEL = cp.RawKernel(_BATCH_UNARY_SRC, "batch_unary")
    return _BATCH_UNARY_KERNEL


def build_resident_operand_table(transformed_vars: np.ndarray, col_specs: Sequence[Any], *, fallback_unaries: Sequence[str] = ()) -> Any:
    """Build a GPU-RESIDENT (n, n_operands) row-major float32 cupy mirror of the host operand table
    ``transformed_vars``, producing the bulk PLAIN-UNARY columns ON the GPU (via ``_unary_apply`` - the
    same math the CPU ``unary_transformations`` applied) and COPYING the rest (prewarp / gate_med /
    hermite-poly / any name in ``fallback_unaries`` / any GPU-unbuildable column) from the host array.

    ``col_specs`` is a list aligned with the operand-table columns: each entry is ``(col_idx, raw_vals,
    unary_name)`` where ``raw_vals`` is the host float64 raw operand input the CPU applied ``unary_name`` to
    (or ``None`` for a column with no GPU recipe -> copied from the host). A column is GPU-built iff
    ``raw_vals is not None``, ``unary_name`` is a known plain unary (``_unary_apply`` accepts it, not in
    ``fallback_unaries``). The unary is applied on the GPU in float64 (the dtype the CPU ``tr_func``
    received) then cast to float32 (mirroring the CPU's compute-in-f64-then-store-f32) so the GPU column
    matches the host column to fp round-off. Any per-column GPU failure falls that column back to the host
    copy (never a correctness regression). Returns the device array (already row-major C-contiguous f32);
    the caller registers it via ``register_prebuilt_operand_table``."""
    import cupy as cp

    from ._gpu_resident_fe import _gpu_apply_prewarp  # lazy: parent-defined, avoids import cycle

    n, n_operands = transformed_vars.shape
    fb = set(fallback_unaries)
    # Allocate the device mirror WITHOUT uploading the host table: the residency win is precisely that the
    # bulk operand bytes never make the host->device trip. We H2D ONLY the small per-operand RAW inputs (n
    # floats each, cached so each distinct raw operand is uploaded ONCE - they recur across a var's unaries)
    # and GPU-build the plain columns from them; the FEW non-plain / failed columns are copied from the host
    # one column at a time. Columns with no spec (the unused tail, if any) are zero-filled - they are never
    # read by the materialise (operand indices are always < the used width), so their content is irrelevant.
    g = cp.zeros((n, n_operands), dtype=cp.float32)
    _ret_build_resident_operand_table_part1 = _build_resident_operand_table_part1(col_specs, fb, cp, n, n_operands, g, _gpu_apply_prewarp, transformed_vars)
    if _ret_build_resident_operand_table_part1[0]:
        return _ret_build_resident_operand_table_part1[1]


def _build_resident_operand_table_part1(
    col_specs: Sequence[Any], fb: Any, cp: Any, n: Any, n_operands: Any, g: Any, _gpu_apply_prewarp: Any, transformed_vars: np.ndarray,
) -> tuple[Any, ...]:
    """Extracted block of ``build_resident_operand_table`` (part 1); returns the names the caller still uses."""
    # ONE-TRANSFER (phase R0): batch the DISTINCT raw operands referenced by the GPU-buildable
    # specs into per-dtype host matrices and upload each in ONE H2D, instead of one cp.asarray per distinct
    # raw. Each raw keeps its NATIVE float dtype (we group BY dtype) so the unary still applies in the exact
    # dtype the CPU ``tr_func`` saw -> the GPU column matches the host column to fp round-off (the invariant
    # the per-operand path enforced). Values are byte-identical; only the H2D packaging changes. Per-dtype
    # grouping means uniform-dtype fits (the common case: all-pandas f64 -> 14 distinct raws) collapse to ONE
    # upload. The device column is a strided VIEW into the group matrix - _unary_apply is elementwise, so the
    # result equals the contiguous-input result bit-for-bit. Any group/build failure falls that column back to
    # the host copy below (never a correctness regression).
    _raw_slot: dict = {}  # id(raw_vals) -> (dtype_key, slot_in_group)
    _groups: dict = {}  # dtype_key -> list[host column in native float dtype]
    for _spec_t in col_specs:
        col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        if raw_vals is not None and unary_name not in fb:
            _rk = id(raw_vals)
            if _rk not in _raw_slot:
                _rv = np.ascontiguousarray(raw_vals)
                if not np.issubdtype(_rv.dtype, np.floating):
                    _rv = _rv.astype(np.float64)  # CPU tr_func on a non-float would also promote
                _dk = _rv.dtype.str
                grp = _groups.setdefault(_dk, [])
                _raw_slot[_rk] = (_dk, len(grp))
                grp.append(_rv)
    _dev_groups: dict = {}  # dtype_key -> device (n, m) array (ONE H2D per dtype group)
    for _dk, cols in _groups.items():
        try:
            _host = np.ascontiguousarray(np.column_stack(cols)) if len(cols) > 1 else np.ascontiguousarray(cols[0]).reshape(-1, 1)
            _dev_groups[_dk] = cp.asarray(_host)
        except Exception as e:  # noqa: PERF203 - per-iteration fault isolation is intentional, not a hoisting candidate
            log_throttle(
                logger,
                "gpu_materialise.group_upload",
                logging.WARNING,
                "resident operand table: the column-stack/upload of group %r failed, its columns take the per-column path: %s",
                _dk,
                e,
            )
            _dev_groups[_dk] = None
    n_gpu = 0
    n_cpu = 0
    # BATCHED PRE-PASS (launch-reduction): collect the GPU-buildable plain-unary columns per dtype-group and
    # apply them with ONE batch_unary kernel each (libdevice math = bit-identical to per-column _unary_apply),
    # writing f32 straight into g - replacing ~3 cuLaunchKernel/col (_unary_apply + astype + slice-assign).
    # Only ops in _BATCH_UNARY_OPS and non-prewarp specs whose group loaded qualify; everything else stays on
    # the exact per-column path below (which skips the already-batched col_idx).
    _batched: set = set()
    _f64_key = np.dtype(np.float64).str  # batch ONLY the float64 group: the kernel computes in f64, matching
    try:  # the CPU tr_func's f64 math; an f32 group must compute in f32 -> per-col
        _bg: dict = {}  # dtype_key -> (slots[], opcs[], out_cols[])
        for _spec_t in col_specs:
            col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
            _payload = _spec_t[3] if len(_spec_t) > 3 else None
            if raw_vals is None or unary_name in fb or unary_name not in _BATCH_UNARY_OPS:
                continue
            if _payload is not None and _payload.get("kind") == "prewarp":
                continue
            _rk = id(raw_vals)
            if _rk not in _raw_slot:
                continue
            _dk, _slot = _raw_slot[_rk]
            if _dk != _f64_key or _dev_groups.get(_dk) is None:
                continue
            s, o, oc = _bg.setdefault(_dk, ([], [], []))
            s.append(_slot)
            o.append(_BATCH_UNARY_OPS[unary_name])
            oc.append(col_idx)
        if _bg:
            _ker = _get_batch_unary_kernel()
            for _dk, (s, o, oc) in _bg.items():
                _dev = _dev_groups[_dk]
                m = int(_dev.shape[1]) if _dev.ndim > 1 else 1
                G = cp.ascontiguousarray(_dev.astype(cp.float64, copy=False))
                slot = cp.asarray(np.asarray(s, dtype=np.int64))
                opc = cp.asarray(np.asarray(o, dtype=np.int32))
                out_col = cp.asarray(np.asarray(oc, dtype=np.int64))
                ncols = len(s)
                total = n * ncols
                threads = 256
                _ker(
                    ((total + threads - 1) // threads,), (threads,), (G, slot, opc, out_col, np.int64(n), np.int32(m), np.int32(ncols), np.int32(n_operands), g)
                )
                _batched.update(oc)
            n_gpu += len(_batched)
    except Exception as e:
        log_throttle(
            logger,
            "gpu_materialise.batched_build",
            logging.WARNING,
            "resident operand table: the batched GPU column build failed, every column takes the per-column path: %s",
            e,
        )
        _batched = set()  # any batch failure -> every column rebuilt by the exact per-column path below
    for _spec_t in col_specs:
        col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        if col_idx in _batched:
            continue
        _payload = _spec_t[3] if len(_spec_t) > 3 else None  # R1: prewarp GPU-apply payload (or None)
        gpu_built = False
        if raw_vals is not None and unary_name not in fb:
            try:
                _dk, _slot = _raw_slot[id(raw_vals)]
                _dev = _dev_groups.get(_dk)
                if _dev is not None:
                    x = _dev[:, _slot]  # native-dtype device view of this raw operand (no per-operand H2D)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        if _payload is not None and _payload.get("kind") == "prewarp":
                            # R1: APPLY the prewarp on the device (preprocess + Clenshaw) from the raw + spec,
                            # mirroring hermite_fe.apply_operand_prewarp - no host-column H2D. _gpu_apply_prewarp
                            # raises for any unported basis -> falls to the host copy below (bit-exact).
                            col = _gpu_apply_prewarp(cp, x, _payload["spec"])
                        else:
                            col = _unary_apply(cp, unary_name, x)
                    # nan_to_num is NOT applied here: the CPU operand table stores the raw unary output
                    # (un-scrubbed) too - the materialise kernel scrubs NaN/inf inline -> bit-equal.
                    g[:, col_idx] = col.astype(cp.float32)
                    gpu_built = True
            except Exception as e:
                log_throttle(
                    logger,
                    "gpu_materialise.per_column_build",
                    logging.WARNING,
                    "resident operand table: a per-column GPU build failed, that column takes the host path: %s",
                    e,
                )
                gpu_built = False
        if not gpu_built:
            # Non-plain (prewarp / gate_med / poly) or failed: copy just THIS column from the host (a single
            # (n,) f32 H2D, not the whole table) so the device column equals the CPU bytes exactly.
            g[:, col_idx] = cp.asarray(np.ascontiguousarray(transformed_vars[:, col_idx], dtype=np.float32))
            n_cpu += 1
        else:
            n_gpu += 1
    return True, (cp.ascontiguousarray(g), n_gpu, n_cpu)
    return False, None
