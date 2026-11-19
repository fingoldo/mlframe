"""GPU-resident FE: batched plain-unary op-code kernel + ``build_resident_operand_table`` (carve sibling of ``_gpu_resident_materialise``).

Moved VERBATIM; ``_gpu_resident_materialise`` re-exports every name here. The raw operands are uploaded one column block of specs at a time (``_spec_blocks``), and the kernel-tuning entry
``operand_table_backend`` chooses between rebuilding the plain columns on the device and uploading the host table. ``_gpu_apply_prewarp`` stays lazy-imported
from ``_gpu_resident_fe`` inside the function body to avoid an import cycle.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.utils.log_throttle import log_throttle
from ._gpu_resident_shared import _unary_apply
from ._kernel_types import ctype_of, render

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
_BATCH_UNARY_TEMPLATE = r"""
extern "C" __global__
void batch_unary(const {REAL}* __restrict__ G, const long long* __restrict__ slot,
                 const int* __restrict__ opc, const long long* __restrict__ out_col,
                 const long long n, const int m, const int ncols, const int n_operands,
                 {EDGE}* __restrict__ out) {   // the operand table is stored at float32 whatever the group type
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)ncols;
    if (t >= total) return;
    int c = (int)(t % (long long)ncols);
    long long row = t / (long long)ncols;
    {REAL} x = G[row * (long long)m + slot[c]];
    {REAL} r;
    switch (opc[c]) {
        case 0:  r = x; break;
        case 1:  r = -x; break;
        case 2:  r = fabs(x); break;
        case 3:  r = x * x; break;
        case 4:  r = pow(x, ({REAL})-1); break;
        case 5:  r = sqrt(fabs(x)); break;
        case 6:  r = sin(x); break;
        case 7:  r = isnan(x) ? x : (x > ({REAL})0 ? ({REAL})1 : (x < ({REAL})0 ? ({REAL})-1 : ({REAL})0)); break;
        case 8:  r = rint(x); break;
        case 9:  r = x * x * x; break;
        case 10: r = ({REAL})1 / (x * x); break;
        case 11: r = ({REAL})1 / (x * x * x); break;
        case 12: r = cbrt(x); break;
        case 13: r = pow(x, ({REAL})(-1.0 / 3.0)); break;
        case 14: r = pow(x, ({REAL})(-1.0 / 2.0)); break;
        case 15: r = exp(x); break;
        case 16: { {REAL} pix = ({REAL})3.141592653589793 * x; r = (x == ({REAL})0) ? ({REAL})1 : sin(pix) / pix; break; }
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
    out[row * (long long)n_operands + out_col[c]] = ({EDGE})r;
}
"""
_BATCH_UNARY_SRC = render(_BATCH_UNARY_TEMPLATE, real=np.float64, edge=np.float32)  # Intentional float64: the float64 source, kept under its old name for the modules that re-export it
_BATCH_UNARY_KERNELS: dict = {}
_BATCH_UNARY_KERNELS_LOCK = threading.Lock()


def _get_batch_unary_kernel(real: Any = np.float64):  # Intentional float64: each float group computes in its own type, as the CPU path does
    """Lazily compile (once per element type) and return the batched plain-unary op-code ``batch_unary`` RawKernel for the group type ``real`` (float32 or float64)."""
    key = ctype_of(real)
    with _BATCH_UNARY_KERNELS_LOCK:
        kernel = _BATCH_UNARY_KERNELS.get(key)
        if kernel is None:
            import cupy as cp

            kernel = _BATCH_UNARY_KERNELS[key] = cp.RawKernel(render(_BATCH_UNARY_TEMPLATE, real=real, edge=np.float32), "batch_unary")
    return kernel


def _spec_blocks(col_specs: Sequence[Any], fb: Any, n: int, budget_bytes: int) -> "list[list]":
    """``col_specs`` cut into consecutive blocks whose DISTINCT GPU-buildable raw operands hold at most ``budget_bytes`` (at least one raw per block, in the raw's native dtype).

    A raw operand is counted in the first block that uses it; the specs of one raw sit next to each other in the operand table, so a raw is normally uploaded by one block only."""
    blocks: "list[list]" = [[]]
    seen: set = set()
    used = 0
    for spec in col_specs:
        raw_vals = spec[1]
        if raw_vals is not None and spec[2] not in fb and id(raw_vals) not in seen:
            nbytes = int(n) * int(np.dtype(getattr(raw_vals, "dtype", np.float64)).itemsize)  # Intentional float64: default itemsize of an operand without a dtype: the host table is double
            if blocks[-1] and used + nbytes > budget_bytes:
                blocks.append([])
                used = 0
            seen.add(id(raw_vals))
            used += nbytes
        blocks[-1].append(spec)
    return blocks


def build_resident_operand_table(transformed_vars: np.ndarray, col_specs: Sequence[Any], *, fallback_unaries: Sequence[str] = ()) -> Any:
    """Build a GPU-RESIDENT (n, n_operands) row-major float32 cupy mirror of the host operand table
    ``transformed_vars``, producing the bulk PLAIN-UNARY columns ON the GPU (via ``_unary_apply`` - the
    same math the CPU ``unary_transformations`` applied) and COPYING the rest (prewarp / gate_med /
    hermite-poly / any name in ``fallback_unaries`` / any GPU-unbuildable column) from the host array.

    ``col_specs`` is a list aligned with the operand-table columns: each entry is ``(col_idx, raw_vals,
    unary_name)`` where ``raw_vals`` is the host double raw operand input the CPU applied ``unary_name`` to
    (or ``None`` for a column with no GPU recipe -> copied from the host). A column is GPU-built iff
    ``raw_vals is not None``, ``unary_name`` is a known plain unary (``_unary_apply`` accepts it, not in
    ``fallback_unaries``). The unary is applied on the GPU in double (the dtype the CPU ``tr_func``
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
    if not operand_table_device_chosen(int(n), int(n_operands)):
        # the kernel-tuning entry says uploading the host table is faster than rebuilding its plain columns on the device at this size
        return cp.asarray(np.ascontiguousarray(transformed_vars, dtype=np.float32)), 0, int(n_operands)
    g = cp.zeros((n, n_operands), dtype=cp.float32)
    # The raw operands are uploaded a column block of specs at a time (the distinct raws of a block are at most a quarter of the room under this thread's pool limit), so the group
    # matrices of a wide table are not all resident at once; the device table ``g`` itself is written column by column in place.
    from ._device_sort_safe import device_room

    n_gpu = n_cpu = 0
    for block in _spec_blocks(col_specs, fb, int(n), int(device_room(cp) // 4)):
        _ret_build_resident_operand_table_part1 = _build_resident_operand_table_part1(block, fb, cp, n, n_operands, g, _gpu_apply_prewarp, transformed_vars)
        n_gpu += int(_ret_build_resident_operand_table_part1[1][1])
        n_cpu += int(_ret_build_resident_operand_table_part1[1][2])
    return cp.ascontiguousarray(g), n_gpu, n_cpu


def _group_raw_operands(col_specs, fb):
    """Group the distinct raw operands of the GPU-buildable specs by native dtype; returns ``(raw_slot, groups)``."""
    _raw_slot: dict = {}  # id(raw_vals) -> (dtype_key, slot_in_group)
    _groups: dict = {}  # dtype_key -> list[host column in native float dtype]
    for _spec_t in col_specs:
        _, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        if raw_vals is not None and unary_name not in fb:
            _rk = id(raw_vals)
            if _rk not in _raw_slot:
                _rv = np.ascontiguousarray(raw_vals)
                if not np.issubdtype(_rv.dtype, np.floating):
                    _rv = _rv.astype(np.float64)  # CPU tr_func on a non-float would also promote  # Intentional float64: CPU-parity promotion to float64 (matches tr_func)
                _dk = _rv.dtype.str
                grp = _groups.setdefault(_dk, [])
                _raw_slot[_rk] = (_dk, len(grp))
                grp.append(_rv)
    return _raw_slot, _groups

def _upload_dtype_groups(cp, _groups):
    """One device upload per dtype group (the column-stacked host matrix); a failed group is recorded as None so its columns take the per-column path."""
    _dev_groups: dict = {}  # dtype_key -> device (n, m) array (ONE H2D per dtype group)
    failed_groups: list = []
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
            failed_groups.append(_dk)
    for _dk in failed_groups:
        _dev_groups[_dk] = None
    return _dev_groups

def _batched_unary_prepass(cp, col_specs, _raw_slot, _dev_groups, fb, g, n, n_operands):
    """Apply the GPU-buildable plain-unary columns per float group (float32 and double, each computed in its own type) with one batch kernel each; returns the set of batched column indices."""
    _batched: set = set()
    _batch_keys = {np.dtype(np.float64).str: np.float64, np.dtype(np.float32).str: np.float32}  # Intentional float64: each group computes in its own type, as the CPU tr_func does
    try:
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
            if _dk not in _batch_keys or _dev_groups.get(_dk) is None:
                continue
            s, o, oc = _bg.setdefault(_dk, ([], [], []))
            s.append(_slot)
            o.append(_BATCH_UNARY_OPS[unary_name])
            oc.append(col_idx)
        if _bg:
            for _dk, (s, o, oc) in _bg.items():
                _ker = _get_batch_unary_kernel(_batch_keys[_dk])
                _dev = _dev_groups[_dk]
                m = int(_dev.shape[1]) if _dev.ndim > 1 else 1
                G = cp.ascontiguousarray(_dev)  # the group's own element type: the kernel is compiled for it
                # columns sorted by op code: the threads of a warp run consecutive columns of one row, so equal ops sit side by side and a warp takes one or two switch branches instead of one per op
                order = np.argsort(np.asarray(o, dtype=np.int32), kind="stable")
                slot = cp.asarray(np.asarray(s, dtype=np.int64)[order])
                opc = cp.asarray(np.asarray(o, dtype=np.int32)[order])
                out_col = cp.asarray(np.asarray(oc, dtype=np.int64)[order])
                ncols = len(s)
                total = n * ncols
                threads = block_size_for(_ker, work_items=total)
                _ker(
                    ((total + threads - 1) // threads,), (threads,), (G, slot, opc, out_col, np.int64(n), np.int32(m), np.int32(ncols), np.int32(n_operands), g)
                )
                _batched.update(oc)
    except Exception as e:
        log_throttle(
            logger,
            "gpu_materialise.batched_build",
            logging.WARNING,
            "resident operand table: the batched GPU column build failed, every column takes the per-column path: %s",
            e,
        )
        _batched = set()  # any batch failure -> every column rebuilt by the exact per-column path below
    return _batched


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
    _raw_slot, _groups = _group_raw_operands(col_specs, fb)
    _dev_groups = _upload_dtype_groups(cp, _groups)
    n_gpu = 0
    n_cpu = 0
    _batched = _batched_unary_prepass(cp, col_specs, _raw_slot, _dev_groups, fb, g, n, n_operands)
    n_gpu += len(_batched)
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


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Host or device for the operand table: the kernel-tuning entry ``operand_table_backend``
DEVICE_FALLBACK_ROWS = 20_000  # rows from which the plain columns are rebuilt on the device before a sweep has run
_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1
_SWEEP_RAWS = 6
_SWEEP_UNARIES = ("identity", "neg", "abs", "sqr", "sin", "cos", "tanh", "cbrt", "exp")


def _sweep_inputs(dims: dict) -> tuple:
    """``dims["n_samples"]`` rows: ``_SWEEP_RAWS`` raw double precision operands, the host operand table of every plain unary of ``_SWEEP_UNARIES`` over each (float32, as the CPU path stores it) and the column specs."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    raws = [rng.standard_normal(n) for _ in range(_SWEEP_RAWS)]
    unaries = [u for u in _SWEEP_UNARIES if u in _BATCH_UNARY_OPS]
    specs: list = []
    cols: list = []
    for raw in raws:
        for u in unaries:
            specs.append((len(specs), raw, u))
            with np.errstate(all="ignore"):
                cols.append(_unary_apply(np, u, raw).astype(np.float32))
    return (np.ascontiguousarray(np.stack(cols, axis=1)), specs)


def _table_stats(cp: Any, table: Any) -> np.ndarray:
    """Per-column mean of the finite values and the count of non-finite values of a device table, as one host vector (the comparison view of the sweep)."""
    finite = cp.isfinite(table)
    safe = cp.where(finite, table, 0.0)
    mean = safe.sum(axis=0, dtype=cp.float64) / cp.maximum(finite.sum(axis=0), 1)  # Intentional float64: comparison statistic
    return np.asarray(np.concatenate([cp.asnumpy(mean), cp.asnumpy((~finite).sum(axis=0))]), dtype=np.float64)  # Intentional float64: comparison statistic


def _sweep_host(table: np.ndarray, specs: list) -> np.ndarray:
    """The upload path: the host table to the device."""
    import cupy as cp

    return _table_stats(cp, cp.asarray(np.ascontiguousarray(table, dtype=np.float32)))


def _sweep_device(table: np.ndarray, specs: list) -> np.ndarray:
    """The rebuild path: the plain columns built on the device from the raw operands (the table is read only for the columns without a recipe)."""
    import cupy as cp

    global _FORCE_DEVICE
    _FORCE_DEVICE = True
    try:
        dev, n_gpu, _n_cpu = build_resident_operand_table(table, specs)
    finally:
        _FORCE_DEVICE = False
    if n_gpu == 0:
        raise RuntimeError("no column was built on the device")
    return _table_stats(cp, dev)


_FORCE_DEVICE = False


from ._ktc_fallback import host_device_sweep as _host_device_sweep

_run_sweep = _host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-6, atol=1e-6)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    from ._ktc_fallback import rows_fallback

    _SPEC: Any = kernel_tuner(
        kernel_name="operand_table_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="operand_table_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for operand_table_backend failed: %s", e)
    _SPEC = None


def operand_table_device_chosen(n_rows: int, n_operands: int = 0) -> bool:
    """Whether the plain columns of the operand table are rebuilt on the device (else the host table is uploaded): the per-host cache by the row count, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    if _FORCE_DEVICE:
        return True
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("operand_table_backend: KTC choose() failed, using the floor: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
