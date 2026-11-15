"""DEVICE-BORN conditional-DISPERSION FE candidate builder + resident MI gate.

The Tier-1 ``local_mi_gate`` of the conditional-dispersion FE family (Family D,
``_extra_fe_families_dispersion.py:563``) scores the ``enc_df`` matrix - one column per ordered
``(x_i, x_j)`` pair x emission kind - by ``MI(col; y)``. Under ``MLFRAME_FE_GPU_STRICT`` that scoring
routes through the resident plug-in MI, but the ``enc_df`` matrix is built ENTIRELY ON THE HOST in
``generate_conditional_dispersion_features`` and then ``cp.asarray``-uploaded (H2D instrumentation:
~288 MB single-site H2D of a 300k STRICT F2 fit, attributed to ``_extra_fe_families_dispersion.py:563``).

Unlike the binned-aggregate family, the dispersion transform is PURE-X and Y-INDEPENDENT (no OOF / fold /
target) - so there is NO leak surface and the device reconstruction is structurally simpler than binagg:
per (x_i, x_j) the only inputs are TWO raw columns (x_i, x_j), the stored ``x_j`` quantile edges, and the
per-bin ``(mu_hat, sigma_hat)`` of x_i. All small + resident-cacheable, so the WHOLE (n, K) candidate
matrix can be built on the device from those operands, collapsing the matrix upload to just the operand
columns (uploaded once per fit via the resident-operand cache).

This mirrors the committed device-born pattern of ``_binned_numeric_agg_resident`` /
``_resident_candidate_mi.gate_grid_mi_resident`` (build the candidate block on the device from resident
operands; score with the SAME percentile-edge resident plug-in MI
``_plugin_mi_classif_batch_cuda_resident`` the host STRICT path already uses - NO estimator switch).

BIT-IDENTICAL STRUCTURE (pure-X, no fold)
-----------------------------------------
The device reconstruction reproduces the host EXACTLY:

* bin codes: the SAME stored ``edges`` (from the recipe) via ``searchsorted(edges[1:-1], xj, 'right')`` on
  the device, then ``clip(0, n_bins-1)`` and NaN ``x_j`` rows forced to bin 0 - the EXACT
  ``_digitize_with_edges`` body (interior edges, side='right', NaN->0), so the integer codes match the host
  ``np.searchsorted`` bit-for-bit (same f64 edges, same side, same clip, same NaN fill).
* z-score: the SAME stored per-bin ``(bin_mean, bin_std)`` gathered by code; ``s = bin_std[code]``, then
  ``s = 1.0`` where ``s < sigma_floor`` (the EXACT ``_zscore_from_bins_njit`` floor), ``z = (x_i - mu)/s``,
  NaN ``x_i`` rows -> 0.0 - byte-identical per-row op sequence (subtract + divide, per-element independent).
* fold (|z| / z^2): the SAME ``_emit_kind`` map applied on the device.

ULP NOTE: the per-bin ``(mu_hat, sigma_hat)`` themselves are NOT recomputed on the device - they are the
SAME host-stored recipe constants (computed once at fit by ``_per_bin_mean_std``). So the ONLY arithmetic on
the device is the per-row gather + subtract + divide + fold, which is per-element independent f64 and matches
the host njit loop to the last ULP (no reduction-order delta). The codes / floor / NaN-fold / fallback
STRUCTURE is bit-exact; the emitted column values agree to ~1e-10 (f64 divide ULP). The parity test asserts
both: device enc columns == host within ~1e-10 AND the code / fold / NaN-fallback structure is identical.

GATE: engages ONLY under ``fe_gpu_device_born_dispersion_enabled`` (DEFAULT ON under
``fe_gpu_strict_resident_enabled``, opt-out ``MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION=0``). On ANY cupy error /
no cupy / non-strict it returns ``None`` and the caller takes the EXACT host ``local_mi_gate`` (byte-identical
default path untouched). NEVER ``free_all_blocks`` (mempool teardown owns that).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype
import pandas as pd

from ._kernel_types import render

logger = logging.getLogger(__name__)

__all__ = [
    "build_dispersion_matrix_gpu",
    "build_residual_abs_matrix_gpu",
    "dual_uplift_sibling_mi_resident",
    "local_mi_gate_dispersion_resident",
]

# Mirror the host floor in ``_extra_fe_families_dispersion``. Kept local (a plain constant, no import) so the
# device path does not reach back into the parent module at hot-loop time.
_DISPERSION_SIGMA_FLOOR: float = 1e-9


def _codes_from_edges_gpu(cp, xj_g, edges_g):  # the cupy-op reference of ``disp_bin_codes`` (one column at a time)
    """Device twin of ``_extra_fe_families._digitize_with_edges``: assign each ``x_j`` value to a bin in
    ``[0, len(edges)-2]`` using the INTERIOR edges (``edges[1:-1]``), side='right'; clip; NaN ``x_j`` -> bin 0.

    Byte-identical integer codes to the host ``np.searchsorted(edges[1:-1], x, 'right')`` + ``np.clip`` +
    NaN-fill (same f64 edges, x rounded to the criterion dtype as the host does, same side, same clip bounds, same NaN handling)."""
    from ._fe_usability_signal import _crit_np_dtype

    n_edges = int(edges_g.shape[0])
    hi = max(0, n_edges - 2)  # == edges.size - 2, the top bin index
    interior = edges_g[1:-1]
    # The host digitiser rounds x to the criterion dtype before it compares with the (float64) edges, so a value that sits on an edge falls into the same bin only if it is rounded the same way.
    xj_g = xj_g.astype(_crit_np_dtype(), copy=False)
    codes = cp.searchsorted(interior, xj_g, side="right").astype(cp.int64)
    codes = cp.clip(codes, 0, hi)
    codes = cp.where(cp.isfinite(xj_g), codes, 0)
    return codes.astype(cp.int64)


def _emit_kind_gpu(cp, z, kind: str):
    """Device twin of ``_extra_fe_families_dispersion._emit_kind`` on cupy ops: map the signed z-score to the requested
    emission kind (identity / abs / square). The builders run the fused ``disp_columns`` kernel; this is its reference in the tests."""
    if kind == "z":
        return z
    if kind == "absz":
        return cp.abs(z)
    if kind == "z2":
        return z * z
    raise ValueError(f"conditional_dispersion: unknown kind {kind!r}")


_COLUMN_SRC = render(
    r"""
extern "C" __global__
void disp_columns(const {REAL}* __restrict__ xi_all, const long long* __restrict__ codes_all, const {REAL}* __restrict__ mean_tab, const {REAL}* __restrict__ std_tab, const int nbmax,
                  const int* __restrict__ xi_idx, const int* __restrict__ code_idx, const int* __restrict__ tab_idx, const int* __restrict__ mode,
                  const long long n, const long long K, const {REAL} floor_v, {EDGE}* __restrict__ out) {
    // Every candidate column in ONE launch over the (K, n) output with the rows fastest (a warp reads consecutive rows of one operand and writes consecutive words of one output row).
    // Column s: gather the per-bin mean (and sigma) of its pair table by the bin code of x_j, floor sigma exactly as _zscore_from_bins_njit (s < floor -> 1), z = (x_i - mu) / s, fold by
    // `mode` (0 z, 1 |z|, 2 z^2, 3 the |x_i - mu| of the Family-B residual, no divide), rows with a non-finite x_i -> 0; the value is cast to the block's dtype.
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < K * n; t += (long long)gridDim.x * blockDim.x) {
        const long long s = t / n;
        const long long r = t - s * n;
        const {REAL} x = xi_all[(long long)xi_idx[s] * n + r];
        const long long c = codes_all[(long long)code_idx[s] * n + r];
        const long long tb = (long long)tab_idx[s] * nbmax + c;
        {REAL} v = ({REAL})0.0;
        if (isfinite(x)) {
            const {REAL} d = x - mean_tab[tb];
            const int m = mode[s];
            if (m == 3) {
                v = {FABS}(d);
            } else {
                {REAL} sg = std_tab[tb];
                sg = sg < floor_v ? ({REAL})1.0 : sg;
                const {REAL} z = d / sg;
                v = m == 0 ? z : (m == 1 ? {FABS}(z) : z * z);
            }
        }
        out[t] = ({EDGE})v;
    }
}
""",
)
_KIND_MODE = {"z": 0, "absz": 1, "z2": 2}
_COLUMN_KERNELS: dict = {}


def kernel(cp: Any, real: Any, out: Any = None) -> Any:
    """``disp_columns`` reading values of the dtype ``real`` and writing the dtype ``out`` (default ``real``); compiled once per pair of types."""
    out = real if out is None else out
    key = (np.dtype(real).str, np.dtype(out).str)
    compiled = _COLUMN_KERNELS.get(key)
    if compiled is None:
        src = render(_COLUMN_SRC, real=real, edge=out).replace("{FABS}", "fabsf" if np.dtype(real) == np.float32 else "fabs")
        compiled = _COLUMN_KERNELS[key] = cp.RawKernel(src, "disp_columns")
    return compiled


def _fused_columns(cp: Any, out: Any, xi_all: Any, codes_all: Any, mean_tab: Any, std_tab: Any, xi_idx: Any, code_idx: Any, tab_idx: Any, mode: Any) -> None:
    """Write the (K, n) candidate block ``out`` with ONE ``disp_columns`` launch: column s uses operand row ``xi_idx[s]`` of ``xi_all``, bin codes ``code_idx[s]`` of ``codes_all``, table row ``tab_idx[s]``
    of the padded (tables, nbmax) mean / sigma matrices and the emission ``mode[s]``."""
    from pyutilz.performance.gpu_hw_launch import block_size_for

    K, n = int(out.shape[0]), int(out.shape[1])
    kern = kernel(cp, xi_all.dtype, out.dtype)
    threads = block_size_for(kern, work_items=K * n)
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    grid = int(max(1, min((K * n + threads - 1) // threads, sms * 32)))
    kern((grid,), (threads,), (xi_all, codes_all, mean_tab, std_tab, np.int32(int(mean_tab.shape[1])), xi_idx, code_idx, tab_idx, mode, np.int64(n), np.int64(K), xi_all.dtype.type(_DISPERSION_SIGMA_FLOOR), out))


def _fused_column(cp: Any, out_row: Any, xi_g: Any, codes_g: Any, mean_g: Any, std_g: Any, mode: int) -> None:
    """One candidate column through :func:`_fused_columns` (a block of one)."""
    z = cp.zeros(1, dtype=cp.int32)
    _fused_columns(cp, out_row[None, :], xi_g[None, :], codes_g[None, :], mean_g[None, :], std_g[None, :], z, z, z, cp.asarray(np.array([mode], dtype=np.int32)))


_CODES_SRC = render(
    r"""
extern "C" __global__
void disp_bin_codes(const {REAL}* __restrict__ xj_all, const {REAL}* __restrict__ edges, const int* __restrict__ n_interior, const int emax, const long long n, const long long J,
                    long long* __restrict__ out) {
    // The bin codes of every distinct x_j in ONE launch over the (J, n) output (rows fastest): the number of interior edges (edges[1:-1] of the stored quantile edges) that are <= x, i.e.
    // np.searchsorted(side="right"), clipped to the top bin, with a non-finite x in bin 0 (the host's _digitize_with_edges). Edges and values are compared at the criterion dtype, as the host
    // digitiser rounds x before it compares.
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < J * n; t += (long long)gridDim.x * blockDim.x) {
        const long long j = t / n;
        const {REAL} x = xj_all[t];
        const int m = n_interior[j];
        const {REAL}* e = edges + j * emax;
        int lo = 0, hi = m;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (e[mid] <= x) lo = mid + 1; else hi = mid;
        }
        out[t] = isfinite(x) ? (long long)lo : 0LL;
    }
}
""",
)
_CODES_KERNELS: dict = {}


def codes_kernel(cp: Any, real: Any) -> Any:
    """``disp_bin_codes`` for values of the dtype ``real``; compiled once per type."""
    key = np.dtype(real).str
    compiled = _CODES_KERNELS.get(key)
    if compiled is None:
        compiled = _CODES_KERNELS[key] = cp.RawKernel(render(_CODES_SRC, real=real), "disp_bin_codes")
    return compiled


def _bin_codes_block(cp: Any, xj_list: list, edge_rows: list) -> Any:
    """The (J, n) int64 bin codes of the resident columns ``xj_list`` against their stored edges ``edge_rows`` (host arrays, ``edges[1:-1]`` is the interior), one ``disp_bin_codes`` launch."""
    from pyutilz.performance.gpu_hw_launch import block_size_for

    dt = crit_float_dtype()
    xj_all = cp.stack([cp.asarray(c, dtype=dt) for c in xj_list])
    inner = [np.asarray(e, dtype=_crit_np_dtype())[1:-1] for e in edge_rows]
    emax = max(1, max(int(e.shape[0]) for e in inner))
    tab = np.zeros((len(inner), emax), dtype=_crit_np_dtype())
    for k, e in enumerate(inner):
        tab[k, : e.shape[0]] = e
    J, n = int(xj_all.shape[0]), int(xj_all.shape[1])
    out = cp.empty((J, n), dtype=cp.int64)
    kern = codes_kernel(cp, dt)
    threads = block_size_for(kern, work_items=J * n)
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    grid = int(max(1, min((J * n + threads - 1) // threads, sms * 32)))
    kern((grid,), (threads,), (xj_all, cp.asarray(tab, dtype=dt), cp.asarray(np.array([e.shape[0] for e in inner], dtype=np.int32)), np.int32(emax), np.int64(n), np.int64(J), out))
    return out


def _stage_columns(cp: Any, X: pd.DataFrame, col_specs: Sequence[dict], table_keys: "tuple[str, str | None]", out_dtype: Any) -> Any:
    """The device operands of a candidate block: ``(out, xi_all, codes_all, mean_tab, std_tab, xi_idx, code_idx, tab_idx, mode)`` for ``col_specs``.

    Each distinct raw column is uploaded once through the resident-operand cache (``("disp_xi", x_i)``), every distinct x_j is binned from its stored edges in one ``disp_bin_codes`` launch, and the per-pair (mean, sigma)
    tables, small host constants, go up as one padded matrix. ``table_keys`` is ``("bin_mean", "bin_std")`` for the dispersion block and ``("bin_mean", None)`` for the residual block (modes 3)."""
    from ._fe_resident_operands import resident_operand

    n = len(X)
    xi_names: dict = {}
    xj_names: dict = {}
    pair_names: dict = {}
    xi_cols: list = []
    xj_cols: list = []
    edge_rows: list = []
    mean_rows: list = []
    std_rows: list = []
    xi_idx: list = []
    code_idx: list = []
    tab_idx: list = []
    modes: list = []
    for spec in col_specs:
        x_i, x_j = spec["x_i"], spec["x_j"]
        if table_keys[1] is not None:
            kind = str(spec["kind"])
            if kind not in _KIND_MODE:
                raise ValueError(f"conditional_dispersion: unknown kind {kind!r}")
            modes.append(_KIND_MODE[kind])
        else:
            modes.append(3)
        if x_j not in xj_names:
            xj = np.asarray(X[x_j].to_numpy(), dtype=_crit_np_dtype())
            xj_names[x_j] = len(xj_cols)
            xj_cols.append(resident_operand(np.ascontiguousarray(xj), ("disp_xj", x_j), dtype=crit_float_dtype()))
            edge_rows.append(np.asarray(spec["edges"], dtype=_crit_np_dtype()))
        if x_i not in xi_names:
            xi = np.asarray(X[x_i].to_numpy(), dtype=_crit_np_dtype())
            xi_names[x_i] = len(xi_cols)
            xi_cols.append(resident_operand(np.ascontiguousarray(xi), ("disp_xi", x_i), dtype=crit_float_dtype()))
        if (x_i, x_j) not in pair_names:
            pair_names[(x_i, x_j)] = len(mean_rows)
            mean_rows.append(np.asarray(spec[table_keys[0]], dtype=_crit_np_dtype()))
            std_rows.append(np.asarray(spec[table_keys[1]], dtype=_crit_np_dtype()) if table_keys[1] is not None else mean_rows[-1])
        xi_idx.append(xi_names[x_i])
        code_idx.append(xj_names[x_j])
        tab_idx.append(pair_names[(x_i, x_j)])
    nbmax = max(int(r.shape[0]) for r in mean_rows)
    mean_h = np.zeros((len(mean_rows), nbmax), dtype=_crit_np_dtype())
    std_h = np.zeros((len(mean_rows), nbmax), dtype=_crit_np_dtype())
    for k, (m_row, s_row) in enumerate(zip(mean_rows, std_rows)):
        mean_h[k, : m_row.shape[0]] = m_row
        std_h[k, : s_row.shape[0]] = s_row
    dt = crit_float_dtype()
    out = cp.empty((len(col_specs), n), dtype=out_dtype)
    return (
        out, cp.stack([cp.asarray(c, dtype=dt) for c in xi_cols]), _bin_codes_block(cp, xj_cols, edge_rows), cp.asarray(mean_h, dtype=dt), cp.asarray(std_h, dtype=dt),
        cp.asarray(np.asarray(xi_idx, dtype=np.int32)), cp.asarray(np.asarray(code_idx, dtype=np.int32)), cp.asarray(np.asarray(tab_idx, dtype=np.int32)), cp.asarray(np.asarray(modes, dtype=np.int32)),
    )


def build_dispersion_matrix_gpu(cp: Any, X: pd.DataFrame, col_specs: Sequence[dict], column_major: bool = False, out_dtype: Any = None) -> Any:
    """Build the conditional-dispersion candidate matrix ON the device, one column per ``col_specs`` entry, in the GIVEN order, with ONE ``disp_columns`` launch.
    Reproduces ``generate_conditional_dispersion_features``'s per-pair body on device from resident operand columns.

    ``col_specs`` is a list of per-output dicts ``{name, x_i, x_j, edges, bin_mean, bin_std, kind}`` (exactly the recipe fields) - the columns the host ``enc_df`` carries, in the SAME order.
    Returns an (n, K) cupy matrix at ``out_dtype`` (the criterion dtype by default; the (K, n) block when ``column_major``) whose columns are the device reconstructions, structurally bit-identical
    to the host (same bin codes, same sigma-floor, same NaN-fold, same emission fold); only the divide differs at ~1e-10 ULP.

    Operands (the two raw columns per pair) ride through the resident-operand cache so each distinct raw column uploads ONCE per fit. The per-bin ``(mu_hat, sigma_hat)`` + edges are SMALL
    host-stored recipe constants. The candidate matrix itself is device-born + transient (NOT cached)."""
    n = len(X)
    dt = crit_float_dtype() if out_dtype is None else out_dtype
    if not len(col_specs):
        return cp.empty((0, n) if column_major else (n, 0), dtype=dt)
    staged = _stage_columns(cp, X, col_specs, ("bin_mean", "bin_std"), dt)
    _fused_columns(cp, *staged)
    out = staged[0]
    # column_major: the rows of the (K, n) block are the columns - what the edge select and the MI kernel read, so no transpose follows
    return out if column_major else cp.ascontiguousarray(out.T)


def build_residual_abs_matrix_gpu(cp: Any, X: pd.DataFrame, col_specs: Sequence[dict], column_major: bool = False, out_dtype: Any = None) -> Any:
    """Build the ABSOLUTE Family-B mean-residual matrix ``|x_i - E[x_i|bin(x_j)]|`` ON the device, one column per ``col_specs`` entry, in the GIVEN order, with ONE ``disp_columns`` launch (mode 3).
    Reproduces ``_extra_fe_families.generate_conditional_residual_features``'s per-pair body on device from resident operand columns - the SAME bin-code gather + per-bin-mean subtract the
    dispersion builder uses, but WITHOUT the ``/sigma`` step, then the absolute value.

    ``col_specs`` is a list of per-output dicts ``{x_i, x_j, edges, bin_mean}`` (the Family-B recipe fields). Returns an (n, K) cupy matrix whose columns are ``|residual|`` (the (K, n) block when
    ``column_major``), structurally bit-identical to the host (same x_j bin codes via the SAME stored edges, same per-bin mean gather, same NaN ``x_i`` -> 0.0 fold); there is no divide, so the values
    match the host to the last ULP. The bin codes and ``x_i`` reuse the ``("disp_xj", x_j)`` / ``("disp_edges", x_j)`` / ``("disp_xi", x_i)`` resident operands of the dispersion builder."""
    n = len(X)
    dt = crit_float_dtype()
    if not len(col_specs):
        return cp.empty((0, n) if column_major else (n, 0), dtype=dt)
    staged = _stage_columns(cp, X, col_specs, ("bin_mean", None), dt)
    _fused_columns(cp, *staged)
    out = staged[0]
    return out if column_major else cp.ascontiguousarray(out.T)


def _mi_by_column_blocks(cp: Any, builder: Callable[..., Any], raw_X: pd.DataFrame, col_specs: Sequence[dict], y_bin: Any, nbins: int) -> np.ndarray:
    """Plug-in MI of every candidate column of ``builder(cp, raw_X, specs, column_major=True, out_dtype=...)`` against ``y_bin``, a column block at a time.

    The candidate matrix is ``K x n`` at the relaxed candidate dtype; a block of columns is built, scored by the resident MI and released, so the working set is a block (a quarter of the room under this
    thread's pool limit, about three copies live) however many candidates there are. Each column's MI depends on that column alone (its own edges, its own histogram), so the blocks give the
    single-matrix values."""
    from ._device_sort_safe import device_room
    from ._fe_resident_operands import resident_operand
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident, relaxed_candidate_dtype

    dt = relaxed_candidate_dtype(cp)
    n = len(raw_X)
    _yi = np.ascontiguousarray(np.asarray(y_bin)).astype(np.int64).ravel()
    y_gpu = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
    _ymin = int(_yi.min()) if _yi.size else 0
    _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
    block_cols = int(max(1, min(len(col_specs), (device_room(cp) // 4) // max(1, 3 * n * np.dtype(dt).itemsize))))
    parts = []
    for c0 in range(0, len(col_specs), block_cols):
        mat_gpu = builder(cp, raw_X, list(col_specs[c0 : c0 + block_cols]), column_major=True, out_dtype=dt)
        if mat_gpu.shape[0] == 0:
            continue
        parts.append(np.asarray(
            _plugin_mi_classif_batch_cuda_resident(mat_gpu, y_gpu, int(nbins), y_min=_ymin, n_classes=_ncls, relax_binning=True, keep_dtype=True, x_is_cm=True),
            dtype=_crit_np_dtype(),
        ))
        del mat_gpu
    return np.concatenate(parts) if parts else np.empty((0,), dtype=_crit_np_dtype())


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Host or device for the Tier-1 gate scoring: the kernel-tuning entry ``dispersion_gate_backend``
DEVICE_FALLBACK_ROWS = 20_000  # rows from which the device-born build is taken before a sweep has run
_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1
_SWEEP_COLS = 6
_SWEEP_TOP = 10


def _sweep_inputs(dims: dict) -> tuple:
    """``dims["n_samples"]`` rows of ``_SWEEP_COLS`` numeric columns with a conditional-variance signal, the host-generated candidate matrix and recipes of every ordered pair (absz and z2), a binary target."""
    from ._extra_fe_families_dispersion import generate_conditional_dispersion_features

    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    X = pd.DataFrame(rng.standard_normal((n, _SWEEP_COLS)).astype(_crit_np_dtype()), columns=[f"x{i}" for i in range(_SWEEP_COLS)])
    X["x1"] = X["x1"] * (1.0 + np.abs(X["x0"]))
    y = ((np.abs(X["x1"]) + 0.5 * rng.standard_normal(n)) > 1.0).astype(np.int64).to_numpy()
    enc_df, recipes = generate_conditional_dispersion_features(X, list(X.columns), n_bins=10, kinds=("absz", "z2"))
    return (X, enc_df, recipes, y, 10)


def _view(mi: np.ndarray) -> np.ndarray:
    """The selection view of a scoring pass: how many candidates, and which are among the best ``_SWEEP_TOP`` (``_ktc_fallback.selection_view``)."""
    from ._ktc_fallback import selection_view

    return selection_view(np.asarray(mi, dtype=np.float64), top=_SWEEP_TOP, tie_band=1e-4)  # Intentional float64: MI statistic


def _sweep_host(X: pd.DataFrame, enc_df: pd.DataFrame, recipes: dict, y: np.ndarray, nbins: int) -> np.ndarray:
    """The host gate's scoring: MI of the host-built candidate matrix."""
    from ._orthogonal_univariate_fe import mi_classif_batch_chunked

    return _view(mi_classif_batch_chunked(enc_df, y, nbins=nbins))


def _sweep_device(X: pd.DataFrame, enc_df: pd.DataFrame, recipes: dict, y: np.ndarray, nbins: int) -> np.ndarray:
    """The device-born scoring from the recipes; a failed launch raises, so the sweep never records a host fallback as a device time."""
    import cupy as cp

    specs = [{"name": c, **{k: recipes[c][k] for k in ("x_i", "x_j", "edges", "bin_mean", "bin_std", "kind")}} for c in enc_df.columns]
    return _view(_mi_by_column_blocks(cp, build_dispersion_matrix_gpu, X, specs, y, nbins))


def _run_sweep() -> list:
    """Time the host and the device scoring per row count; both must put the same candidates at the top."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(sweep_backend_grid({"host": _sweep_host, "device": _sweep_device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6))


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    from ._ktc_fallback import rows_fallback

    _SPEC: Any = kernel_tuner(
        kernel_name="dispersion_gate_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="dispersion_gate_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for dispersion_gate_backend failed: %s", e)
    _SPEC = None


def dispersion_device_chosen(n_rows: int) -> bool:
    """Whether the dispersion gate of ``n_rows`` rows is scored on the device: the per-host cache by the row count, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("dispersion_gate_backend: KTC choose() failed, using the floor: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def dual_uplift_sibling_mi_resident(
    raw_X: pd.DataFrame,
    y_bin: Any,
    col_specs: Sequence[dict],
    *,
    nbins: int,
) -> Optional[np.ndarray]:
    """DEVICE-BORN MI of the Family-B mean-residual SIBLING matrix ``|x_i - E[x_i|bin(x_j)]|`` for the
    conditional-dispersion DUAL-UPLIFT filter.

    Builds the ``sib_abs`` matrix ON the device from the SMALL resident operand columns (collapsing the
    ~120 MB host matrix upload at ``_extra_fe_families_dispersion.py:489``) and scores per-column MI with the
    SAME percentile-edge resident plug-in MI ``_plugin_mi_classif_batch_cuda_resident`` the host STRICT
    ``_mi_classif_batch`` routes to (NO estimator switch). Returns a host (K,) double precision MI array in ``col_specs``
    order, OR ``None`` on any cupy failure / no-cupy so the caller falls back to the exact host
    ``_mi_classif_batch`` over the host-built ``sib_abs`` (byte-identical default path untouched)."""
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed: %s", e)
        return None
    if not col_specs:
        return np.empty((0,), dtype=_crit_np_dtype())
    try:
        return _mi_by_column_blocks(cp, build_residual_abs_matrix_gpu, raw_X, list(col_specs), y_bin, int(nbins))
    except Exception as _exc:
        logger.debug("dual_uplift_sibling_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None


def local_mi_gate_dispersion_resident(
    enc_df: pd.DataFrame,
    y: Any,
    raw_X: pd.DataFrame,
    recipes: dict,
    *,
    top_k: Optional[int] = None,
    nbins: int = 10,
    mad_mult: float = 3.5,
    floor: Optional[float] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> Optional[list]:
    """DEVICE-BORN resident twin of ``_unified_fe_gate.local_mi_gate`` for the conditional-dispersion family.

    Builds the ``enc_df`` matrix ON the device from the SMALL resident operand columns (collapsing the
    ~288 MB host matrix upload at ``_extra_fe_families_dispersion.py:563``) and scores per-column MI with the
    SAME percentile-edge resident plug-in MI ``_plugin_mi_classif_batch_cuda_resident`` the host STRICT path
    uses (NO estimator switch). Returns the survivor column-name list (descending MI), IDENTICAL contract to
    ``local_mi_gate``, OR ``None`` on any cupy failure / no-cupy / non-strict so the caller falls back to the
    exact host gate (byte-identical default path untouched).

    The MI noise FLOOR is computed on the host exactly as ``local_mi_gate`` does (``raw_mi_noise_floor`` over
    raw_X) - that path is the small raw matrix already routed through STRICT ``_mi_classif_batch`` and is NOT
    the collapsed upload; reproducing it host-side keeps the floor (and thus the keep/drop decision)
    identical."""
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed: %s", e)
        return None
    if not isinstance(enc_df, pd.DataFrame) or enc_df.shape[1] == 0:
        return []
    if raw_X is not None and not dispersion_device_chosen(len(raw_X)):
        return None  # the kernel-tuning entry says the host gate scores this row count faster: the caller runs it
    cand_cols = [c for c in enc_df.columns if pd.api.types.is_numeric_dtype(enc_df[c])]
    if not cand_cols:
        return []
    # Every emitted enc column MUST have a recipe carrying its operand basis; if any is missing fall back to the
    # host gate (we cannot reconstruct it device-born).
    col_specs = []
    for c in cand_cols:
        r = recipes.get(c) if isinstance(recipes, dict) else None
        if not r or "x_i" not in r or "x_j" not in r or "edges" not in r or "bin_mean" not in r or "bin_std" not in r or "kind" not in r:
            return None
        col_specs.append({
            "name": c, "x_i": r["x_i"], "x_j": r["x_j"], "edges": r["edges"],
            "bin_mean": r["bin_mean"], "bin_std": r["bin_std"], "kind": r["kind"],
        })

    from ._unified_fe_gate import raw_mi_noise_floor, _coerce_y_classes

    if floor is None:
        floor = raw_mi_noise_floor(raw_X, y, nbins=nbins, mad_mult=mad_mult) if raw_X is not None else 0.0
    y_bin = _coerce_y_classes(y)

    try:
        cand_mi = _mi_by_column_blocks(cp, build_dispersion_matrix_gpu, raw_X, col_specs, y_bin, int(nbins))
        if cand_mi.shape[0] == 0:
            return []
    except Exception as _exc:
        logger.debug("local_mi_gate_dispersion_resident: GPU path failed (%s); host fallback", _exc)
        return None

    # Keep/rank IDENTICAL to local_mi_gate: floor -> survivors -> sort by MI desc -> top_k.
    scored = [(col, float(cand_mi[j])) for j, col in enumerate(cand_cols) if np.isfinite(cand_mi[j]) and cand_mi[j] >= floor]
    if reject_sink is not None:
        for j, col in enumerate(cand_cols):
            _mi = cand_mi[j]
            if not np.isfinite(_mi) or _mi < floor:
                try:
                    reject_sink(
                        gate="marginal_uplift_floor",
                        candidate=str(col),
                        operands=None,
                        operator="unified_local_mi_gate",
                        observed=float(_mi),
                        threshold=float(floor),
                        reason=(
                            "unified local-MI abs-MAD floor: MI below med+k*MAD raw noise floor"
                            if np.isfinite(_mi)
                            else "unified local-MI gate: MI could not be computed (non-finite), candidate dropped"
                        ),
                    )
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed: %s", e)
                    pass
    scored.sort(key=lambda t: t[1], reverse=True)
    if top_k is not None and int(top_k) > 0:
        scored = scored[: int(top_k)]
    return [col for col, _mi in scored]
