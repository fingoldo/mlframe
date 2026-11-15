"""DEVICE-BORN extra-basis MI-uplift engineered scorer (SF1c :311 collapse, 2026-07-01).

``score_features_by_mi_uplift`` scores each engineered column by MI uplift vs its raw source. For the
EXTRA-BASIS families (spline / Fourier / chirp / wavelet) the host-materialised ``engineered_X`` matrix was
Uploaded whole at ``_orth_mi_backends.py's `_mi_classif_batch` host-input `cp.asarray` upload site (dropped the exact line number, which had already gone stale)`` (the poly-leg twin ``_uplift_univariate_resident`` handled only
He/T/L/LL). This module rebuilds the WHOLE extra-basis engineered matrix ON the device from the small resident
raw operand columns + the per-column fit ``meta`` (the exact frequencies / knots / lo/span the host baked in),
and scores it through the SAME percentile-edge resident plug-in MI - so nothing crosses H2D and the uplift
RATIO stays internally consistent (engineered + raw baseline on the same estimator).

ALL-DEVICE (no per-column host routing): every extra-basis family the generator can emit is ported here -
Fourier (linear ``power`` argument), chirp (``arg="quadratic"`` axis ``u = sign(z)*z^2``), Haar wavelet (the
shipped device leg ``_wavelet_basis_fe_batched._dyadic_haar_leg_gpu``), and the cubic B-spline (Cox-de Boor).
If ANY column carries a basis this module does not recognise, or on any cupy fault, it returns ``None`` and the
caller keeps the WHOLE matrix on the exact host scorer (byte-identical default path untouched - the safety
fallback, not a routine per-column split).

BIT-EQUIVALENCE: each device column reproduces the host formula verbatim (same lo/span/mean/std/freq/power/knots
from ``meta``, same axis + clip), differing only in FP reduction order (~1e-15) -> the binned-MI partition is
selection-identical. The device single-basis Cox-de Boor equals the host de-Boor span algorithm's B-spline
value on each point's support (Cox-de Boor is basis-unique); outside the support both are 0.

GATE: ``fe_gpu_device_born_extra_basis_enabled`` (DEFAULT ON under STRICT-residency, opt-out
``MLFRAME_FE_GPU_DEVICE_BORN_EXTRA_BASIS=0``). NEVER ``free_all_blocks``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype
import pandas as pd

from .._kernel_types import render

logger = logging.getLogger(__name__)

__all__ = ["extra_basis_eng_mi_resident"]


def _fourier_col_gpu(cp, xg, m: dict):
    """Device twin of the host Fourier/chirp emission ``sin|cos(2*pi*freq*z)``.

    Linear (default): ``xp = x**power; z = (xp - lo)/max(span, 1e-12)`` (the host ``_fit_fourier_for_col`` axis,
    its lo/span baked into ``meta``). Chirp (``arg == "quadratic"``): ``zc = (x - mean)/std; u = sign(zc)*zc^2;
    z = (u - lo)/max(span, 1e-12)`` (the host ``_chirp_axis``). Same 2*pi*freq*z angle -> bit-equivalent column.
    """
    lo = float(m["lo"])
    span = float(m["span"])
    span = span if span > 1e-12 else 1e-12
    if m.get("arg") == "quadratic":
        mean = float(m["mean"])
        std = float(m["std"])
        std = std if std > 1e-12 else 1e-12
        zc = (xg - mean) / std
        u = cp.sign(zc) * (zc * zc)
        z = (u - lo) / span
    else:
        p = int(m.get("power", 1))
        xp = xg if p == 1 else xg**p
        z = (xp - lo) / span
    ang = (2.0 * math.pi * float(m["freq"])) * z
    return cp.sin(ang) if m.get("kind") == "sin" else cp.cos(ang)


def _bspline_col_gpu(cp, z_g, knots: np.ndarray, idx: int, degree: int = 3):
    """Device twin of ``engineered_recipes._orth_basis_recipes._bspline_basis_values`` for basis ``idx``.

    Vectorised single-basis Cox-de Boor recursion over all points: ``B_{i,0}(z) = 1[knots[i] <= z < knots[i+1]]``
    then ``B_{i,p} = (z-k_i)/(k_{i+p}-k_i) B_{i,p-1} + (k_{i+p+1}-z)/(k_{i+p+1}-k_{i+1}) B_{i+1,p-1}`` with the
    host's zero-denominator guard (``denom <= 1e-12 -> term 0``). The host clips the eval point into
    ``[knots[degree]+1e-12, knots[nk-degree-1]-1e-12]`` before the de-Boor span algorithm - replicated by the
    same clip here (cp.clip lower/upper match the host's ``zi <= knots[degree]`` / ``zi >= knots[nk-degree-1]``
    branches). B_{idx,degree} is 0 outside its support ``[knots[idx], knots[idx+degree+1])`` - exactly the host's
    ``rel not in [0, degree] -> 0``."""
    knots = np.asarray(knots, dtype=_crit_np_dtype())
    nk = int(knots.shape[0])
    lo_c = float(knots[degree]) + 1e-12
    hi_c = float(knots[nk - degree - 1]) - 1e-12
    z_g = cp.asarray(z_g, dtype=crit_float_dtype())
    if z_g.dtype == cp.float32:
        # The host clips z to hi - 1e-12 in float64 so that z never sits ON the repeated right boundary knot (there the last basis function's degenerate recursion collapses to 0 instead of its
        # limit ~1). float32's spacing near 1.0 is ~6e-8, so a clip to hi - 1e-12 rounds back to hi itself and does nothing; clip to the largest float32 below the boundary instead (the value
        # then differs from the float64 host's by the float32 spacing, not by 1).
        hi_c = float(np.nextafter(np.float32(knots[nk - degree - 1]), np.float32(-np.inf)))
    zc = cp.clip(z_g, lo_c, hi_c)
    _cache: dict = {}

    def _N(i: int, p: int):
        """Memoised Cox-de Boor basis value ``B_{i,p}(zc)`` over the whole clipped point array; recurses down to the degree-0 indicator base case."""
        key = (i, p)
        v = _cache.get(key)
        if v is not None:
            return v
        if p == 0:
            v = ((zc >= float(knots[i])) & (zc < float(knots[i + 1]))).astype(crit_float_dtype())
        else:
            d1 = float(knots[i + p]) - float(knots[i])
            d2 = float(knots[i + p + 1]) - float(knots[i + 1])
            v = cp.zeros(zc.shape, dtype=crit_float_dtype())
            if d1 > 1e-12:
                v = v + ((zc - float(knots[i])) / d1) * _N(i, p - 1)
            if d2 > 1e-12:
                v = v + ((float(knots[i + p + 1]) - zc) / d2) * _N(i + 1, p - 1)
        _cache[key] = v
        return v

    return _N(int(idx), int(degree))


_BSPLINE_SRC = render(
    r"""
extern "C" __global__
void bspline_block(const {REAL}* __restrict__ xs, const int* __restrict__ src_idx, const {ACC}* __restrict__ lo, const {ACC}* __restrict__ span,
                   const {ACC}* __restrict__ lo_c, const {ACC}* __restrict__ hi_c, const {REAL}* __restrict__ knots, const int* __restrict__ basis_idx, const int nk,
                   const long long n, const long long S, {REAL}* __restrict__ out) {
    // The cubic B-spline basis function `basis_idx[s]` of every spline column s at every row, in ONE launch over the (S, n) output with the rows fastest: the cupy Cox-de Boor chain of
    // ``_bspline_col_gpu`` (z = clip((x - lo) / span, 0, 1) at the element type, clip into [knots[3] + 1e-12, knots[nk - 4] - 1e-12], degree-0 indicators, the three triangle levels with the
    // zero-denominator guard d <= 1e-12) written as the 4-wide triangle in registers. A NaN x stays NaN through the same arithmetic as in the chain.
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < S * n; t += (long long)gridDim.x * blockDim.x) {
        const long long s = t / n;
        const long long r = t - s * n;
        const {REAL}* k = knots + s * nk;
        const int i0 = basis_idx[s];
        const {REAL} x = xs[(long long)src_idx[s] * n + r];
        {REAL} z = (x - ({REAL})lo[s]) / ({REAL})span[s];
        z = z < ({REAL})0.0 ? ({REAL})0.0 : (z > ({REAL})1.0 ? ({REAL})1.0 : z);
        const {REAL} lc = ({REAL})lo_c[s], hc = ({REAL})hi_c[s];
        const {REAL} zc = z < lc ? lc : (z > hc ? hc : z);
        {REAL} nb[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) nb[a] = (zc >= k[i0 + a] && zc < k[i0 + a + 1]) ? ({REAL})1.0 : ({REAL})0.0;
        #pragma unroll
        for (int p = 1; p <= 3; ++p) {
            #pragma unroll
            for (int a = 0; a + p <= 3; ++a) {
                const int i = i0 + a;
                const {ACC} d1 = ({ACC})k[i + p] - ({ACC})k[i];
                const {ACC} d2 = ({ACC})k[i + p + 1] - ({ACC})k[i + 1];
                {REAL} v = ({REAL})0.0;
                if (d1 > 1e-12) v = v + ((zc - k[i]) / ({REAL})d1) * nb[a];
                if (d2 > 1e-12) v = v + ((k[i + p + 1] - zc) / ({REAL})d2) * nb[a + 1];
                nb[a] = v;
            }
        }
        out[t] = nb[0];
    }
}
""",
)
_BSPLINE_KERNELS: dict = {}


def kernel_spline(cp: Any, real: Any) -> Any:
    """``bspline_block`` for values of the dtype ``real``; compiled once per type, with ``--fmad=false`` so that the multiply-adds round as the separate cupy launches did."""
    key = np.dtype(real).str
    compiled = _BSPLINE_KERNELS.get(key)
    if compiled is None:
        compiled = _BSPLINE_KERNELS[key] = cp.RawKernel(render(_BSPLINE_SRC, real=real), "bspline_block", options=("--fmad=false",))  # separate multiply and add, as the cupy chain rounds
    return compiled


def _bspline_block_gpu(cp, xs_list: list, metas: list):
    """The (S, n) block of spline columns for ``metas`` (each ``{lo, hi, knots, idx}``) over the device source columns ``xs_list`` (one per column, in the criterion dtype), one launch.

    Degree 3 only (the family emits cubic splines). Raises ``ValueError`` for a knot vector that is too short for the degree."""
    from pyutilz.performance.gpu_hw_launch import block_size_for

    dt = crit_float_dtype()
    degree = 3
    uniq: dict = {}
    for x in xs_list:
        uniq.setdefault(id(x), len(uniq))
    srcs = list({id(x): x for x in xs_list}.values())
    xs = cp.stack([cp.asarray(c, dtype=dt) for c in srcs])
    src_idx = np.array([uniq[id(x)] for x in xs_list], dtype=np.int32)
    nk = max(int(np.asarray(m["knots"]).shape[0]) for m in metas)
    knots = np.zeros((len(metas), nk), dtype=_crit_np_dtype())
    lo = np.empty(len(metas))
    span = np.empty(len(metas))
    lo_c = np.empty(len(metas))
    hi_c = np.empty(len(metas))
    for j, m in enumerate(metas):
        kn = np.asarray(m["knots"], dtype=_crit_np_dtype())
        if kn.shape[0] < 2 * (degree + 1):
            raise ValueError("extra-basis device build: spline knot vector too short")
        knots[j, : kn.shape[0]] = kn
        sp = float(m["hi"]) - float(m["lo"])
        lo[j], span[j] = float(m["lo"]), (sp if sp > 1e-12 else 1e-12)
        lo_c[j] = float(kn[degree]) + 1e-12
        top = float(kn[kn.shape[0] - degree - 1])
        # the host clips z to hi - 1e-12 in float64; at float32 that rounds back to the knot itself, so clip to the largest float32 below it (as ``_bspline_col_gpu``)
        hi_c[j] = float(np.nextafter(np.float32(top), np.float32(-np.inf))) if dt == cp.float32 else top - 1e-12
    basis_idx = np.array([int(m["idx"]) for m in metas], dtype=np.int32)
    S, n = len(metas), int(xs.shape[1])
    out = cp.empty((S, n), dtype=dt)
    kern = kernel_spline(cp, dt)
    threads = block_size_for(kern, work_items=S * n)
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    grid = int(max(1, min((S * n + threads - 1) // threads, sms * 32)))
    kern((grid,), (threads,), (xs, cp.asarray(src_idx), cp.asarray(lo), cp.asarray(span), cp.asarray(lo_c), cp.asarray(hi_c), cp.asarray(knots), cp.asarray(basis_idx), np.int32(nk), np.int64(n), np.int64(S), out))
    return out


def _rows_of_basis(names, meta: dict, raw_X: pd.DataFrame, basis: str) -> dict:
    """``{position in names: row in that basis's block}`` for the names whose fit metadata says ``basis`` and whose source column is in ``raw_X`` (the order of the block's rows)."""
    rows: dict = {}
    for pos, name in enumerate(names):
        m = meta.get(name)
        if m is not None and m.get("basis") == basis and m.get("src") in raw_X.columns:
            rows[pos] = len(rows)
    return rows


def _one_extra_column(cp, m: dict, name: str, xg, pos: int, fourier_block, fourier_row: dict, spline_block, spline_row: dict):
    """The device column of one name: its row of the Fourier or spline block when the block was built, else the per-column chain; the wavelet legs are always per column."""
    from .._wavelet_basis_fe_batched import _dyadic_haar_leg_gpu

    basis = m.get("basis")
    if basis == "fourier":
        return fourier_block[fourier_row[pos]] if fourier_block is not None else _fourier_col_gpu(cp, xg, m)
    if basis == "spline":
        if spline_block is not None:
            return spline_block[spline_row[pos]]
        span = float(m["hi"]) - float(m["lo"])
        span = span if span > 1e-12 else 1e-12
        z = cp.clip((xg - float(m["lo"])) / span, 0.0, 1.0)
        return _bspline_col_gpu(cp, z, m["knots"], int(m["idx"]), degree=3)
    if basis == "wavelet":
        span = float(m["span"])
        span = span if span > 1e-12 else 1e-12
        z = cp.clip((xg - float(m["lo"])) / span, 0.0, 1.0)
        return _dyadic_haar_leg_gpu(cp, z, int(m["j"]), int(m["k"]))
    raise ValueError(f"extra-basis device build: unsupported basis {basis!r} for {name!r}")


def _build_extra_basis_matrix_gpu(cp, raw_X: pd.DataFrame, names, meta: dict, column_major: bool = False):
    """Build the (n, K) device engineered matrix for ``names`` from the resident raw operands + ``meta``.

    Raises ``ValueError`` on any unrecognised basis so the caller falls back to the whole-matrix host scorer.
    The raw source column rides the resident-operand cache under the SAME ``("xbasis_op", src)`` role the poly /
    cross-basis device builders use, so each distinct source column uploads ONCE per fit and is shared."""
    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    from .._fe_resident_operands import resident_operand
    n = len(raw_X)
    op_cache: dict = {}

    def _raw(src):
        """Device column for raw source ``src``, uploading + caching it once via the shared resident-operand cache."""
        from .._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
        g = op_cache.get(src)
        if g is None:
            from .._fe_usability_signal import _crit_np_dtype
            _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust; operand + resident cache share one dtype
            xf = np.ascontiguousarray(np.asarray(raw_X[src].to_numpy(), dtype=_dt))
            g = resident_operand(xf, ("xbasis_op", src), dtype=_dt)
            op_cache[src] = g
        return g

    # Every Fourier / chirp name of a pool in ONE launch (bit-identical to the per-column build); the per-column loop below is the fallback and builds the other bases.
    fourier_row = _rows_of_basis(names, meta, raw_X, "fourier")
    fourier_block = None
    from ._extra_basis_fourier_cuda import block_preferred, fourier_block_cuda

    if block_preferred(len(fourier_row)):

        fourier_block = fourier_block_cuda(cp, [_raw(meta[names[pos]]["src"]) for pos in fourier_row], [meta[names[pos]] for pos in fourier_row])
    if fourier_block is not None and len(fourier_row) == len(names):
        return cp.ascontiguousarray(fourier_block.astype(crit_float_dtype(), copy=False)) if column_major else cp.ascontiguousarray(fourier_block.T.astype(crit_float_dtype(), copy=False))

    # Every cubic-spline name of the pool in ONE launch as well (the per-column Cox-de Boor chain below is the reference and the fallback).
    spline_row = _rows_of_basis(names, meta, raw_X, "spline")
    spline_block = None
    if spline_row:
        spline_block = _bspline_block_gpu(cp, [_raw(meta[names[pos]]["src"]) for pos in spline_row], [meta[names[pos]] for pos in spline_row])

    cols = []
    for pos, name in enumerate(names):
        m = meta.get(name)
        if m is None:
            raise ValueError(f"extra-basis device build: no meta for {name!r}")
        src = m.get("src")
        if src is None or src not in raw_X.columns:
            raise ValueError(f"extra-basis device build: src {src!r} not in raw_X for {name!r}")
        col = _one_extra_column(cp, m, name, _raw(src), pos, fourier_block, fourier_row, spline_block, spline_row)
        cols.append(col.astype(crit_float_dtype(), copy=False))
    if not cols:
        return cp.empty((0, n) if column_major else (n, 0), dtype=crit_float_dtype())
    # column_major: the rows of the (K, n) block are the columns, which is what the edge select and the MI kernel read, so no transpose follows
    return cp.ascontiguousarray(cp.stack(cols, axis=0 if column_major else 1).astype(crit_float_dtype(), copy=False))


def _mi_by_name_blocks(cp: Any, raw_X: pd.DataFrame, names: list, meta: dict, y: Any, nbins: int) -> np.ndarray:
    """Plug-in MI of every extra-basis column of ``names`` against ``y``, a column block at a time.

    The engineered matrix is ``n x K`` at the criterion dtype; a block of names is built (``_build_extra_basis_matrix_gpu``), scored by the resident plug-in MI and released, so the working set is a
    block (a quarter of the room under this thread's pool limit over three copies of its columns) however many columns the pool has. A column's MI depends on that column alone, so the blocks give the
    single-matrix values."""
    from .._device_sort_safe import device_room
    from ._gpu_resident_cross_basis import _resident_mi

    n = len(raw_X)
    block_cols = int(max(1, min(len(names), (device_room(cp) // 4) // max(1, 3 * n * np.dtype(crit_float_dtype()).itemsize))))
    parts = []
    for c0 in range(0, len(names), block_cols):
        mat_gpu = _build_extra_basis_matrix_gpu(cp, raw_X, names[c0 : c0 + block_cols], meta, column_major=True)
        parts.append(_resident_mi(cp, mat_gpu, y, nbins, column_major=True))
        del mat_gpu
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float64)  # Intentional float64: engineered-MI statistic


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Host or device for the extra-basis MI scoring: the kernel-tuning entry ``extra_basis_gate_backend``
DEVICE_FALLBACK_ROWS = 20_000  # rows from which the device-born build is taken before a sweep has run
_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1
_SWEEP_COLS = 4
_SWEEP_TOP = 10


def _sweep_inputs(dims: dict) -> tuple:
    """``dims["n_samples"]`` rows of ``_SWEEP_COLS`` numeric columns, the host-built spline and Fourier columns of every source with the fit metadata, a binary target."""
    from ._orth_extra_basis_fe_generate import generate_extra_basis_features

    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    X = pd.DataFrame(rng.standard_normal((n, _SWEEP_COLS)).astype(np.float32), columns=[f"x{i}" for i in range(_SWEEP_COLS)])
    y = ((np.sin(2.0 * X["x0"]) + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    eng, meta = generate_extra_basis_features(X, extra_bases=("spline", "fourier"), dedup_collinear_sources=False)
    return (X, eng, meta, y, 10)


def _view(mi: np.ndarray) -> np.ndarray:
    """The selection view of a scoring pass: how many candidates, and which are among the best ``_SWEEP_TOP`` (``_ktc_fallback.selection_view``)."""
    from .._ktc_fallback import selection_view

    return selection_view(np.asarray(mi, dtype=np.float64), top=_SWEEP_TOP, tie_band=1e-4)  # Intentional float64: MI statistic


def _sweep_host(X: pd.DataFrame, eng: pd.DataFrame, meta: dict, y: np.ndarray, nbins: int) -> np.ndarray:
    """The host path under STRICT residency: the host-built engineered matrix is uploaded and scored by the same resident plug-in MI (what ``_mi_classif_batch`` does when this module answers ``None``)."""
    import cupy as cp

    from ._gpu_resident_cross_basis import _resident_mi

    mat = cp.asarray(np.ascontiguousarray(eng.to_numpy(dtype=_crit_np_dtype())))
    return _view(_resident_mi(cp, mat, y, nbins))


def _sweep_device(X: pd.DataFrame, eng: pd.DataFrame, meta: dict, y: np.ndarray, nbins: int) -> np.ndarray:
    """The device-born scoring from the metadata; a failed launch raises, so the sweep never records a host fallback as a device time."""
    import cupy as cp

    return _view(_mi_by_name_blocks(cp, X, list(eng.columns), meta, y, nbins))


def _run_sweep() -> list:
    """Time the host and the device scoring per row count; both must put the same candidates at the top."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(sweep_backend_grid({"host": _sweep_host, "device": _sweep_device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6))


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    from .._ktc_fallback import rows_fallback

    _SPEC: Any = kernel_tuner(
        kernel_name="extra_basis_gate_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="extra_basis_gate_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for extra_basis_gate_backend failed: %s", e)
    _SPEC = None


def extra_basis_device_chosen(n_rows: int) -> bool:
    """Whether the extra-basis scoring of ``n_rows`` rows is made on the device: the per-host cache by the row count, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("extra_basis_gate_backend: KTC choose() failed, using the floor: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def extra_basis_eng_mi_resident(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: Any,
    meta: dict,
    *,
    nbins: int,
) -> Optional[np.ndarray]:
    """DEVICE-BORN twin of ``mi_classif_batch_chunked(engineered_X)`` for the EXTRA-BASIS uplift scorer.

    Rebuilds the WHOLE extra-basis engineered matrix on the device from the resident raw operands + ``meta`` and
    scores it through the resident plug-in MI. Returns the (K,) host double precision MI array in ``engineered_X.columns``
    order, OR ``None`` (STRICT-residency off / no cupy / any unrecognised basis / any cupy fault) -> the caller
    keeps the WHOLE matrix on the exact host scorer (byte-identical)."""
    try:
        from .._gpu_strict_fe import fe_gpu_device_born_extra_basis_enabled

        if not fe_gpu_device_born_extra_basis_enabled():
            return None
    except Exception as e:
        # fallback: a failing flag check keeps the whole matrix on the exact host scorer, the path that is always correct.
        logger.debug("extra_basis_eng_mi_resident: fe_gpu_device_born_extra_basis_enabled() check failed, caller keeps the whole matrix on the exact host scorer: %s", e)
        return None
    if engineered_X is None or engineered_X.shape[1] == 0 or not meta:
        return None
    names = list(engineered_X.columns)
    try:
        import cupy as cp

        if not extra_basis_device_chosen(len(raw_X)):
            return None  # the kernel-tuning entry says the host scorer is faster at this row count: the caller runs it
        return _mi_by_name_blocks(cp, raw_X, names, meta, y, int(nbins))
    except Exception as _exc:
        logger.debug("extra_basis_eng_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None
