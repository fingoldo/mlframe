"""DEVICE-BORN extra-basis MI-uplift engineered scorer (SF1c :311 collapse, 2026-07-01).

``score_features_by_mi_uplift`` scores each engineered column by MI uplift vs its raw source. For the
EXTRA-BASIS families (spline / Fourier / chirp / wavelet) the host-materialised ``engineered_X`` matrix was
uploaded whole at ``_orth_mi_backends.py's `_mi_classif_batch` host-input `cp.asarray` upload site (ORTH_BASIS_B-5 fix, mrmr_audit_2026-07-22: dropped the exact line number, which had already gone stale)`` (the poly-leg twin ``_uplift_univariate_resident`` handled only
He/T/L/LL). This module rebuilds the WHOLE extra-basis engineered matrix ON the device from the small resident
raw operand columns + the per-column fit ``meta`` (the exact frequencies / knots / lo/span the host baked in),
and scores it through the SAME percentile-edge resident plug-in MI -- so nothing crosses H2D and the uplift
RATIO stays internally consistent (engineered + raw baseline on the same estimator).

ALL-DEVICE (no per-column host routing): every extra-basis family the generator can emit is ported here --
Fourier (linear ``power`` argument), chirp (``arg="quadratic"`` axis ``u = sign(z)*z^2``), Haar wavelet (the
shipped device leg ``_wavelet_basis_fe_batched._dyadic_haar_leg_gpu``), and the cubic B-spline (Cox-de Boor).
If ANY column carries a basis this module does not recognise, or on any cupy fault, it returns ``None`` and the
caller keeps the WHOLE matrix on the exact host scorer (byte-identical default path untouched -- the safety
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
import pandas as pd

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
    ``[knots[degree]+1e-12, knots[nk-degree-1]-1e-12]`` before the de-Boor span algorithm -- replicated by the
    same clip here (cp.clip lower/upper match the host's ``zi <= knots[degree]`` / ``zi >= knots[nk-degree-1]``
    branches). B_{idx,degree} is 0 outside its support ``[knots[idx], knots[idx+degree+1])`` -- exactly the host's
    ``rel not in [0, degree] -> 0``."""
    knots = np.asarray(knots, dtype=np.float64)
    nk = int(knots.shape[0])
    lo_c = float(knots[degree]) + 1e-12
    hi_c = float(knots[nk - degree - 1]) - 1e-12
    # Unconditional float64 (matches the host's `z = np.asarray(z, dtype=np.float64)`, NOT gated on
    # MLFRAME_CRIT_DTYPE_RELAXED): the 1e-12 boundary-safety clip below is only representable in
    # float64 -- float32's epsilon near 1.0 is ~1.19e-7, so clipping a float32 z to hi_c=1.0-1e-12
    # silently no-ops (rounds back to exactly 1.0), leaving z sitting ON a repeated boundary knot and
    # collapsing the LAST basis function's degenerate recursion to 0 instead of its correct ~1.0 value
    # (found via test_extra_basis_device_born_parity.py: maxerr=1.0 on the *__sp8 column, pre-existing,
    # reproduces on a clean pre-wave-10 baseline).
    z_g = cp.asarray(z_g, dtype=cp.float64)
    zc = cp.clip(z_g, lo_c, hi_c)
    _cache: dict = {}

    def _N(i: int, p: int):
        """Memoised Cox-de Boor basis value ``B_{i,p}(zc)`` over the whole clipped point array; recurses down to the degree-0 indicator base case."""
        key = (i, p)
        v = _cache.get(key)
        if v is not None:
            return v
        if p == 0:
            v = ((zc >= float(knots[i])) & (zc < float(knots[i + 1]))).astype(cp.float64)
        else:
            d1 = float(knots[i + p]) - float(knots[i])
            d2 = float(knots[i + p + 1]) - float(knots[i + 1])
            v = cp.zeros(zc.shape, dtype=cp.float64)
            if d1 > 1e-12:
                v = v + ((zc - float(knots[i])) / d1) * _N(i, p - 1)
            if d2 > 1e-12:
                v = v + ((float(knots[i + p + 1]) - zc) / d2) * _N(i + 1, p - 1)
        _cache[key] = v
        return v

    return _N(int(idx), int(degree))


def _build_extra_basis_matrix_gpu(cp, raw_X: pd.DataFrame, names, meta: dict):
    """Build the (n, K) device engineered matrix for ``names`` from the resident raw operands + ``meta``.

    Raises ``ValueError`` on any unrecognised basis so the caller falls back to the whole-matrix host scorer.
    The raw source column rides the resident-operand cache under the SAME ``("xbasis_op", src)`` role the poly /
    cross-basis device builders use, so each distinct source column uploads ONCE per fit and is shared."""
    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    from .._fe_resident_operands import resident_operand
    from .._wavelet_basis_fe_batched import _dyadic_haar_leg_gpu

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

    cols = []
    for name in names:
        m = meta.get(name)
        if m is None:
            raise ValueError(f"extra-basis device build: no meta for {name!r}")
        basis = m.get("basis")
        src = m.get("src")
        if src is None or src not in raw_X.columns:
            raise ValueError(f"extra-basis device build: src {src!r} not in raw_X for {name!r}")
        xg = _raw(src)
        if basis == "fourier":
            col = _fourier_col_gpu(cp, xg, m)
        elif basis == "spline":
            span = float(m["hi"]) - float(m["lo"])
            span = span if span > 1e-12 else 1e-12
            z = cp.clip((xg - float(m["lo"])) / span, 0.0, 1.0)
            col = _bspline_col_gpu(cp, z, m["knots"], int(m["idx"]), degree=3)
        elif basis == "wavelet":
            span = float(m["span"])
            span = span if span > 1e-12 else 1e-12
            z = cp.clip((xg - float(m["lo"])) / span, 0.0, 1.0)
            col = _dyadic_haar_leg_gpu(cp, z, int(m["j"]), int(m["k"]))
        else:
            raise ValueError(f"extra-basis device build: unsupported basis {basis!r} for {name!r}")
        cols.append(col.astype(cp.float64, copy=False))
    if not cols:
        return cp.empty((n, 0), dtype=cp.float64)
    return cp.ascontiguousarray(cp.stack(cols, axis=1).astype(cp.float64, copy=False))


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
    scores it through the resident plug-in MI. Returns the (K,) host float64 MI array in ``engineered_X.columns``
    order, OR ``None`` (STRICT-residency off / no cupy / any unrecognised basis / any cupy fault) -> the caller
    keeps the WHOLE matrix on the exact host scorer (byte-identical)."""
    try:
        from .._gpu_strict_fe import fe_gpu_device_born_extra_basis_enabled

        if not fe_gpu_device_born_extra_basis_enabled():
            return None
    except Exception:
        return None
    if engineered_X is None or engineered_X.shape[1] == 0 or not meta:
        return None
    names = list(engineered_X.columns)
    try:
        import cupy as cp

        from ._gpu_resident_cross_basis import _resident_mi

        mat_gpu = _build_extra_basis_matrix_gpu(cp, raw_X, names, meta)
        if mat_gpu.shape[1] != len(names):
            return None  # never emit a misaligned MI vector
        return _resident_mi(cp, mat_gpu, y, int(nbins))
    except Exception as _exc:
        logger.debug("extra_basis_eng_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None
