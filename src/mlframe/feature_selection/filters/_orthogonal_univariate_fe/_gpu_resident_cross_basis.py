"""DEVICE-BORN orthogonal CROSS-BASIS FE candidate builder + resident MI uplift scorer (2026-06-30).

The orthogonal cross-basis FE families (pair / triplet / quadruplet / adaptive-arity) each build their
engineered candidate matrix ENTIRELY ON THE HOST -- ``h_a * h_b [* h_c [* h_d]]`` products of per-leg
orthogonal-polynomial basis columns ``h = T_deg(z)`` -- and then score the matrix by ``MI(col; y)`` via
``mi_classif_batch_chunked``. Under ``MLFRAME_FE_GPU_STRICT`` that MI routes through the resident plug-in,
so the WHOLE host product matrix is ``cp.asarray``-uploaded at ``_orth_mi_backends.py:311`` (H2D
instrumentation of a 300k STRICT F2 fit: the cross-basis families are the dominant remaining Group-1
single-site H2D -- pair-cross ~112 MB, triplet ~32 MB, quadruplet ~20 MB, attributed to the
``score_*_cross_basis_by_mi_uplift`` callers + the adaptive internal MI batch).

This module rebuilds the product matrix ON the device from the SMALL raw operand columns (uploaded once per
fit via the resident-operand cache) and scores it with the SAME percentile-edge resident plug-in MI
``_plugin_mi_classif_batch_cuda_resident`` the host STRICT path already uses (NO estimator switch), so the
host (n, K) product matrix is never materialised/uploaded. It mirrors the committed device-born pattern of
``_extra_fe_families_dispersion_resident`` / ``_resident_candidate_mi.gate_grid_mi_resident`` /
``_binned_numeric_agg_resident``.

DEVICE BASIS LEGS (reuse the shipped kernels)
---------------------------------------------
Each leg ``h = basis(x)_deg`` is evaluated on the device by the EXISTING batched basis evaluator
``_gpu_resident_basis._gpu_evaluate_basis_matrix`` (the fused Clenshaw RawKernel + the per-basis preprocess),
fed the resident raw operand column. The products are plain cupy elementwise (``h_a * h_b``), the leg-1 / lower
-arity baselines and the raw operands route through the SAME resident plug-in MI -- so the uplift RATIO
``engineered_mi / baseline_mi`` is internally consistent (both numerator and baseline on the SAME estimator;
no EDGE/RANK switch, no host/device estimator mismatch that could flip selection).

PARITY (selection-equivalence hard gate)
----------------------------------------
The host evaluates cheb/leg/herme by a FORWARD recurrence (``polyeval_dispatch`` -> njit) while the device
uses BACKWARD Clenshaw; the two agree to ~1e-12 at the default low degrees (laguerre is forward on both ->
bit-consistent). That ~1e-12 leg drift propagates through the products to ~1e-12 (relative) in the candidate
columns -- far below any selection threshold. The per-family parity test asserts BOTH legs of the contract:
the device product matrix == the host product matrix within ~1e-12 AND the resident-scored selection (top
winner / ranking) == the host selection. Per the residency mandate a wall-loss on this card is ACCEPTED.

GATE: engages ONLY under ``fe_gpu_device_born_crossbasis_enabled`` (DEFAULT ON under
``fe_gpu_strict_resident_enabled``, opt-out ``MLFRAME_FE_GPU_DEVICE_BORN_CROSSBASIS=0``). On ANY cupy error /
no cupy / non-strict it returns ``None`` and the caller takes the EXACT host scorer (byte-identical default
path untouched). NEVER ``free_all_blocks`` (mempool teardown owns that).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "build_leg_product_matrix_gpu",
    "raw_and_product_mi_resident",
    "_parse_code_deg",
]


def _parse_code_deg(token: str):
    """Parse a ``"{code}{degree}"`` leg token (e.g. ``"He2"``) -> ``degree`` (int) or ``None``. The basis code is
    IGNORED for the device leg spec -- the device builder re-routes each leg via ``basis_route_by_moments`` EXACTLY
    as the host generator did (the name carries only leg-1's code, the same convention the recipe builders unwind)."""
    for code in ("LL", "He", "T", "L"):
        if token.startswith(code):
            rest = token[len(code):]
            if rest.isdigit():
                return int(rest)
    return None


def _route_basis_host(x: np.ndarray, basis: str) -> str:
    """Resolve the leg basis EXACTLY as the host generators do: ``basis_route_by_moments`` when ``auto``,
    else the explicit basis. Cheap host-side moment probe on the (already NaN-filled) operand column."""
    from ..hermite_fe import basis_route_by_moments

    return basis_route_by_moments(x) if basis == "auto" else basis


def _operand_filled(X: pd.DataFrame, col: str) -> np.ndarray:
    """The host generators copy the column, then mean-fill non-finite rows BEFORE the basis preprocess. Reproduce
    that EXACT host pre-fill so the device leg sees the same operand values (otherwise NaN handling could diverge
    a row). ``np.array(copy=True)`` to never alias / mutate the caller's frame (the host path's no-mutation note)."""
    from .._fe_usability_signal import _crit_np_dtype
    x = np.array(X[col].to_numpy(), dtype=_crit_np_dtype())  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    finite_mask = np.isfinite(x)
    if not finite_mask.all():
        fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
        np.copyto(x, np.where(finite_mask, x, fill))
    return x


def build_leg_product_matrix_gpu(cp: Any, X: pd.DataFrame, col_specs: Sequence[dict], *, basis: str = "auto") -> Any:
    """Build the cross-basis product matrix ON the device, one column per ``col_specs`` entry, in the GIVEN
    order. Reproduces the host generators' per-cell body (``prod_i basis(x_i)_deg_i``) on device from resident
    operand columns.

    ``col_specs`` is a list of per-output dicts ``{"legs": [(col, degree), ...]}`` -- the legs of each emitted
    product column, in name order. The per-leg basis is routed EXACTLY as the host (``basis_route_by_moments``
    under ``auto``) on the SAME mean-filled operand. A leg may instead be a 3-tuple ``(col, degree, basis)`` to
    pin an EXPLICIT basis name (e.g. ``"laguerre"``) -- used when the engineered column NAME already carries the
    exact basis code the host generator chose (the univariate uplift scorer), so the device must reproduce THAT
    basis rather than re-deriving it from moments (which can route differently). Returns an (n, K) cupy float64
    matrix whose columns are the device reconstructions, matching the host products within ~1e-12 (FP; the device
    Clenshaw vs host forward recurrence delta at the default low degrees).

    Each distinct ``(col, basis, degree)`` basis leg is evaluated ONCE on the device (cache) and reused across
    every product that references it (the host generators cache identically). The raw operand columns ride the
    resident-operand cache so each distinct column uploads ONCE per fit. The candidate matrix itself is
    device-born + transient (NOT cached)."""
    from .._fe_resident_operands import resident_operand
    from .._gpu_resident_basis import _gpu_evaluate_basis_matrix

    # robust_axis decision mirrors the host _evaluate_basis_column path (its preprocess fit runs the robust
    # heavy-tail detection iff the robust-axis env is on). Read it once for the whole matrix.
    try:
        from ..hermite_fe._hermite_robust import _robust_axis_enabled
        robust_axis = bool(_robust_axis_enabled())
    except Exception:
        robust_axis = False

    n = len(X)
    leg_cache: dict = {}      # (col, basis, degree) -> (n,) cupy float64 leg values
    basis_of_col: dict = {}   # col -> routed basis (host moment route, computed once)

    def _leg(col: str, degree: int, explicit_basis: str = None):
        # Ensure the mean-filled operand is materialised + cached for this col (needed by both routes).
        if ("_x", col) not in basis_of_col:
            basis_of_col[("_x", col)] = _operand_filled(X, col)
        if explicit_basis is not None:
            # The engineered name already pins the host's chosen basis -> reproduce THAT basis exactly (do NOT
            # re-derive from moments, which can route a column to a different basis than the host generator did).
            b = explicit_basis
        else:
            b = basis_of_col.get(col)
            if b is None:
                b = _route_basis_host(basis_of_col[("_x", col)], basis)
                basis_of_col[col] = b
        key = (col, b, int(degree))
        h = leg_cache.get(key)
        if h is None:
            xf = basis_of_col[("_x", col)]
            xg = resident_operand(np.ascontiguousarray(xf), ("xbasis_op", col), dtype=cp.float64)
            # _gpu_evaluate_basis_matrix is BATCHED over (n, K) columns x degrees; here one column, one degree.
            cand, meta = _gpu_evaluate_basis_matrix(
                cp, xg[:, None], [b], [int(degree)], robust_axis=robust_axis,
            )
            if cand is None or cand.shape[1] != 1:
                # basis not GPU-ported for this leg -> signal a host fallback for the WHOLE matrix.
                raise ValueError(f"cross-basis device leg unsupported: col={col!r} basis={b!r} deg={degree}")
            h = cand[:, 0]
            leg_cache[key] = h
        return h

    out_cols = []
    for spec in col_specs:
        legs = spec["legs"]
        prod = None
        for leg in legs:
            if len(leg) == 3:
                col, degree, explicit_basis = leg
            else:
                col, degree = leg
                explicit_basis = None
            h = _leg(col, int(degree), explicit_basis)
            prod = h if prod is None else (prod * h)
        if prod is None:
            prod = cp.ones(n, dtype=cp.float64)
        out_cols.append(prod)

    if not out_cols:
        return cp.empty((n, 0), dtype=cp.float64)
    return cp.ascontiguousarray(cp.stack(out_cols, axis=1).astype(cp.float64, copy=False))


def _resident_mi(cp, mat_gpu, y, nbins: int) -> np.ndarray:
    """Score per-column MI of an ALREADY-RESIDENT (n, K) cupy matrix through the SAME percentile-edge resident
    plug-in MI the host STRICT ``_mi_classif_batch`` routes to (no estimator switch). Returns a host (K,) f64
    array. The y label vector rides the resident-operand cache (uploaded once per fit) keyed identically to the
    host STRICT path's ``y_mi_classif`` role, so the baseline + engineered scoring share the SAME y device buffer."""
    from .._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
    from .._fe_resident_operands import resident_operand

    if mat_gpu.shape[1] == 0:
        return np.empty((0,), dtype=np.float64)
    _yi = np.ascontiguousarray(np.asarray(y)).astype(np.int64).ravel()
    y_gpu = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
    _ymin = int(_yi.min()) if _yi.size else 0
    _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
    return np.asarray(
        _plugin_mi_classif_batch_cuda_resident(mat_gpu, y_gpu, int(nbins), y_min=_ymin, n_classes=_ncls),
        dtype=np.float64,
    )


def raw_and_product_mi_resident(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: Any,
    col_specs: Sequence[dict],
    *,
    nbins: int,
    basis: str = "auto",
) -> Optional[tuple]:
    """DEVICE-BORN twin of the two MI calls inside every ``score_*_cross_basis_by_mi_uplift``: the RAW baseline
    MI (``_mi_classif_batch(raw_X)``) and the ENGINEERED product-matrix MI (``mi_classif_batch_chunked``).

    Rebuilds the engineered product matrix ON the device from the small resident operand columns (collapsing the
    host product-matrix upload at ``_orth_mi_backends.py:311``) and scores BOTH raw and engineered through the
    SAME percentile-edge resident plug-in MI -- so the uplift RATIO is internally consistent (no estimator
    switch / no host-vs-device mismatch that could flip selection). Returns ``(raw_mi_map, eng_mi)`` where
    ``raw_mi_map`` is ``{raw_col: mi}`` and ``eng_mi`` is the (K,) host float64 array in ``engineered_X.columns``
    order, OR ``None`` on any cupy failure / no-cupy / unsupported basis so the caller falls back to the EXACT
    host scorer (byte-identical default path untouched).

    ``col_specs`` aligns 1:1 with ``engineered_X.columns``: each entry is ``{"legs": [(col, degree), ...]}``."""
    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None
    if engineered_X is None or engineered_X.shape[1] == 0:
        return None
    try:
        mat_gpu = build_leg_product_matrix_gpu(cp, raw_X, list(col_specs), basis=basis)
        if mat_gpu.shape[1] != engineered_X.shape[1]:
            # Spec / column-count mismatch -> bail to the host scorer (never emit a misaligned ratio).
            return None
        eng_mi = _resident_mi(cp, mat_gpu, y, nbins)
        # RAW baseline through the SAME resident plug-in (uploaded once via the operand cache). The host scorer's
        # _mi_classif_batch(raw_X) under STRICT already routes here; reproducing it keeps the baseline -- and thus
        # the uplift ratio + the abs-MI floor -- selection-equivalent.
        from .._fe_resident_operands import assemble_resident_matrix
        raw_cols = list(raw_X.columns)
        from .._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust; baseline matrix + resident cache share one dtype
        raw_np = np.ascontiguousarray(raw_X.to_numpy(dtype=_dt))
        # DEVICE-ASSEMBLE the raw baseline from its per-column resident operands: each raw column is already
        # uploaded once by the basis builders, so stacking the resident columns content-hits the cache and the
        # whole (n, k) matrix never crosses H2D (vs the prior single whole-matrix upload, a distinct blob that
        # never deduped). Column j is raw_X[raw_cols[j]] verbatim -> same bytes -> selection-identical.
        raw_gpu = assemble_resident_matrix(raw_np, raw_cols, ("xbasis_raw_baseline", tuple(raw_cols)),
                                           dtype=_dt)
        raw_mi = _resident_mi(cp, raw_gpu, y, nbins)
        raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
        return raw_mi_map, eng_mi
    except Exception as _exc:  # noqa: BLE001
        logger.debug("raw_and_product_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None
