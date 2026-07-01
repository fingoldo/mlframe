"""DEVICE-BORN conditional-DISPERSION FE candidate builder + resident MI gate (2026-06-30).

The Tier-1 ``local_mi_gate`` of the conditional-dispersion FE family (Family D,
``_extra_fe_families_dispersion.py:563``) scores the ``enc_df`` matrix -- one column per ordered
``(x_i, x_j)`` pair x emission kind -- by ``MI(col; y)``. Under ``MLFRAME_FE_GPU_STRICT`` that scoring
routes through the resident plug-in MI, but the ``enc_df`` matrix is built ENTIRELY ON THE HOST in
``generate_conditional_dispersion_features`` and then ``cp.asarray``-uploaded (H2D instrumentation:
~288 MB single-site H2D of a 300k STRICT F2 fit, attributed to ``_extra_fe_families_dispersion.py:563``).

Unlike the binned-aggregate family, the dispersion transform is PURE-X and Y-INDEPENDENT (no OOF / fold /
target) -- so there is NO leak surface and the device reconstruction is structurally simpler than binagg:
per (x_i, x_j) the only inputs are TWO raw columns (x_i, x_j), the stored ``x_j`` quantile edges, and the
per-bin ``(mu_hat, sigma_hat)`` of x_i. All small + resident-cacheable, so the WHOLE (n, K) candidate
matrix can be built on the device from those operands, collapsing the matrix upload to just the operand
columns (uploaded once per fit via the resident-operand cache).

This mirrors the committed device-born pattern of ``_binned_numeric_agg_resident`` /
``_resident_candidate_mi.gate_grid_mi_resident`` (build the candidate block on the device from resident
operands; score with the SAME percentile-edge resident plug-in MI
``_plugin_mi_classif_batch_cuda_resident`` the host STRICT path already uses -- NO estimator switch).

BIT-IDENTICAL STRUCTURE (pure-X, no fold)
-----------------------------------------
The device reconstruction reproduces the host EXACTLY:

* bin codes: the SAME stored ``edges`` (from the recipe) via ``searchsorted(edges[1:-1], xj, 'right')`` on
  the device, then ``clip(0, n_bins-1)`` and NaN ``x_j`` rows forced to bin 0 -- the EXACT
  ``_digitize_with_edges`` body (interior edges, side='right', NaN->0), so the integer codes match the host
  ``np.searchsorted`` bit-for-bit (same f64 edges, same side, same clip, same NaN fill).
* z-score: the SAME stored per-bin ``(bin_mean, bin_std)`` gathered by code; ``s = bin_std[code]``, then
  ``s = 1.0`` where ``s < sigma_floor`` (the EXACT ``_zscore_from_bins_njit`` floor), ``z = (x_i - mu)/s``,
  NaN ``x_i`` rows -> 0.0 -- byte-identical per-row op sequence (subtract + divide, per-element independent).
* fold (|z| / z^2): the SAME ``_emit_kind`` map applied on the device.

ULP NOTE: the per-bin ``(mu_hat, sigma_hat)`` themselves are NOT recomputed on the device -- they are the
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
import pandas as pd

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


def _codes_from_edges_gpu(cp, xj_g, edges_g):
    """Device twin of ``_extra_fe_families._digitize_with_edges``: assign each ``x_j`` value to a bin in
    ``[0, len(edges)-2]`` using the INTERIOR edges (``edges[1:-1]``), side='right'; clip; NaN ``x_j`` -> bin 0.

    Byte-identical integer codes to the host ``np.searchsorted(edges[1:-1], x, 'right')`` + ``np.clip`` +
    NaN-fill (same f64 edges, same side, same clip bounds, same NaN handling)."""
    n_edges = int(edges_g.shape[0])
    hi = max(0, n_edges - 2)  # == edges.size - 2, the top bin index
    interior = edges_g[1:-1]
    codes = cp.searchsorted(interior, xj_g, side="right").astype(cp.int64)
    codes = cp.clip(codes, 0, hi)
    codes = cp.where(cp.isfinite(xj_g), codes, 0)
    return codes.astype(cp.int64)


def _emit_kind_gpu(cp, z, kind: str):
    """Device twin of ``_extra_fe_families_dispersion._emit_kind``: map the signed z-score to the requested
    emission kind. Same per-element op (identity / abs / square)."""
    if kind == "z":
        return z
    if kind == "absz":
        return cp.abs(z)
    if kind == "z2":
        return z * z
    raise ValueError(f"conditional_dispersion: unknown kind {kind!r}")


def build_dispersion_matrix_gpu(cp: Any, X: pd.DataFrame, col_specs: Sequence[dict]) -> Any:
    """Build the conditional-dispersion candidate matrix ON the device, one column per ``col_specs`` entry, in
    the GIVEN order. Reproduces ``generate_conditional_dispersion_features``'s per-pair body on device from
    resident operand columns.

    ``col_specs`` is a list of per-output dicts ``{name, x_i, x_j, edges, bin_mean, bin_std, kind}`` (exactly
    the recipe fields) -- the columns the host ``enc_df`` carries, in the SAME order. Returns an (n, K) cupy
    float64 matrix whose columns are the device reconstructions, structurally bit-identical to the host (same
    bin codes, same sigma-floor, same NaN-fold, same emission fold); only the f64 divide differs at ~1e-10 ULP.

    Operands (the two raw columns per pair) ride through the resident-operand cache so each distinct raw column
    uploads ONCE per fit. The per-bin ``(mu_hat, sigma_hat)`` + edges are SMALL host-stored recipe constants
    uploaded once per distinct (x_j) / pair. The candidate matrix itself is device-born + transient (NOT
    cached)."""
    from ._fe_resident_operands import resident_operand

    n = len(X)
    floor = float(_DISPERSION_SIGMA_FLOOR)

    # Per x_j the edges / codes are SHARED across all (x_i, x_j) columns + kinds -> compute once, keyed on x_j.
    # Per x_i the raw column is SHARED across all its (x_i, x_j) pairs -> uploaded once via the operand cache.
    # The (bin_mean, bin_std) are per-(x_i, x_j) recipe constants (uploaded once per pair). The per-kind fold is
    # the only per-column tail work after the shared z-score.
    code_cache: dict = {}   # x_j -> codes_g
    z_cache: dict = {}      # (x_i, x_j) -> z_g (signed z-score, shared across kinds of the same pair)

    out_cols = []
    for spec in col_specs:
        x_i = spec["x_i"]
        x_j = spec["x_j"]
        kind = str(spec["kind"])

        zk = (x_i, x_j)
        z_g = z_cache.get(zk)
        if z_g is None:
            codes_g = code_cache.get(x_j)
            if codes_g is None:
                xj = np.asarray(X[x_j].to_numpy(), dtype=np.float64)
                edges = np.asarray(spec["edges"], dtype=np.float64)
                xj_g = resident_operand(np.ascontiguousarray(xj), ("disp_xj", x_j), dtype=cp.float64)
                edges_g = resident_operand(np.ascontiguousarray(edges), ("disp_edges", x_j), dtype=cp.float64)
                codes_g = _codes_from_edges_gpu(cp, xj_g, edges_g)
                code_cache[x_j] = codes_g
            xi = np.asarray(X[x_i].to_numpy(), dtype=np.float64)
            xi_g = resident_operand(np.ascontiguousarray(xi), ("disp_xi", x_i), dtype=cp.float64)
            bin_mean = np.asarray(spec["bin_mean"], dtype=np.float64)
            bin_std = np.asarray(spec["bin_std"], dtype=np.float64)
            mean_g = resident_operand(np.ascontiguousarray(bin_mean), ("disp_mean", x_i, x_j), dtype=cp.float64)
            std_g = resident_operand(np.ascontiguousarray(bin_std), ("disp_std", x_i, x_j), dtype=cp.float64)
            # Gather per-row mu/sigma by code; floor sigma EXACTLY as _zscore_from_bins_njit (s<floor -> 1.0);
            # divide; NaN x_i rows -> 0.0. Per-element independent f64 -> ULP-matches the host njit loop.
            mu = mean_g[codes_g]
            s = std_g[codes_g]
            s = cp.where(s < floor, 1.0, s)
            finite_i = cp.isfinite(xi_g)
            z_g = cp.where(finite_i, (xi_g - mu) / s, 0.0)
            z_cache[zk] = z_g

        out_cols.append(_emit_kind_gpu(cp, z_g, kind))

    if not out_cols:
        return cp.empty((n, 0), dtype=cp.float64)
    return cp.ascontiguousarray(cp.stack(out_cols, axis=1).astype(cp.float64, copy=False))


def build_residual_abs_matrix_gpu(cp: Any, X: pd.DataFrame, col_specs: Sequence[dict]) -> Any:
    """Build the ABSOLUTE Family-B mean-residual matrix ``|x_i - E[x_i|bin(x_j)]|`` ON the device, one column
    per ``col_specs`` entry, in the GIVEN order. Reproduces ``_extra_fe_families.generate_conditional_residual_features``'s
    per-pair body on device from resident operand columns -- the SAME bin-code gather + per-bin-mean subtract
    the device dispersion builder uses, but WITHOUT the ``/sigma`` step (this is the dispersion z-score
    NUMERATOR before the divide), then ``cp.abs``.

    ``col_specs`` is a list of per-output dicts ``{x_i, x_j, edges, bin_mean}`` (the Family-B recipe fields).
    Returns an (n, K) cupy float64 matrix whose columns are ``|residual|``, structurally bit-identical to the
    host (same x_j bin codes via the SAME stored edges, same per-bin mean gather, same NaN ``x_i`` -> 0.0
    fold). There is NO divide here -- only a per-element gather + subtract + abs -- so the emitted values match
    the host to the last ULP (per-element independent f64).

    The bin codes reuse the SAME ``("disp_xj", x_j)`` / ``("disp_edges", x_j)`` resident operands the dispersion
    builder caches (the x_j edges are computed the SAME way -- ``_quantile_edges`` -- in both families, so a
    column built on the same ``(x_i, x_j)`` pair shares the codes); ``x_i`` rides ``("disp_xi", x_i)``; the
    per-bin mean is the Family-B-specific ``("resid_mean", x_i, x_j)`` constant uploaded once per pair. The
    candidate matrix itself is device-born + transient (NOT cached)."""
    from ._fe_resident_operands import resident_operand

    n = len(X)

    code_cache: dict = {}    # x_j -> codes_g (shared across all (x_i, x_j) pairs on the same x_j)
    resid_cache: dict = {}   # (x_i, x_j) -> |resid|_g (a pair appears once per winner, but guard dupes anyway)

    out_cols = []
    for spec in col_specs:
        x_i = spec["x_i"]
        x_j = spec["x_j"]
        rk = (x_i, x_j)
        r_g = resid_cache.get(rk)
        if r_g is None:
            codes_g = code_cache.get(x_j)
            if codes_g is None:
                xj = np.asarray(X[x_j].to_numpy(), dtype=np.float64)
                edges = np.asarray(spec["edges"], dtype=np.float64)
                xj_g = resident_operand(np.ascontiguousarray(xj), ("disp_xj", x_j), dtype=cp.float64)
                edges_g = resident_operand(np.ascontiguousarray(edges), ("disp_edges", x_j), dtype=cp.float64)
                codes_g = _codes_from_edges_gpu(cp, xj_g, edges_g)
                code_cache[x_j] = codes_g
            xi = np.asarray(X[x_i].to_numpy(), dtype=np.float64)
            xi_g = resident_operand(np.ascontiguousarray(xi), ("disp_xi", x_i), dtype=cp.float64)
            bin_mean = np.asarray(spec["bin_mean"], dtype=np.float64)
            mean_g = resident_operand(np.ascontiguousarray(bin_mean), ("resid_mean", x_i, x_j), dtype=cp.float64)
            # Gather per-row mean by code; subtract; NaN x_i rows -> 0.0 (the EXACT host fold), then abs. No
            # divide -> bit-identical to the host per-element op sequence.
            mu = mean_g[codes_g]
            finite_i = cp.isfinite(xi_g)
            r_g = cp.where(finite_i, cp.abs(xi_g - mu), 0.0)
            resid_cache[rk] = r_g
        out_cols.append(r_g)

    if not out_cols:
        return cp.empty((n, 0), dtype=cp.float64)
    return cp.ascontiguousarray(cp.stack(out_cols, axis=1).astype(cp.float64, copy=False))


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
    ``_mi_classif_batch`` routes to (NO estimator switch). Returns a host (K,) float64 MI array in ``col_specs``
    order, OR ``None`` on any cupy failure / no-cupy so the caller falls back to the exact host
    ``_mi_classif_batch`` over the host-built ``sib_abs`` (byte-identical default path untouched)."""
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None
    if not col_specs:
        return np.empty((0,), dtype=np.float64)
    try:
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
        from ._fe_resident_operands import resident_operand

        mat_gpu = build_residual_abs_matrix_gpu(cp, raw_X, list(col_specs))
        if mat_gpu.shape[1] == 0:
            return np.empty((0,), dtype=np.float64)
        _yi = np.ascontiguousarray(np.asarray(y_bin)).astype(np.int64).ravel()
        y_gpu = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
        _ymin = int(_yi.min()) if _yi.size else 0
        _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
        return np.asarray(
            _plugin_mi_classif_batch_cuda_resident(mat_gpu, y_gpu, int(nbins), y_min=_ymin, n_classes=_ncls),
            dtype=np.float64,
        )
    except Exception as _exc:  # noqa: BLE001
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
    raw_X) -- that path is the small raw matrix already routed through STRICT ``_mi_classif_batch`` and is NOT
    the collapsed upload; reproducing it host-side keeps the floor (and thus the keep/drop decision)
    identical."""
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None
    if not isinstance(enc_df, pd.DataFrame) or enc_df.shape[1] == 0:
        return []
    cand_cols = [c for c in enc_df.columns if pd.api.types.is_numeric_dtype(enc_df[c])]
    if not cand_cols:
        return []
    # Every emitted enc column MUST have a recipe carrying its operand basis; if any is missing fall back to the
    # host gate (we cannot reconstruct it device-born).
    col_specs = []
    for c in cand_cols:
        r = recipes.get(c) if isinstance(recipes, dict) else None
        if (
            not r
            or "x_i" not in r or "x_j" not in r or "edges" not in r
            or "bin_mean" not in r or "bin_std" not in r or "kind" not in r
        ):
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
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
        from ._fe_resident_operands import resident_operand

        mat_gpu = build_dispersion_matrix_gpu(cp, raw_X, col_specs)
        if mat_gpu.shape[1] == 0:
            return []
        _yi = np.ascontiguousarray(np.asarray(y_bin)).astype(np.int64).ravel()
        y_gpu = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
        _ymin = int(_yi.min()) if _yi.size else 0
        _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
        cand_mi = np.asarray(
            _plugin_mi_classif_batch_cuda_resident(mat_gpu, y_gpu, int(nbins), y_min=_ymin, n_classes=_ncls),
            dtype=np.float64,
        )
    except Exception as _exc:  # noqa: BLE001
        logger.debug("local_mi_gate_dispersion_resident: GPU path failed (%s); host fallback", _exc)
        return None

    # Keep/rank IDENTICAL to local_mi_gate: floor -> survivors -> sort by MI desc -> top_k.
    scored = [
        (col, float(cand_mi[j]))
        for j, col in enumerate(cand_cols)
        if np.isfinite(cand_mi[j]) and cand_mi[j] >= floor
    ]
    if reject_sink is not None:
        for j, col in enumerate(cand_cols):
            _mi = cand_mi[j]
            if np.isfinite(_mi) and _mi < floor:
                try:
                    reject_sink(
                        gate="marginal_uplift_floor",
                        candidate=str(col),
                        operands=None,
                        operator="unified_local_mi_gate",
                        observed=float(_mi),
                        threshold=float(floor),
                        reason="unified local-MI abs-MAD floor: MI below med+k*MAD raw noise floor",
                    )
                except Exception:
                    pass
    scored.sort(key=lambda t: t[1], reverse=True)
    if top_k is not None and int(top_k) > 0:
        scored = scored[: int(top_k)]
    return [col for col, _mi in scored]
