"""DEVICE-BORN OOF binned-aggregate FE candidate builder + resident MI gate (2026-06-30).

The Tier-1 ``local_mi_gate`` of the binned-numeric-aggregate FE family (``_binned_numeric_agg_fe.py:360``)
scores the OOF ``feat_df`` matrix -- one column per (group, agg, kept_stat) -- by ``MI(col; y)``. Under
``MLFRAME_FE_GPU_STRICT`` that scoring routes through ``_mi_classif_batch`` -> the resident plug-in MI, but the
``feat_df`` matrix is built ENTIRELY ON THE HOST in ``fit_binned_numeric_agg`` and then ``cp.asarray``-uploaded
(H2D instrumentation: ~192 MB / the #2 single-site H2D of a 300k STRICT F2 fit, attributed to
``_binned_numeric_agg_fe.py:360``). Unlike the conditional gate-grid the columns ARE derived from a SMALL,
CACHEABLE operand basis: per (group, agg) pair the only inputs are TWO raw columns (the group column + the
aggregated column), the OOF fold-id vector, and the quantile edges -- all small and resident-cacheable. So the
WHOLE (n, K) candidate matrix can be built on the device from those operands, collapsing the matrix upload to
just the operand columns (uploaded once per fit via the resident-operand cache).

This mirrors the committed device-born pattern of ``_resident_candidate_mi.gate_grid_mi_resident`` (build the
candidate block on the device from resident operands; score with the SAME percentile-edge resident plug-in MI
``_plugin_mi_classif_batch_cuda_resident`` the host STRICT path already uses -- NO estimator switch).

BIT-IDENTICAL OOF STRUCTURE
---------------------------
The device reconstruction reproduces ``fit_binned_numeric_agg``'s OOF construction EXACTLY:

* fold ids: generated on the HOST with the SAME ``np.random.default_rng(random_state).permutation(n)`` (cheap,
  n ints) and uploaded -- byte-identical fold assignment, no device RNG.
* quantile codes: the SAME stored ``edges`` (from the recipe) via ``searchsorted(edges, gvals, 'right')`` on
  device -- integer codes match the host ``np.searchsorted`` bit-for-bit (same f64 edges, same side).
* per-fold gather: per fold f, per-cell raw moments over the TRAIN rows (``fold_ids != f`` AND finite) via
  ``cp.bincount`` raw-moment accumulation, then ``per[s][codes[test]]`` gathered to the TEST rows -- the SAME
  index structure as the host.
* global fallback: the SAME host-computed per-stat ``global`` (from the recipe) substituted wherever the
  per-cell stat is non-finite (empty cell) -- byte-identical fallback constant + byte-identical WHERE structure.

ULP NOTE: the raw-moment math (mean/std/skew/kurt from ``cp.bincount`` raw moments) differs from numpy only at
the last ULP (the SAME approved selection-equivalent trade the host njit accumulator already documents vs numpy
``v**3`` / ``v**4``). The fold / gather / fallback STRUCTURE is bit-exact; only the per-cell stat VALUES can
differ at ~1e-12. The parity test asserts both: device feat columns == host feat columns within ~1e-10 AND the
fold-id / code / fallback-mask structure is identical.

GATE: engages ONLY under ``fe_gpu_device_born_binagg_enabled`` (DEFAULT ON under
``fe_gpu_strict_resident_enabled``, opt-out ``MLFRAME_FE_GPU_DEVICE_BORN_BINAGG=0``). On ANY cupy error / no
cupy / non-strict it returns ``None`` and the caller takes the EXACT host ``local_mi_gate`` (byte-identical
default path untouched). NEVER ``free_all_blocks`` (mempool teardown owns that).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "binagg_fold_ids",
    "build_binagg_oof_matrix_gpu",
    "local_mi_gate_binagg_resident",
]


def binagg_fold_ids(n: int, n_folds: int, random_state: int) -> np.ndarray:
    """The EXACT OOF fold-id vector ``fit_binned_numeric_agg`` builds (host RNG, n ints).

    Byte-identical to the host: ``fold_ids[rng.permutation(n)] = arange(n) % n_folds`` with
    ``rng = np.random.default_rng(random_state)``. Kept host-side (cheap) so the device path inherits the
    SAME fold assignment without reproducing the RNG on the GPU."""
    rng = np.random.default_rng(int(random_state))
    fold_ids = np.empty(int(n), dtype=np.int64)
    fold_ids[rng.permutation(int(n))] = np.arange(int(n)) % int(n_folds)
    return fold_ids


def _per_cell_raw_moments_gpu(cp, codes_g, v_g, n_cells: int):
    """Per-cell raw moments ``(cnt, s1, s2, s3, s4)`` on the device via ``cp.bincount`` -- the device twin of
    ``_per_cell_raw_moments_njit``. Each ``s_k = sum(v**k)`` per cell; powers built by repeated multiply
    (``x2 = v*v``; ``s3 = bincount(v*x2)``; ``s4 = bincount(x2*x2)``) so the raw moments match the host njit
    accumulator to the last ULP (the SAME approved trade vs numpy ``v**3``/``v**4``)."""
    nc = int(n_cells)
    x = v_g
    x2 = x * x
    cnt = cp.bincount(codes_g, minlength=nc).astype(cp.float64)[:nc]
    s1 = cp.bincount(codes_g, weights=x, minlength=nc).astype(cp.float64)[:nc]
    s2 = cp.bincount(codes_g, weights=x2, minlength=nc).astype(cp.float64)[:nc]
    s3 = cp.bincount(codes_g, weights=x2 * x, minlength=nc).astype(cp.float64)[:nc]
    s4 = cp.bincount(codes_g, weights=x2 * x2, minlength=nc).astype(cp.float64)[:nc]
    return cnt, s1, s2, s3, s4


def _per_cell_stats_gpu(cp, codes_g, v_g, n_cells: int, stats: Sequence[str]) -> dict:
    """Device twin of ``per_cell_stats_bincount``: per-cell ``{stat: cupy(n_cells)}`` from ``cp.bincount`` raw
    moments. Empty cells -> NaN (caller substitutes the global). Same arithmetic FORM as the host (so the WHERE
    structure / NaN placement is bit-identical); the moment VALUES differ only at ULP."""
    cnt, s1, s2, s3, s4 = _per_cell_raw_moments_gpu(cp, codes_g, v_g, n_cells)
    safe = cp.maximum(cnt, 1.0)
    mean = s1 / safe
    out: dict = {}
    need_hi = any(s in ("std", "skew", "kurt") for s in stats)
    if need_hi:
        m2 = cp.maximum(s2 / safe - mean * mean, 0.0)
        std = cp.sqrt(m2)
    for stat in stats:
        if stat == "mean":
            raw = mean
        elif stat == "std":
            raw = std
        elif stat == "skew":
            m3 = s3 / safe - 3.0 * mean * (s2 / safe) + 2.0 * mean ** 3
            raw = cp.where(std > 1e-9, m3 / (std ** 3 + 1e-12), 0.0)
        elif stat == "kurt":
            m4 = s4 / safe - 4.0 * mean * (s3 / safe) + 6.0 * mean ** 2 * (s2 / safe) - 3.0 * mean ** 4
            raw = cp.where(m2 > 1e-12, m4 / (m2 * m2 + 1e-12) - 3.0, 0.0)
        else:
            raise ValueError(f"binned_numeric_agg stat {stat!r} not supported")
        out[stat] = cp.where(cnt > 0, raw, cp.nan)
    return out


def build_binagg_oof_matrix_gpu(
    cp: Any, X: pd.DataFrame, col_specs: Sequence[dict], fold_ids: np.ndarray, n_folds: int,
) -> Any:
    """Build the OOF binned-aggregate candidate matrix ON the device, one column per ``col_specs`` entry, in
    the GIVEN order. Reproduces ``fit_binned_numeric_agg``'s OOF loop on device from resident operand columns.

    ``col_specs`` is a list of per-output dicts ``{name, group_col, agg_col, stat, edges, global}`` (exactly the
    recipe fields) -- the columns the host ``feat_df`` carries, in the SAME order. ``fold_ids`` is the
    host-generated OOF fold-id vector (uploaded once). Returns an (n, K) cupy float64 matrix whose columns are
    the device OOF reconstructions, structurally bit-identical to the host (same codes, same per-fold gather,
    same global fallback); the per-cell moment VALUES differ only at ULP.

    Operands (the two raw columns per pair, y is NOT needed here) ride through the resident-operand cache so each
    distinct raw column uploads ONCE per fit. The candidate matrix itself is device-born + transient (NOT
    cached)."""
    from ._fe_resident_operands import resident_operand

    n = len(X)
    fold_g = resident_operand(np.ascontiguousarray(fold_ids, dtype=np.int64), "binagg_foldids", dtype=np.int64)
    nf = int(n_folds)

    # Per group column the codes / n_cells are SHARED across its agg columns + stats -> compute once, keyed on
    # the group column name. Per (group, agg) the finite mask + train moments per fold are shared across its
    # stats -> compute once, keyed on (group, agg). The per-stat gather/fallback is the only per-column work.
    code_cache: dict = {}   # gcol -> (codes_g, n_cells)
    pair_cache: dict = {}   # (gcol, acol) -> (av_g, finite_g, [per-fold train stat dicts placeholder])

    out_cols = []
    for spec in col_specs:
        gcol = spec["group_col"]
        acol = spec["agg_col"]
        stat = spec["stat"]
        glob = float(spec["global"])

        cc = code_cache.get(gcol)
        if cc is None:
            gvals = np.asarray(X[gcol].to_numpy(), dtype=np.float64)
            edges = np.asarray(spec["edges"], dtype=np.float64)
            gvals_g = resident_operand(np.ascontiguousarray(gvals), ("binagg_g", gcol), dtype=cp.float64)
            edges_g = resident_operand(np.ascontiguousarray(edges), ("binagg_edges", gcol), dtype=cp.float64)
            codes_g = cp.searchsorted(edges_g, gvals_g, side="right").astype(cp.int64)
            n_cells = int(codes_g.max()) + 1
            cc = (codes_g, n_cells)
            code_cache[gcol] = cc
        codes_g, n_cells = cc

        pc = pair_cache.get((gcol, acol))
        if pc is None:
            av = np.asarray(X[acol].to_numpy(), dtype=np.float64)
            av_g = resident_operand(np.ascontiguousarray(av), ("binagg_a", acol), dtype=cp.float64)
            finite_g = cp.isfinite(av_g)
            pc = (av_g, finite_g, {})
            pair_cache[(gcol, acol)] = pc
        av_g, finite_g, fold_stat_cache = pc

        # OOF: init the whole column to the global fallback, then overwrite each fold's TEST rows with the
        # train-fold per-cell stat (global where the cell was empty in train). Structurally identical to host.
        oof = cp.full(n, glob, dtype=cp.float64)
        for f in range(nf):
            tr_mask = (fold_g != f) & finite_g
            if not bool(tr_mask.any()):
                continue
            # The TRAIN per-cell stats for THIS stat over THIS fold are shared by no other column (each column
            # is one stat), but the train per-cell raw moments are shared across the stats of the same pair.
            key = (f, stat)
            per_s = fold_stat_cache.get(key)
            if per_s is None:
                tr_idx = cp.where(tr_mask)[0]
                per = _per_cell_stats_gpu(cp, codes_g[tr_idx], av_g[tr_idx], n_cells, [stat])
                per_s = per[stat]
                fold_stat_cache[key] = per_s
            test_idx = cp.where(fold_g == f)[0]
            ct = codes_g[test_idx]
            vals = per_s[ct]
            oof[test_idx] = cp.where(cp.isfinite(vals), vals, glob)
        out_cols.append(oof)

    if not out_cols:
        return cp.empty((n, 0), dtype=cp.float64)
    return cp.ascontiguousarray(cp.stack(out_cols, axis=1).astype(cp.float64, copy=False))


def local_mi_gate_binagg_resident(
    feat_df: pd.DataFrame,
    y: Any,
    raw_X: pd.DataFrame,
    recipes: dict,
    *,
    n_folds: int = 5,
    random_state: int = 0,
    top_k: Optional[int] = None,
    nbins: int = 10,
    mad_mult: float = 3.5,
    floor: Optional[float] = None,
    reject_sink: Optional[Callable[..., None]] = None,
    cand_cols: Optional[list] = None,
    n_rows: Optional[int] = None,
) -> Optional[list]:
    """DEVICE-BORN resident twin of ``_unified_fe_gate.local_mi_gate`` for the binned-aggregate family.

    Builds the OOF ``feat_df`` matrix ON the device from the SMALL resident operand columns (collapsing the
    ~192 MB host matrix upload at ``_binned_numeric_agg_fe.py:360``) and scores per-column MI with the SAME
    percentile-edge resident plug-in MI ``_plugin_mi_classif_batch_cuda_resident`` the host STRICT path uses
    (NO estimator switch). Returns the survivor column-name list (descending MI), IDENTICAL contract to
    ``local_mi_gate``, OR ``None`` on any cupy failure / no-cupy / non-strict so the caller falls back to the
    exact host gate (byte-identical default path untouched).

    The MI noise FLOOR is computed on the host exactly as ``local_mi_gate`` does (``raw_mi_noise_floor`` over
    raw_X) -- that path is the small raw matrix already routed through STRICT ``_mi_classif_batch`` and is NOT
    the collapsed upload; reproducing it host-side keeps the floor (and thus the keep/drop decision) identical."""
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None
    # ``cand_cols`` / ``n_rows`` let the DEVICE-BORN caller (binned_numeric_agg_with_recipes' recipes-only path)
    # gate WITHOUT materialising the host OOF feat_df -- it fits recipes-only, gates from those recipes here, then
    # builds the OOF for the FEW survivors. When a ``feat_df`` is supplied (the host callers) both are derived
    # from it exactly as before.
    if feat_df is not None:
        if not isinstance(feat_df, pd.DataFrame) or feat_df.shape[1] == 0:
            return []
        cand_cols = [c for c in feat_df.columns if pd.api.types.is_numeric_dtype(feat_df[c])]
        n = len(feat_df)
    else:
        cand_cols = list(cand_cols or [])
        n = int(n_rows or 0)
    if not cand_cols or n <= 0:
        return []
    # Every emitted feat column MUST have a recipe carrying its operand basis; if any is missing fall back to
    # the host gate (we cannot reconstruct it device-born).
    col_specs = []
    for c in cand_cols:
        r = recipes.get(c) if isinstance(recipes, dict) else None
        if not r or "group_col" not in r or "agg_col" not in r or "edges" not in r or "global" not in r:
            return None
        col_specs.append({
            "name": c, "group_col": r["group_col"], "agg_col": r["agg_col"],
            "stat": r.get("stat"), "edges": r["edges"], "global": r["global"],
        })

    from ._unified_fe_gate import raw_mi_noise_floor, _coerce_y_classes

    if floor is None:
        floor = raw_mi_noise_floor(raw_X, y, nbins=nbins, mad_mult=mad_mult) if raw_X is not None else 0.0
    y_bin = _coerce_y_classes(y)

    try:
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
        from ._fe_resident_operands import resident_operand

        fold_ids = binagg_fold_ids(n, n_folds, random_state)
        mat_gpu = build_binagg_oof_matrix_gpu(cp, raw_X, col_specs, fold_ids, n_folds)
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
        logger.debug("local_mi_gate_binagg_resident: GPU path failed (%s); host fallback", _exc)
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
