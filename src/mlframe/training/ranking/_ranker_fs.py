"""Feature selection for the LTR ranker suite, driven by the common ``FeatureSelectionConfig``.

The ranker suite is a separate dispatch path, so it does not go through the main per-target FS loop. ``use_mrmr_fs``
/ ``rfecv_models`` / ``use_boruta_shap`` are honoured here so LTR uses the same FS settings as every other target
type -- no LTR-specific FS config.

GROUP-AWARE relevance is the DEFAULT for the MRMR path on LTR (this is the whole point of LtR FS): relevance is a
PER-QUERY notion, so pooled (pointwise) MI(feature, relevance) is misleading -- a feature that merely encodes the
QUERY (constant within a query, varying across queries) gets high pooled MI yet carries ZERO within-query ranking
signal. The group-aware MRMR ranks features by MI(feature, relevance) computed WITHIN each query and averaged
(size-weighted), and removes redundancy via the standard feature-feature SU matrix (target-independent, so pooling
is correct there). Falls back to the pooled registry MRMR only when no query groups are available.

RFECV / BorutaShap remain available via the common config; RFECV is made query-aware by passing ``groups`` to its
GroupKFold CV. Core MRMR / RFECV / MI procedures are NOT modified -- only constructed, fit, and (for the group-aware
path) re-orchestrated from public helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numba
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_pandas_features(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "to_pandas"):  # polars
        return X.to_pandas()
    return pd.DataFrame(np.asarray(X))


@numba.njit(cache=True)
def _mi_from_edges(x: np.ndarray, y: np.ndarray, xe: np.ndarray, ye: np.ndarray) -> float:
    """Plug-in MI (nats) given precomputed unique quantile edges ``xe``/``ye`` (both size >= 2).
    ``x``/``y`` are the finite-filtered samples. Since the edges are precomputed by the caller
    (np.quantile, batched), this kernel has NO quantile inside -- so it is fully njit-able without
    the edge-divergence risk that blocked a full ``_binned_mi`` njit. The per-element bin is the
    exact ``np.searchsorted(edges[1:-1], v, side='right')`` clip (count of inner edges <= v), so the
    joint histogram is bit-identical; only the entropy reduction order differs (~1e-15, selection-
    safe). Collapses ~5 numpy dispatches/call into one njit call: 17x on the ~25-row LTR groups."""
    nx = xe.shape[0] - 1
    ny = ye.shape[0] - 1
    joint = np.zeros((nx, ny), dtype=np.float64)
    n = x.shape[0]
    for i in range(n):
        xv = x[i]
        b = 0
        while b < nx - 1 and xv >= xe[b + 1]:
            b += 1
        yv = y[i]
        c = 0
        while c < ny - 1 and yv >= ye[c + 1]:
            c += 1
        joint[b, c] += 1.0
    joint /= n
    px = np.zeros(nx)
    py = np.zeros(ny)
    for a in range(nx):
        for b2 in range(ny):
            px[a] += joint[a, b2]
            py[b2] += joint[a, b2]
    mi = 0.0
    for a in range(nx):
        for b2 in range(ny):
            j = joint[a, b2]
            if j > 0.0:
                mi += j * np.log(j / (px[a] * py[b2]))
    return mi


@numba.njit(cache=True, parallel=True)
def _group_features_mi_njit(block, y_g, qa, ye, col_fin, out_mi):
    """Fused per-group MI over ALL finite feature columns in ONE njit(parallel) dispatch.

    Replaces the Python ``for j in range(ncols)`` loop that called ``_mi_from_edges`` (plus a per-column
    ``np.unique(qa[:, j])`` + ``np.ptp``) once per (group, feature) -- at 300k that Python/numpy glue was
    the top self-time cost. Each finite column's x-edges are deduped INLINE from the (already monotone)
    ``qa[:, j]`` -- bit-identical to ``np.unique`` on sorted input (strictly-greater keep). Non-finite
    columns (``col_fin[j]`` False) are left 0.0 for the Python ``_binned_mi`` fallback. ``ye`` is the
    shared, precomputed unique y-edges. Output ``out_mi[j]`` matches ``_mi_from_edges`` bit-for-bit
    (same searchsorted binning + joint histogram + entropy)."""
    ncols = block.shape[1]
    n = block.shape[0]
    nqe = qa.shape[0]
    ny = ye.shape[0] - 1
    for j in numba.prange(ncols):
        out_mi[j] = 0.0
        if not col_fin[j]:
            continue
        # ptp guard (constant column contributes 0)
        cmin = block[0, j]
        cmax = block[0, j]
        for i in range(1, n):
            v = block[i, j]
            if v < cmin:
                cmin = v
            elif v > cmax:
                cmax = v
        if cmax <= cmin:
            continue
        # Dedup the monotone qa[:, j] into xe (== np.unique on already-sorted edges)
        xe = np.empty(nqe, dtype=np.float64)
        xe[0] = qa[0, j]
        m = 1
        for e in range(1, nqe):
            if qa[e, j] > xe[m - 1]:
                xe[m] = qa[e, j]
                m += 1
        if m < 2:
            continue
        nx = m - 1
        joint = np.zeros((nx, ny), dtype=np.float64)
        for i in range(n):
            xv = block[i, j]
            b = 0
            while b < nx - 1 and xv >= xe[b + 1]:
                b += 1
            yv = y_g[i]
            c = 0
            while c < ny - 1 and yv >= ye[c + 1]:
                c += 1
            joint[b, c] += 1.0
        for a in range(nx):
            for b2 in range(ny):
                joint[a, b2] /= n
        px = np.zeros(nx)
        py = np.zeros(ny)
        for a in range(nx):
            for b2 in range(ny):
                px[a] += joint[a, b2]
                py[b2] += joint[a, b2]
        mi = 0.0
        for a in range(nx):
            for b2 in range(ny):
                jj = joint[a, b2]
                if jj > 0.0:
                    mi += jj * np.log(jj / (px[a] * py[b2]))
        out_mi[j] = mi


def _binned_mi(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Plug-in MI between two 1-D arrays via quantile binning + a joint histogram (nats). Self-contained."""
    n = x.shape[0]
    if n < 4:
        return 0.0
    fin = np.isfinite(x) & np.isfinite(y)
    if int(fin.sum()) < 4:
        return 0.0
    x, y = x[fin], y[fin]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    # Hot path: called n_features * n_groups times (cProfile: np.quantile dominates the
    # full group_aware_relevance wall via per-call dispatch on tiny ~25-row query groups).
    # The all-finite group fast path in group_aware_relevance batches the per-column x-edges
    # into ONE np.quantile(axis=0) + hoists ye once/group; this per-pair form is the exact
    # fallback for groups with any non-finite value (the finite mask is JOINT per (x, y)).
    # bench-attempt-rejected (2026-06-23): np.add.at -> np.bincount joint histogram was
    # 0.98x (SLOWER) on the real small-group mix; a full @njit _binned_mi diverges ~1e-16
    # from np.quantile on tie edges -> selection-altering. See
    # _benchmarks/bench_binned_mi_group_relevance.py.
    xe = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    ye = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if xe.size < 2 or ye.size < 2:
        return 0.0
    return _mi_from_edges(x, y, xe, ye)


def group_aware_relevance(cols: list, arr: np.ndarray, y: np.ndarray, groups: np.ndarray, bins: int = 8) -> dict:
    """Per-feature query-aware relevance: MI(feature, relevance) computed WITHIN each query group, averaged with
    group-size weights. The ranking-correct relevance signal -- a feature constant within a query scores ~0 here
    (no within-query ranking power) even if its pooled MI is high. Self-contained; does not call the core MI."""
    out: dict = {}
    groups = np.asarray(groups)
    # Sort rows by group ONCE so each query is a contiguous block: the per-(feature, group)
    # boolean-index copies ``xj[m]``/``y[m]`` (each an O(n) scan of a length-n mask, done
    # n_features * n_groups times) collapse to O(group_size) slices, and ``y`` per group is
    # sliced once instead of re-extracted per feature. ``_binned_mi`` is order-invariant
    # (quantile edges + joint histogram), and groups are visited in the same sorted order the
    # old ``np.unique`` path used, so the size-weighted accumulation is bit-identical.
    order = np.argsort(groups, kind="mergesort")
    gs = groups[order]
    arr_s = arr[order]
    y_s = y[order]
    boundaries = np.flatnonzero(gs[1:] != gs[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [gs.size]))
    sizes = (stops - starts).astype(np.float64)
    # Normalise by the sum of CONTRIBUTING (size>=4) group sizes, not all rows: tiny queries are skipped in the accumulation, so dividing by total rows shrinks the size-weighted average toward 0 by the fraction of rows living in those skipped tiny queries -- a systematic downward relevance bias.
    contributing_total = float(sizes[sizes >= 4].sum()) or 1.0
    ncols = len(cols)
    acc = np.zeros(ncols, dtype=np.float64)
    probs = np.linspace(0, 1, bins + 1)
    for b in range(starts.size):
        s = int(starts[b])
        e = int(stops[b])
        if e - s < 4:
            continue
        y_g = y_s[s:e]
        block = arr_s[s:e]
        w = float(e - s)
        # BATCHED-QUANTILE FAST PATH (per COLUMN, not per group). When ``y_g`` is finite the joint
        # (x, y) finite mask reduces to x's own finite mask, so every all-finite column's per-column
        # np.quantile collapses into ONE batched np.quantile(block, axis=0) (bit-identical to
        # per-column, verified) and ye is computed once/group. Columns that carry a non-finite value
        # (their qa edge is NaN and unusable) fall back to the exact per-column ``_binned_mi`` -- so a
        # single injected NaN column no longer poisons batching for the whole group's other columns.
        # ``y_g`` non-finite routes the whole group to the per-column fallback (the mask is then
        # genuinely joint per feature).
        if np.isfinite(y_g).all() and np.ptp(y_g) > 0.0:
            ye = np.unique(np.quantile(y_g, probs))
            if ye.size < 2:
                continue  # ye identical across features -> every feature scores 0 for this group
            col_fin = np.isfinite(block).all(axis=0)  # (ncols,) one vectorised pass
            if col_fin.any():
                qa = np.quantile(block, probs, axis=0)
                block_c = np.ascontiguousarray(block)
                out_mi = np.zeros(ncols, dtype=np.float64)
                # One njit(parallel) dispatch scores EVERY finite column (dedup + bin + entropy inline,
                # bit-identical to the per-column _mi_from_edges path); non-finite columns stay 0.0 and
                # take the Python _binned_mi fallback below. Replaces the per-(group, feature) Python loop
                # + per-column np.unique/np.ptp that dominated self-time at large n.
                _group_features_mi_njit(block_c, y_g, qa, ye, col_fin, out_mi)
                acc += w * out_mi
            for j in range(ncols):
                if not col_fin[j]:
                    acc[j] += w * _binned_mi(block[:, j], y_g, bins=bins)
            continue
        for j in range(ncols):
            acc[j] += w * _binned_mi(block[:, j], y_g, bins=bins)
    for j, name in enumerate(cols):
        out[name] = acc[j] / contributing_total
    return out


def group_aware_mrmr_select(
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_features: Optional[int] = None,
    redundancy_weight: float = 1.0,
    relevance_floor: float = 1e-6,
    nbins: int = 10,
    bins: int = 8,
    verbose: int = 0,
) -> list:
    """Greedy mRMR (Peng 2005) with QUERY-AWARE relevance + feature-feature SU redundancy.

    relevance(f) = group_aware_relevance (per-query MI of f with the relevance label); redundancy(f, S) = mean SU(f, s)
    over the already-selected S (SU is target-independent so pooling is correct). Pick argmax(relevance - w*redundancy),
    stop when no candidate clears the relevance floor or the marginal mRMR score turns non-positive. Returns column names.
    """
    cols = list(X_df.select_dtypes(include=[np.number]).columns)
    if len(cols) <= 1:
        return cols
    arr = X_df[cols].to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    rel_map = group_aware_relevance(cols, arr, y, np.asarray(groups), bins=bins)
    rel = np.array([rel_map[c] for c in cols], dtype=np.float64)

    from mlframe.feature_selection.filters.group_aware import _su_redundancy_matrix
    red = _su_redundancy_matrix(X_df[cols], nbins=nbins)  # (n_features, n_features) SU in [0, 1]

    n = len(cols)
    cap = min(n, int(max_features)) if max_features else n
    # Adaptive floor: finite-sample binned MI is biased upward (a pure-noise feature scores a small positive MI), so
    # gate on a fraction of the strongest feature's relevance as well as the absolute floor -- keeps genuine signal,
    # drops the noise pedestal. ``relevance_frac`` of the max is the discriminator that separates s_within from noise.
    eff_floor = max(float(relevance_floor), 0.2 * float(rel.max()) if rel.size else 0.0)
    eligible = np.where(rel > eff_floor)[0]
    if eligible.size == 0:
        return []  # nothing carries within-query signal -> select nothing (caller keeps all)

    selected: list = [int(eligible[np.argmax(rel[eligible])])]
    remaining = [i for i in range(n) if i != selected[0]]
    while remaining and len(selected) < cap:
        best_i, best_score = None, -np.inf
        for i in remaining:
            if rel[i] <= eff_floor:
                continue
            redundancy = float(np.mean([red[i, s] for s in selected]))
            score = rel[i] - redundancy_weight * redundancy
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None or best_score <= 0.0:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    chosen = [cols[i] for i in selected]
    if verbose:
        logger.info("group-aware mRMR (LTR): selected %d/%d features by per-query relevance.", len(chosen), n)
    return chosen


def select_ltr_features(
    X: Any,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    feature_selection_config: Any,
    rfecv_models: Optional[list] = None,
    target_type: Any = None,
    fs_random_seed: int = 42,
    verbose: int = 0,
) -> Optional[list]:
    """Select feature columns for the LTR rankers using the common ``FeatureSelectionConfig``.

    MRMR (``use_mrmr_fs``) uses the GROUP-AWARE per-query relevance path by default when ``groups`` is available
    (pooled MI is misleading for ranking); it falls back to the pooled registry MRMR only without groups. RFECV
    (``rfecv_models``) and BorutaShap (``use_boruta_shap``) are built via the main suite's ``_build_pre_pipelines``
    (target-type-aware) and fit on ``(X, relevance)``; RFECV receives ``groups`` for query-aware GroupKFold CV.
    Returns the UNION of selected columns, or ``None`` when no FS is enabled. Never raises (a failing selector is
    skipped with a warning).
    """
    fsc = feature_selection_config
    if fsc is None:
        return None
    use_mrmr = bool(getattr(fsc, "use_mrmr_fs", False))
    use_bs = bool(getattr(fsc, "use_boruta_shap", False))
    rfecv_models = list(rfecv_models or [])
    if not (use_mrmr or use_bs or rfecv_models):
        return None

    X_df = _to_pandas_features(X)
    y_arr = np.asarray(y)
    selected: set = set()
    ran_any = False

    if use_mrmr:
        mrmr_kwargs = dict(getattr(fsc, "mrmr_kwargs", None) or {})
        if groups is not None:
            try:
                cols = group_aware_mrmr_select(
                    X_df, y_arr, np.asarray(groups),
                    max_features=mrmr_kwargs.get("max_features"),
                    nbins=int(mrmr_kwargs.get("quantization_nbins", 10)),
                    verbose=verbose,
                )
                ran_any = True
                selected.update(cols)
            except Exception as exc:
                logger.warning("LTR group-aware MRMR failed (%s: %s); falling back to pooled MRMR.", type(exc).__name__, exc)
                groups = None  # fall through to pooled path below
        if groups is None:
            try:
                from mlframe.feature_selection.registry import get
                sel = get("MRMR").instantiate(**mrmr_kwargs)
                sel.fit(X_df, pd.Series(y_arr, name="relevance"))
                ran_any = True
                sup = np.asarray(getattr(sel, "support_", []))
                if sup.dtype == bool:
                    selected.update(X_df.columns[i] for i in np.where(sup)[0])
                else:
                    selected.update(X_df.columns[int(i)] for i in sup.tolist() if 0 <= int(i) < X_df.shape[1])
            except Exception as exc:
                logger.warning("LTR pooled MRMR failed (%s: %s); skipping MRMR.", type(exc).__name__, exc)

    if use_bs or rfecv_models:
        try:
            ran_any |= _run_wrapper_selectors(X_df, y_arr, groups, fsc, rfecv_models, target_type, fs_random_seed, verbose, selected)
        except Exception as exc:
            logger.warning("LTR wrapper selectors (RFECV/BorutaShap) failed (%s: %s); skipping them.", type(exc).__name__, exc)

    if not ran_any:
        return None
    out = [c for c in X_df.columns if c in selected]
    if not out:
        logger.warning("LTR feature selection produced an empty union; keeping all features.")
        return list(X_df.columns)
    return out


def _run_wrapper_selectors(X_df, y_arr, groups, fsc, rfecv_models, target_type, fs_random_seed, verbose, selected) -> bool:
    """Run RFECV / BorutaShap via the main suite's _build_pre_pipelines and union their selections into ``selected``."""
    rfecv_models_params = {}
    if rfecv_models:
        from ..trainer import configure_training_params

        _, _, cb_rfecv, lgb_rfecv, xgb_rfecv, _, _ = configure_training_params(
            train_df=X_df, train_target=y_arr, target=y_arr, use_regression=True,
            mlframe_models=list(rfecv_models), prefer_gpu_configs=False, verbose=bool(verbose), rfecv_model_verbose=False,
        )
        rfecv_models_params = {"cb_rfecv": cb_rfecv, "lgb_rfecv": lgb_rfecv, "xgb_rfecv": xgb_rfecv}

    from ..core._setup_helpers_pre_pipelines import _build_pre_pipelines

    pre_pipelines, _names = _build_pre_pipelines(
        use_ordinary_models=False, rfecv_models=rfecv_models, rfecv_models_params=rfecv_models_params,
        use_mrmr_fs=False,  # MRMR handled by the group-aware path above
        mrmr_kwargs={}, use_boruta_shap=bool(getattr(fsc, "use_boruta_shap", False)),
        boruta_shap_kwargs=dict(getattr(fsc, "boruta_shap_kwargs", None) or {}),
        use_shap_proxied_fs=bool(getattr(fsc, "use_shap_proxied_fs", False)),
        shap_proxied_fs_kwargs=dict(getattr(fsc, "shap_proxied_fs_kwargs", None) or {}),
        rfecv_cluster_reduce=bool(getattr(fsc, "rfecv_cluster_reduce", True)),
        rfecv_cluster_corr_threshold=float(getattr(fsc, "rfecv_cluster_corr_threshold", 0.9)),
        rfecv_cluster_min_reduction=float(getattr(fsc, "rfecv_cluster_min_reduction", 0.05)),
        rfecv_cluster_corr_method=str(getattr(fsc, "rfecv_cluster_corr_method", "pearson")),
        target_type=target_type, fs_random_seed=fs_random_seed,
    )
    y_ser = pd.Series(y_arr, name="relevance")
    ran = False
    for pp in pre_pipelines:
        if pp is None:
            continue
        try:
            # RFECV accepts groups for GroupKFold (query-aware CV); BorutaShap ignores the kwarg.
            fit_kw = {}
            if groups is not None and getattr(pp, "_mlframe_selector_kind_", "") == "RFECV":
                fit_kw["groups"] = np.asarray(groups)
            Xt = pp.fit_transform(X_df, y_ser, **fit_kw)
            ran = True
            cols = list(Xt.columns) if hasattr(Xt, "columns") else None
            if cols:
                selected.update(c for c in cols if c in set(X_df.columns))
        except Exception as exc:
            logger.warning("LTR selector %s failed (%s: %s); skipping it.", type(pp).__name__, type(exc).__name__, exc)
    return ran
