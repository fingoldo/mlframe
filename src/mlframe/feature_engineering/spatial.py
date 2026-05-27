"""Spatial kNN aggregator features.

For each row in a query set, find the ``k`` nearest reference rows in
2-D / 3-D / N-D Euclidean coordinate space and emit aggregate
statistics (median / IQR / std / mean) of an arbitrary label column.

Distinct from ``feature_engineering.transformer.compute_row_attention``:
that module attends in HIGH-DIM FEATURE space (per-row representation
attention); this one attends in PHYSICAL COORDINATE space (geo / lat-
long / wellbore XY / 3D point cloud).

Leak-safe by default: the reference pool is supplied separately from
the query pool, so callers can pass e.g. ``ref_df = full[is_train]``
and ``q_df = full`` to get per-row features computed against ONLY
training rows (no test labels leaking into engineered features).

Same-group filter: pass ``q_group_ids`` + ``ref_group_ids`` to exclude
ref rows that share a group with the query row (typical wellbore /
panel-data leak guard: don't let row i's neighbours include other rows
from the SAME well).

Bucketed kNN: pass ``q_bucket`` + ``ref_bucket`` to restrict
neighbours to the SAME bucket value as the query. Useful for
azimuth-binned spatial kNN (only neighbours within ~20deg azimuth).
"""

from __future__ import annotations

__all__ = [
    "knn_aggregate",
    "knn_within_bucket_aggregate",
    "local_density_features",
    "inverse_distance_weighted_aggregate",
    "knn_label_dispersion_features",
    "radius_aggregate",
    "knn_gradient_features",
]

import logging
from typing import Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_AGG_FNS: tuple = ("median", "iqr", "std", "mean")


def _resolve_aggs(values: np.ndarray, agg_fns: Iterable[str]) -> dict:
    """Apply each named aggregator to ``values`` (axis=1) and return a dict."""
    out: dict = {}
    for name in agg_fns:
        if name == "median":
            out[name] = np.median(values, axis=1)
        elif name == "mean":
            out[name] = values.mean(axis=1)
        elif name == "std":
            out[name] = values.std(axis=1, ddof=1) if values.shape[1] > 1 else np.zeros(values.shape[0])
        elif name == "iqr":
            q25 = np.percentile(values, 25, axis=1)
            q75 = np.percentile(values, 75, axis=1)
            out[name] = q75 - q25
        elif name == "min":
            out[name] = values.min(axis=1)
        elif name == "max":
            out[name] = values.max(axis=1)
        elif name == "p10":
            out[name] = np.percentile(values, 10, axis=1)
        elif name == "p90":
            out[name] = np.percentile(values, 90, axis=1)
        else:
            raise ValueError(
                f"unknown agg_fn={name!r}; allowed: "
                "{'median','mean','std','iqr','min','max','p10','p90'}"
            )
    return out


def knn_aggregate(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    *,
    k: int = 10,
    agg_fns: Sequence[str] = _DEFAULT_AGG_FNS,
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
    distance_weighted: bool = False,
    weight_eps: float = 1.0,
    leaf_size: int = 40,
) -> dict:
    """For each query row, aggregate ``ref_labels`` over its k nearest neighbours.

    Parameters
    ----------
    query_coords
        ``(n_q, d)`` coordinate array (typically 2-D or 3-D).
    ref_coords
        ``(n_r, d)`` reference coordinates.
    ref_labels
        ``(n_r,)`` numeric labels (the value to aggregate over the
        neighbour ring). Non-finite entries are dropped from the ref
        pool BEFORE the KDTree build.
    k
        Neighbour count. The actual query asks for ``k + 1`` and skips
        the self-match if the same point appears in both pools (so
        ``query is ref`` is leak-safe by construction).
    agg_fns
        Iterable of aggregator names to compute. Default is the
        canonical four (median / iqr / std / mean).
    q_group_ids, ref_group_ids
        Per-row group identifiers. When BOTH supplied, neighbour
        candidates that share a group with the query row are EXCLUDED
        from the k-nearest selection. Use this for panel / clustered
        data where same-group rows leak.
    distance_weighted
        When True, aggregators use ``1 / (dist + weight_eps)`` weights
        instead of uniform. Currently only ``mean`` is weighted;
        median / IQR / std stay uniform (a weighted-median impl is
        future work).
    weight_eps
        Added to distance to avoid div-by-zero on coincident points.
    leaf_size
        sklearn KDTree leaf_size; default 40 matches sklearn default.

    Returns
    -------
    dict
        ``{agg_name: np.ndarray of shape (n_q,)}`` for every requested
        aggregator, plus ``"_nearest_distance": np.ndarray (n_q,)``
        (distance to k=1 neighbour after group filtering, for the
        common "nearest-neighbour distance" feature).
    """
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError(
            "knn_aggregate requires scikit-learn (KDTree). Install via "
            "`pip install scikit-learn`."
        ) from e

    ref_coords = np.ascontiguousarray(ref_coords, dtype=np.float64)
    ref_labels = np.ascontiguousarray(ref_labels, dtype=np.float64)
    query_coords = np.ascontiguousarray(query_coords, dtype=np.float64)
    if ref_coords.shape[0] != ref_labels.shape[0]:
        raise ValueError(
            f"ref_coords rows {ref_coords.shape[0]} != ref_labels "
            f"len {ref_labels.shape[0]}"
        )
    # Drop non-finite ref rows so KDTree gets a clean pool.
    ref_finite_mask = (
        np.isfinite(ref_coords).all(axis=1) & np.isfinite(ref_labels)
    )
    if not ref_finite_mask.all():
        ref_coords = ref_coords[ref_finite_mask]
        ref_labels = ref_labels[ref_finite_mask]
        if ref_group_ids is not None:
            ref_group_ids = np.asarray(ref_group_ids)[ref_finite_mask]

    if ref_coords.shape[0] < k + 1:
        raise ValueError(
            f"need >= k+1 ({k + 1}) finite reference rows; got "
            f"{ref_coords.shape[0]}"
        )

    tree = KDTree(ref_coords, leaf_size=leaf_size)

    # Decide query k. With group filtering we may need to overquery to
    # have enough usable neighbours after dropping same-group hits.
    q_k = k + 1
    if q_group_ids is not None and ref_group_ids is not None:
        # Cap overquery at the ref pool size; double k as the
        # heuristic safe budget for typical panel data (1-5 group
        # members on average).
        q_k = min(ref_coords.shape[0], k * 4 + 1)

    # Replace non-finite query coords with 0; we'll fill output with
    # NaN at the end for those rows. (KDTree.query rejects non-finite.)
    q_finite = np.isfinite(query_coords).all(axis=1)
    q_clean = np.where(np.isfinite(query_coords), query_coords, 0.0)
    distances, indices = tree.query(q_clean, k=q_k)

    n_q = query_coords.shape[0]

    if q_group_ids is not None and ref_group_ids is not None:
        q_group_ids = np.asarray(q_group_ids)
        ref_group_ids = np.asarray(ref_group_ids)
        # mask[i, j] = True iff neighbour j of query i belongs to a
        # different group than query i.
        same_group = ref_group_ids[indices] == q_group_ids[:, None]
        # Compact: for each row, take first k columns where same_group=False
        sorted_mask = np.where(same_group, np.iinfo(np.int64).max, np.arange(q_k))
        order = np.argsort(sorted_mask, axis=1)
        keep_idx = order[:, :k]
        compact_indices = np.take_along_axis(indices, keep_idx, axis=1)
        compact_dist = np.take_along_axis(distances, keep_idx, axis=1)
        compact_mask = ~np.take_along_axis(same_group, keep_idx, axis=1)
        # Where mask is False (no valid neighbour for that slot),
        # carry NaN; aggregators below handle NaN via np.nanmean / etc.
        labels_arr = np.where(
            compact_mask, ref_labels[compact_indices], np.nan,
        )
    else:
        compact_indices = indices[:, :k]
        compact_dist = distances[:, :k]
        labels_arr = ref_labels[compact_indices]

    # Aggregate. Use nan-aware variants if same-group filter introduced NaN.
    if q_group_ids is not None:
        out_aggs: dict = {}
        for name in agg_fns:
            if name == "median":
                out_aggs[name] = np.nanmedian(labels_arr, axis=1)
            elif name == "mean":
                if distance_weighted:
                    w = 1.0 / (compact_dist + weight_eps)
                    w = np.where(np.isfinite(labels_arr), w, 0.0)
                    out_aggs[name] = (
                        (np.where(np.isfinite(labels_arr), labels_arr, 0.0) * w).sum(axis=1)
                        / (w.sum(axis=1) + 1e-12)
                    )
                else:
                    out_aggs[name] = np.nanmean(labels_arr, axis=1)
            elif name == "std":
                out_aggs[name] = np.nanstd(labels_arr, axis=1, ddof=1)
            elif name == "iqr":
                q25 = np.nanpercentile(labels_arr, 25, axis=1)
                q75 = np.nanpercentile(labels_arr, 75, axis=1)
                out_aggs[name] = q75 - q25
            elif name == "min":
                out_aggs[name] = np.nanmin(labels_arr, axis=1)
            elif name == "max":
                out_aggs[name] = np.nanmax(labels_arr, axis=1)
            elif name == "p10":
                out_aggs[name] = np.nanpercentile(labels_arr, 10, axis=1)
            elif name == "p90":
                out_aggs[name] = np.nanpercentile(labels_arr, 90, axis=1)
            else:
                raise ValueError(f"unknown agg_fn={name!r}")
        out_aggs["_nearest_distance"] = compact_dist[:, 0]
    else:
        out_aggs = _resolve_aggs(labels_arr, agg_fns)
        if distance_weighted and "mean" in agg_fns:
            w = 1.0 / (compact_dist + weight_eps)
            out_aggs["mean"] = (labels_arr * w).sum(axis=1) / (w.sum(axis=1) + 1e-12)
        out_aggs["_nearest_distance"] = compact_dist[:, 0]

    # Fill non-finite-query rows with NaN.
    if not q_finite.all():
        for k_, v in out_aggs.items():
            out_aggs[k_] = np.where(q_finite, v, np.nan)
    return out_aggs


def knn_within_bucket_aggregate(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    *,
    q_bucket: np.ndarray,
    ref_bucket: np.ndarray,
    k: int = 5,
    agg_fns: Sequence[str] = _DEFAULT_AGG_FNS,
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
) -> dict:
    """kNN aggregator restricted to a per-row bucket match.

    For each query, only ref rows with ``ref_bucket == q_bucket`` are
    eligible as neighbours. Useful when the bucket encodes a categorical
    similarity dimension that geometric distance doesn't capture (e.g.
    azimuth bucket, formation label, season, customer segment).

    Builds one KDTree per unique bucket value (lazy on demand) so the
    per-query work stays sub-linear in ref size. Single-pass over the
    query array.

    Same group filter (``q_group_ids`` + ``ref_group_ids``) layers on top.
    """
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError("knn_within_bucket_aggregate requires scikit-learn") from e

    ref_coords = np.ascontiguousarray(ref_coords, dtype=np.float64)
    ref_labels = np.ascontiguousarray(ref_labels, dtype=np.float64)
    query_coords = np.ascontiguousarray(query_coords, dtype=np.float64)
    q_bucket = np.asarray(q_bucket)
    ref_bucket = np.asarray(ref_bucket)

    if ref_coords.shape[0] != len(ref_bucket):
        raise ValueError("ref_coords / ref_bucket length mismatch")
    if query_coords.shape[0] != len(q_bucket):
        raise ValueError("query_coords / q_bucket length mismatch")

    # Drop non-finite refs
    ref_finite = (
        np.isfinite(ref_coords).all(axis=1)
        & np.isfinite(ref_labels)
    )
    ref_coords = ref_coords[ref_finite]
    ref_labels = ref_labels[ref_finite]
    ref_bucket = ref_bucket[ref_finite]
    if ref_group_ids is not None:
        ref_group_ids = np.asarray(ref_group_ids)[ref_finite]

    n_q = query_coords.shape[0]

    # Pre-bucket the ref pool. For each bucket value -> KDTree + label
    # subset + group_ids subset.
    bucket_to_tree: dict = {}
    bucket_to_labels: dict = {}
    bucket_to_groups: dict = {}
    for b in np.unique(ref_bucket):
        mask = ref_bucket == b
        if mask.sum() < k:
            # Too few candidates to satisfy k-query; skip this bucket.
            continue
        bucket_to_tree[b] = KDTree(ref_coords[mask])
        bucket_to_labels[b] = ref_labels[mask]
        if ref_group_ids is not None:
            bucket_to_groups[b] = ref_group_ids[mask]

    out_aggs: dict = {name: np.full(n_q, np.nan, dtype=np.float64) for name in agg_fns}
    out_aggs["_nearest_distance"] = np.full(n_q, np.nan, dtype=np.float64)

    # Iterate unique query buckets so we batch tree.query within bucket.
    for b in np.unique(q_bucket):
        q_idx = np.where(q_bucket == b)[0]
        if b not in bucket_to_tree:
            continue
        tree = bucket_to_tree[b]
        labels_b = bucket_to_labels[b]
        n_ref_b = labels_b.size
        q_k = min(n_ref_b, k + 1)
        q_clean = np.where(
            np.isfinite(query_coords[q_idx]),
            query_coords[q_idx], 0.0,
        )
        distances, indices = tree.query(q_clean, k=q_k)
        if q_group_ids is not None and b in bucket_to_groups:
            ref_g_b = bucket_to_groups[b]
            q_g_b = np.asarray(q_group_ids)[q_idx]
            same_group = ref_g_b[indices] == q_g_b[:, None]
            sorted_mask = np.where(same_group, np.iinfo(np.int64).max, np.arange(q_k))
            order = np.argsort(sorted_mask, axis=1)
            keep_idx = order[:, :k]
            compact_indices = np.take_along_axis(indices, keep_idx, axis=1)
            compact_dist = np.take_along_axis(distances, keep_idx, axis=1)
            compact_mask = ~np.take_along_axis(same_group, keep_idx, axis=1)
            labels_arr = np.where(compact_mask, labels_b[compact_indices], np.nan)
            use_nan = True
        else:
            compact_indices = indices[:, :k]
            compact_dist = distances[:, :k]
            labels_arr = labels_b[compact_indices]
            use_nan = False
        for name in agg_fns:
            if name == "median":
                out_aggs[name][q_idx] = np.nanmedian(labels_arr, axis=1) if use_nan else np.median(labels_arr, axis=1)
            elif name == "mean":
                out_aggs[name][q_idx] = np.nanmean(labels_arr, axis=1) if use_nan else labels_arr.mean(axis=1)
            elif name == "std":
                out_aggs[name][q_idx] = (
                    np.nanstd(labels_arr, axis=1, ddof=1) if use_nan
                    else labels_arr.std(axis=1, ddof=1) if labels_arr.shape[1] > 1
                    else np.zeros(labels_arr.shape[0])
                )
            elif name == "iqr":
                q25 = np.nanpercentile(labels_arr, 25, axis=1) if use_nan else np.percentile(labels_arr, 25, axis=1)
                q75 = np.nanpercentile(labels_arr, 75, axis=1) if use_nan else np.percentile(labels_arr, 75, axis=1)
                out_aggs[name][q_idx] = q75 - q25
            else:
                raise ValueError(f"agg_fn={name!r} not supported in bucket mode")
        out_aggs["_nearest_distance"][q_idx] = compact_dist[:, 0]
    return out_aggs


def local_density_features(
    q_coords: np.ndarray,
    ref_coords: np.ndarray,
    *,
    k: int = 10,
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
    leaf_size: int = 40,
) -> dict:
    """Distance / density features computed FROM neighbour distances only.

    Returns dict with:
    * ``dist_to_kth`` — distance to the k-th nearest neighbour
    * ``dist_median`` — median pairwise distance to k neighbours
    * ``dist_iqr`` — IQR of distances to k neighbours
    * ``local_density`` — ``k / (pi * dist_to_kth^d)`` (d-dim ball
      density estimator)

    Free-lunch outlier signal and a sparse/dense submarket detector
    — no label dependency. Used in real estate (regime by density),
    epidemiology (population proxy), panel data (cluster size for
    hierarchical effects).
    """
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError("local_density_features requires scikit-learn") from e
    ref = np.ascontiguousarray(ref_coords, dtype=np.float64)
    q = np.ascontiguousarray(q_coords, dtype=np.float64)
    if q.ndim != 2 or ref.ndim != 2 or q.shape[1] != ref.shape[1]:
        raise ValueError("q_coords / ref_coords must both be 2-D with matching d")
    d = ref.shape[1]
    finite_ref = np.isfinite(ref).all(axis=1)
    ref = ref[finite_ref]
    if ref_group_ids is not None:
        ref_group_ids = np.asarray(ref_group_ids)[finite_ref]
    if ref.shape[0] < k + 1:
        raise ValueError(f"need >= k+1 ({k+1}) finite refs; got {ref.shape[0]}")
    tree = KDTree(ref, leaf_size=leaf_size)
    q_clean = np.where(np.isfinite(q), q, 0.0)
    q_k = k + 1 if (q_group_ids is not None and ref_group_ids is not None) else k + 1
    if q_group_ids is not None and ref_group_ids is not None:
        # overquery to filter same-group
        q_k = min(ref.shape[0], k * 4 + 1)
    distances, indices = tree.query(q_clean, k=min(q_k, ref.shape[0]))
    if q_group_ids is not None and ref_group_ids is not None:
        q_group_ids = np.asarray(q_group_ids)
        ref_group_ids = np.asarray(ref_group_ids)
        same_group = ref_group_ids[indices] == q_group_ids[:, None]
        sorted_mask = np.where(same_group, np.iinfo(np.int64).max, np.arange(distances.shape[1]))
        order = np.argsort(sorted_mask, axis=1)
        keep_idx = order[:, :k]
        compact_dist = np.take_along_axis(distances, keep_idx, axis=1)
    else:
        compact_dist = distances[:, :k]

    dist_to_kth = compact_dist[:, -1]
    dist_median = np.median(compact_dist, axis=1)
    q25 = np.quantile(compact_dist, 0.25, axis=1)
    q75 = np.quantile(compact_dist, 0.75, axis=1)
    dist_iqr = q75 - q25
    # d-dim ball volume up to constant: density = k / r^d (drop the
    # pi/Gamma constant; it's the same for every row -> meaningless to ML).
    local_density = float(k) / (dist_to_kth ** d + 1e-12)
    return {
        "dist_to_kth": dist_to_kth,
        "dist_median": dist_median,
        "dist_iqr": dist_iqr,
        "local_density": local_density,
    }


def inverse_distance_weighted_aggregate(
    q_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    *,
    k: int = 10,
    power: float = 2.0,
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
) -> dict:
    """IDW (inverse-distance-weighted) interpolation as a feature.

    Geostat classic; dominates uniform-mean kNN when distance carries
    decay information. Returns dict with:
    * ``idw`` — IDW estimate (weights ``1 / dist^power``)
    * ``idw_loo_residual`` — leave-one-out residual of the closest
      neighbour (cheap uncertainty surrogate without kriging variance)

    Use cases: weather (temp/precip from station network — IDW is the
    operational baseline), real estate (price-per-sqft from comps with
    distance decay), epi (incidence smoothing across reporting units).
    """
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError("inverse_distance_weighted_aggregate requires scikit-learn") from e
    ref = np.ascontiguousarray(ref_coords, dtype=np.float64)
    q = np.ascontiguousarray(q_coords, dtype=np.float64)
    labels = np.ascontiguousarray(ref_labels, dtype=np.float64)
    finite_ref = np.isfinite(ref).all(axis=1) & np.isfinite(labels)
    ref = ref[finite_ref]
    labels = labels[finite_ref]
    if ref_group_ids is not None:
        ref_group_ids = np.asarray(ref_group_ids)[finite_ref]
    if ref.shape[0] < k + 1:
        raise ValueError(f"need >= k+1 finite refs; got {ref.shape[0]}")
    tree = KDTree(ref)
    q_clean = np.where(np.isfinite(q), q, 0.0)
    q_k = (k * 4 + 1) if (q_group_ids is not None and ref_group_ids is not None) else (k + 1)
    distances, indices = tree.query(q_clean, k=min(q_k, ref.shape[0]))
    if q_group_ids is not None and ref_group_ids is not None:
        q_group_ids = np.asarray(q_group_ids)
        ref_group_ids = np.asarray(ref_group_ids)
        same_group = ref_group_ids[indices] == q_group_ids[:, None]
        sorted_mask = np.where(same_group, np.iinfo(np.int64).max, np.arange(distances.shape[1]))
        order = np.argsort(sorted_mask, axis=1)
        keep_idx = order[:, :k]
        compact_indices = np.take_along_axis(indices, keep_idx, axis=1)
        compact_dist = np.take_along_axis(distances, keep_idx, axis=1)
    else:
        compact_indices = indices[:, :k]
        compact_dist = distances[:, :k]
    weights = 1.0 / (compact_dist ** power + 1e-12)
    w_sum = weights.sum(axis=1, keepdims=True) + 1e-12
    weights_norm = weights / w_sum
    label_arr = labels[compact_indices]
    idw = (label_arr * weights_norm).sum(axis=1)
    # Leave-one-out residual: predict from neighbours 1..k-1 (drop the
    # nearest), compare to the nearest's label.
    if k >= 2:
        loo_w = weights[:, 1:]
        loo_sum = loo_w.sum(axis=1, keepdims=True) + 1e-12
        loo_pred = (label_arr[:, 1:] * (loo_w / loo_sum)).sum(axis=1)
        loo_residual = label_arr[:, 0] - loo_pred
    else:
        loo_residual = np.full(q.shape[0], np.nan, dtype=np.float64)
    return {"idw": idw, "idw_loo_residual": loo_residual}


def knn_label_dispersion_features(
    q_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    *,
    k: int = 10,
    task: str = "regression",
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
    n_bins: int = 8,
) -> dict:
    """Label-uncertainty signals over the kNN ring.

    Returns dict with:
    * ``local_label_entropy`` — Shannon entropy of the (binned for
      regression / class-frequency for classification) neighbour labels.
    * ``local_disagreement_ratio`` — fraction of neighbours on the
      opposite side of the GLOBAL median (regression) / fraction of
      neighbours with NON-majority class (classification).
    * ``local_majority_share`` — largest fraction in any bin / class.

    Cheap regime detector: stable submarket vs heterogeneous boundary.
    """
    if task not in {"regression", "classification"}:
        raise ValueError(f"task={task!r}")
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError("knn_label_dispersion_features requires scikit-learn") from e
    ref = np.ascontiguousarray(ref_coords, dtype=np.float64)
    q = np.ascontiguousarray(q_coords, dtype=np.float64)
    if task == "regression":
        labels = np.ascontiguousarray(ref_labels, dtype=np.float64)
        finite_ref = np.isfinite(ref).all(axis=1) & np.isfinite(labels)
        global_median = float(np.median(labels[np.isfinite(labels)]))
    else:
        labels = np.asarray(ref_labels)
        finite_ref = np.isfinite(ref).all(axis=1)
    ref = ref[finite_ref]
    labels = labels[finite_ref]
    if ref_group_ids is not None:
        ref_group_ids = np.asarray(ref_group_ids)[finite_ref]
    tree = KDTree(ref)
    q_clean = np.where(np.isfinite(q), q, 0.0)
    q_k = (k * 4 + 1) if (q_group_ids is not None and ref_group_ids is not None) else (k + 1)
    distances, indices = tree.query(q_clean, k=min(q_k, ref.shape[0]))
    if q_group_ids is not None and ref_group_ids is not None:
        q_group_ids = np.asarray(q_group_ids)
        ref_group_ids = np.asarray(ref_group_ids)
        same_group = ref_group_ids[indices] == q_group_ids[:, None]
        sorted_mask = np.where(same_group, np.iinfo(np.int64).max, np.arange(distances.shape[1]))
        order = np.argsort(sorted_mask, axis=1)
        keep_idx = order[:, :k]
        compact_indices = np.take_along_axis(indices, keep_idx, axis=1)
    else:
        compact_indices = indices[:, :k]
    label_arr = labels[compact_indices]

    n_q = q.shape[0]
    entropy = np.full(n_q, np.nan, dtype=np.float64)
    disagreement = np.full(n_q, np.nan, dtype=np.float64)
    majority_share = np.full(n_q, np.nan, dtype=np.float64)

    if task == "regression":
        # Quantile-bin globally for stability.
        finite_lab = labels[np.isfinite(labels)]
        if finite_lab.size > 0:
            edges = np.quantile(finite_lab, np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if edges.size >= 2:
                # Per-row binned histogram.
                binned = np.searchsorted(edges[1:-1], label_arr, side="right")
                # For each row count occurrences.
                for i in range(n_q):
                    bc = np.bincount(binned[i], minlength=max(2, edges.size - 1))
                    p = bc.astype(np.float64) / bc.sum()
                    p_nz = p[p > 0]
                    entropy[i] = float(-np.sum(p_nz * np.log(p_nz)))
                    majority_share[i] = float(p.max())
                # Disagreement: fraction of neighbour labels on opposite side
                # of the global median.
                opp = (label_arr > global_median) != (label_arr[:, :1] > global_median)
                disagreement = opp.mean(axis=1)
    else:
        # Classification: per-row class fraction.
        for i in range(n_q):
            row = label_arr[i]
            uniq, counts = np.unique(row, return_counts=True)
            p = counts.astype(np.float64) / counts.sum()
            p_nz = p[p > 0]
            entropy[i] = float(-np.sum(p_nz * np.log(p_nz)))
            majority_share[i] = float(p.max())
            # Disagreement: fraction NOT equal to nearest neighbour's label.
            disagreement[i] = float((row != row[0]).mean())
    return {
        "local_label_entropy": entropy,
        "local_disagreement_ratio": disagreement,
        "local_majority_share": majority_share,
    }


def radius_aggregate(
    q_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    *,
    radius: float,
    min_count: int = 3,
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
) -> dict:
    """Fixed-radius (not fixed-k) neighbour aggregator.

    Returns dict with:
    * ``n_within_r`` — count of ref points inside radius
    * ``mean_within_r`` / ``median_within_r`` — label aggregators (NaN
      where ``n_within_r < min_count``)

    Use cases: regulatory "comps within X miles", contact-tracing
    catchment, station-network averaging where station density varies.
    Distinct from kNN aggregator which masks density variation.
    """
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError("radius_aggregate requires scikit-learn") from e
    ref = np.ascontiguousarray(ref_coords, dtype=np.float64)
    q = np.ascontiguousarray(q_coords, dtype=np.float64)
    labels = np.ascontiguousarray(ref_labels, dtype=np.float64)
    finite_ref = np.isfinite(ref).all(axis=1) & np.isfinite(labels)
    ref = ref[finite_ref]
    labels = labels[finite_ref]
    if ref_group_ids is not None:
        ref_group_ids = np.asarray(ref_group_ids)[finite_ref]
    if ref.shape[0] == 0:
        raise ValueError("ref pool is empty after non-finite filtering")
    tree = KDTree(ref)
    q_clean = np.where(np.isfinite(q), q, 0.0)
    idx_per_row = tree.query_radius(q_clean, r=radius)
    n_q = q.shape[0]
    n_within = np.zeros(n_q, dtype=np.float64)
    mean_in = np.full(n_q, np.nan, dtype=np.float64)
    median_in = np.full(n_q, np.nan, dtype=np.float64)
    for i, ids in enumerate(idx_per_row):
        if ids.size == 0:
            continue
        if q_group_ids is not None and ref_group_ids is not None:
            mask = ref_group_ids[ids] != np.asarray(q_group_ids)[i]
            ids = ids[mask]
        n_within[i] = float(ids.size)
        if ids.size >= min_count:
            lab = labels[ids]
            mean_in[i] = float(lab.mean())
            median_in[i] = float(np.median(lab))
    return {
        "n_within_r": n_within,
        "mean_within_r": mean_in,
        "median_within_r": median_in,
    }


def knn_gradient_features(
    q_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    *,
    k: int = 20,
    q_group_ids: Optional[np.ndarray] = None,
    ref_group_ids: Optional[np.ndarray] = None,
) -> dict:
    """Local linear gradient via WLS fit on the kNN ring.

    Fit ``label ~ 1 + dx_1 + dx_2 + ... + dx_d`` over the k nearest
    reference rows (weights ``1 / (dist + eps)``). Emits:
    * ``grad_norm`` — magnitude of gradient vector
    * ``grad_axis_0``, ``grad_axis_1``, ... — per-axis slopes
    * ``wls_residual_std`` — std of fit residuals (local curvature)

    Use cases: real estate price gradient toward CBD/waterfront,
    weather temperature / pressure gradient (physically meaningful),
    epi incidence gradient = wavefront velocity proxy.
    """
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise ImportError("knn_gradient_features requires scikit-learn") from e
    ref = np.ascontiguousarray(ref_coords, dtype=np.float64)
    q = np.ascontiguousarray(q_coords, dtype=np.float64)
    labels = np.ascontiguousarray(ref_labels, dtype=np.float64)
    d = ref.shape[1]
    finite_ref = np.isfinite(ref).all(axis=1) & np.isfinite(labels)
    ref = ref[finite_ref]; labels = labels[finite_ref]
    if ref_group_ids is not None:
        ref_group_ids = np.asarray(ref_group_ids)[finite_ref]
    if ref.shape[0] < k + 1:
        raise ValueError(f"need >= k+1 refs; got {ref.shape[0]}")
    tree = KDTree(ref)
    q_clean = np.where(np.isfinite(q), q, 0.0)
    q_k = (k * 4 + 1) if (q_group_ids is not None and ref_group_ids is not None) else (k + 1)
    distances, indices = tree.query(q_clean, k=min(q_k, ref.shape[0]))
    if q_group_ids is not None and ref_group_ids is not None:
        q_group_ids = np.asarray(q_group_ids)
        ref_group_ids = np.asarray(ref_group_ids)
        same_group = ref_group_ids[indices] == q_group_ids[:, None]
        sorted_mask = np.where(same_group, np.iinfo(np.int64).max, np.arange(distances.shape[1]))
        order = np.argsort(sorted_mask, axis=1)
        keep_idx = order[:, :k]
        compact_indices = np.take_along_axis(indices, keep_idx, axis=1)
        compact_dist = np.take_along_axis(distances, keep_idx, axis=1)
    else:
        compact_indices = indices[:, :k]
        compact_dist = distances[:, :k]

    n_q = q.shape[0]
    grads = np.full((n_q, d), np.nan, dtype=np.float64)
    grad_norm = np.full(n_q, np.nan, dtype=np.float64)
    resid_std = np.full(n_q, np.nan, dtype=np.float64)

    # Solve weighted normal equations per row. With small k (~20) and
    # d (~2-3), Python loop is fine.
    for i in range(n_q):
        if not np.isfinite(q[i]).all():
            continue
        nbrs = compact_indices[i]
        x_diff = ref[nbrs] - q[i]   # shape (k, d)
        y = labels[nbrs]            # shape (k,)
        w = 1.0 / (compact_dist[i] + 1.0)
        # Design matrix [1, dx_1, ..., dx_d]
        X = np.concatenate([np.ones((k, 1)), x_diff], axis=1)
        Xw = X * w[:, None]
        yw = y * w
        # Solve normal eq (X^T W X) b = X^T W y
        XtX = X.T @ Xw
        Xty = X.T @ yw
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue
        grads[i] = beta[1:]
        grad_norm[i] = float(np.linalg.norm(beta[1:]))
        # Residual std (unweighted; gives curvature/heterogeneity proxy)
        pred = X @ beta
        resid_std[i] = float(np.std(y - pred, ddof=1)) if k > 1 else 0.0

    out = {
        "grad_norm": grad_norm,
        "wls_residual_std": resid_std,
    }
    for ax in range(d):
        out[f"grad_axis_{ax}"] = grads[:, ax]
    return out
