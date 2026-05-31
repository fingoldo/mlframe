"""Symmetric-Uncertainty pairwise clustering for ShapProxiedFS.

Sibling module of ``_shap_proxy_cluster.py`` (Pearson-based clustering). Reuses
the pre-binned per-column class arrays produced by MRMR's ``categorize_dataset``
(surfaced via ``MRMR.export_artifacts()['bins']``) so two features are linked
into the same cluster when ``SU(X_i, X_j) >= threshold``. Catches non-linear
redundancy (XOR, saddle, sinusoidal) that the Pearson backend misses because
``|corr|`` is near zero on those relationships even when full mutual
information is high.

The return contract mirrors ``cluster_correlated_features``: an ``np.ndarray``
of cluster labels with shape ``(n_features,)`` indexed 0..K-1 contiguous, so
the downstream ``build_unit_matrix`` / ``cluster_summary`` callers stay
unchanged.

Reuses ``compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y)`` from
``filters.info_theory`` (numba-cached) — same SU primitive the MRMR screen and
DCD pairwise SU branch run. The marginal entropies (and bincount frequencies)
are cached per column inside one ``cluster_correlated_features_su`` call so
each column's marginal pass runs once even though the column appears in
``n_features - 1`` pairs.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
from numba import njit, prange

from mlframe.feature_selection._shap_proxy_cluster import _uf_labels

logger = logging.getLogger(__name__)


def _resolve_parallel_min_features(default: int = 50) -> int:
    """Smallest feature count at which the parallel prange kernel beats the serial path.

    Below this width the per-pair work is small enough that prange thread-spawn dwarfs
    the saved CPU time. Above it, the O(f^2) pair count scales the wall and parallel
    pays off. The default (50) is dispatcher-tunable per HW via
    ``pyutilz.system.kernel_tuning_cache`` (key
    ``mlframe.shap_proxied_fs.cluster_su.parallel_min_features``).
    """
    try:
        from pyutilz.system import kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su.parallel_min_features", default=default)
        return int(value)
    except Exception:
        return default


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def _pairwise_su_edges(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    freqs_packed: np.ndarray,
    freqs_offsets: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Pairwise SU matrix above ``threshold`` returned as a dense flag matrix.

    ``bins_packed`` is a ``(n_features, n_samples)`` int32 view (column-major
    layout: each feature's per-sample bin ids occupy a contiguous row, so the
    inner sample-scan reads two contiguous int32 strips and saturates the L1
    cache line instead of jumping ``n_features * 4`` bytes per sample);
    ``nbins[i]`` is the cardinality used to size the joint-counts matrix on
    column ``i``; ``freqs_packed`` is the concatenation of all per-column
    marginal probability vectors with offsets in ``freqs_offsets`` (shape
    ``(n_features + 1,)``); ``h_marginals[i]`` is the pre-computed Shannon
    entropy of column ``i``; ``constant_mask[i]`` is ``True`` when column ``i``
    has <=1 distinct bin (SU=0 vs anyone).

    Returns an upper-triangle flag matrix (``flag[i, j] = 1`` iff ``SU(i, j) >= threshold``
    and ``i < j``). Edge extraction happens outside the njit kernel so the prange
    iterations stay purely numeric.
    """
    n_features, n_samples = bins_packed.shape
    flags = np.zeros((n_features, n_features), dtype=np.uint8)
    # max joint cardinality controls a single thread-local reusable buffer per
    # outer iteration; avoids per-pair np.zeros allocation that bottlenecks at
    # width >= 2000 (numba memory allocator under contention).
    max_nb = 0
    for i in range(n_features):
        if nbins[i] > max_nb:
            max_nb = nbins[i]
    for i in prange(n_features):
        if constant_mask[i]:
            continue
        nb_i = nbins[i]
        h_i = h_marginals[i]
        off_i = freqs_offsets[i]
        # one int64 buffer per outer-i (thread-local because prange allocates
        # locals inside the parallel region on the worker thread's stack).
        joint = np.zeros((max_nb, max_nb), dtype=np.int64)
        for j in range(i + 1, n_features):
            if constant_mask[j]:
                continue
            nb_j = nbins[j]
            # reset only the cells we'll touch.
            for a in range(nb_i):
                for b in range(nb_j):
                    joint[a, b] = 0
            for k in range(n_samples):
                joint[bins_packed[i, k], bins_packed[j, k]] += 1
            inv_n = 1.0 / n_samples
            mi = 0.0
            off_j = freqs_offsets[j]
            for a in range(nb_i):
                px = freqs_packed[off_i + a]
                if px <= 0.0:
                    continue
                for b in range(nb_j):
                    jc = joint[a, b]
                    if jc == 0:
                        continue
                    py = freqs_packed[off_j + b]
                    if py <= 0.0:
                        continue
                    jf = jc * inv_n
                    mi += jf * math.log(jf / (px * py))
            denom = h_i + h_marginals[j]
            if denom <= 1e-12:
                continue
            su = 2.0 * mi / denom
            if su >= threshold:
                flags[i, j] = 1
    return flags


def _pack_bins_for_kernel(
    arrays: list[np.ndarray],
    marginals: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pack per-column bin arrays + marginals into the contiguous buffers the kernel reads.

    Returns ``None`` when columns have heterogeneous sample counts (the dict layout
    that MRMR's ``categorize_dataset`` produces always shares ``n_samples`` so this
    is just a defensive guard). Each returned array is C-contiguous and dtyped to
    match the kernel's signature exactly so numba doesn't re-specialize per call.
    """
    if not arrays:
        return None
    n_samples = int(arrays[0].shape[0])
    for a in arrays[1:]:
        if int(a.shape[0]) != n_samples:
            return None
    n_features = len(arrays)
    # Column-major layout: feature i's per-sample bin ids live in row i so the
    # kernel's inner sample loop reads two contiguous int32 strips
    # (bins_packed[i, :] and bins_packed[j, :]) — one cache line load per ~16
    # samples instead of one per sample under the prior (n_samples, n_features)
    # row-major layout where the i and j columns lived on different cache lines.
    bins_packed = np.empty((n_features, n_samples), dtype=np.int32, order="C")
    nbins = np.empty(n_features, dtype=np.int64)
    freqs_offsets = np.empty(n_features + 1, dtype=np.int64)
    h_marginals = np.empty(n_features, dtype=np.float64)
    constant_mask = np.empty(n_features, dtype=np.bool_)

    total_freqs = 0
    for i, (_classes_unused, freqs) in enumerate(marginals):
        total_freqs += int(freqs.shape[0])
    freqs_packed = np.empty(total_freqs, dtype=np.float64)
    offset = 0
    for i, (classes_i, freqs_i) in enumerate(marginals):
        # classes_i is int64 by construction in _column_marginal; downcast to the
        # int32 packed dtype (bin ids are tiny, never overflow int32). Writing
        # the whole row at once stays sequential in the destination buffer.
        bins_packed[i, :] = classes_i.astype(np.int32, copy=False)
        nb = int(freqs_i.shape[0])
        nbins[i] = nb
        freqs_offsets[i] = offset
        freqs_packed[offset:offset + nb] = freqs_i
        offset += nb
        constant_mask[i] = nb <= 1
        h = 0.0
        for p in freqs_i:
            if p > 0.0:
                h -= float(p) * math.log(float(p))
        h_marginals[i] = h
    freqs_offsets[n_features] = offset
    return bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask


def _resolve_columns(
    bins: dict[str, np.ndarray],
    feature_names: Iterable[str] | None,
) -> tuple[list[str], list[np.ndarray]]:
    """Materialise an ordered (names, bin-arrays) view of the bins dict.

    ``feature_names`` pins the iteration order so the returned labels array is
    axis-aligned with the caller's ``X_search.columns``. ``None`` falls back to
    ``bins.keys()`` order (insertion order; only safe when the caller has not
    re-ordered the columns since artifact export).
    """
    if feature_names is None:
        names = list(bins.keys())
    else:
        names = [n for n in feature_names if n in bins]
    arrays = [np.ascontiguousarray(bins[n]) for n in names]
    return names, arrays


def _column_marginal(
    classes: np.ndarray,
    n_bins_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(classes_int64, freqs_float64)`` ready for ``compute_su_from_classes``.

    The frequency array is a probability vector (sums to 1 if ``classes.size > 0``);
    its length matches the maximum observed bin id + 1, or ``n_bins_hint`` if that
    is larger (keeps the joint-counts matrix dimension matching across pairs even
    when one column happens to never realise its highest bin in this sample).
    """
    cls = np.ascontiguousarray(classes, dtype=np.int64)
    if cls.size == 0:
        return cls, np.empty(0, dtype=np.float64)
    observed_max = int(cls.max()) + 1 if cls.size else 0
    nb = max(observed_max, int(n_bins_hint) if n_bins_hint else 0)
    counts = np.bincount(cls, minlength=nb).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return cls, counts
    return cls, counts / total


def cluster_correlated_features_su(
    bins: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
    feature_names: Iterable[str] | None = None,
    nbins_per_feature: dict[str, int] | None = None,
    edge_cap: int = 20_000_000,
    use_parallel: bool = True,
    parallel_min_features: int | None = None,
) -> np.ndarray:
    """Cluster features by single-linkage on ``SU(X_i, X_j) >= threshold``.

    Mirrors ``_shap_proxy_cluster.cluster_correlated_features``'s return type
    (``np.ndarray`` shape ``(n_features,)`` dtype int64, contiguous 0..K-1
    cluster ids) so the downstream ``build_unit_matrix`` consumer is reused
    verbatim.

    Parameters
    ----------
    bins
        ``feature_name -> per-row int bin labels`` (e.g.
        ``MRMR.export_artifacts()['bins']`` after ``restrict_artifacts``).
    threshold
        SU cutoff in [0, 1]. Pairs with ``SU >= threshold`` are linked. The
        scale differs from Pearson ``|corr|``: SU is bounded by 1 but reaches
        it only for deterministic relationships; mid-strong dependencies
        cluster around SU ~0.4-0.6. The default 0.5 is calibrated to roughly
        match the linking density of Pearson at ``|corr| >= 0.7``.
    feature_names
        Ordering pin. When provided, the returned labels array indexes against
        this ordering (typically ``X_search.columns``). ``None`` falls back to
        ``bins.keys()``.
    nbins_per_feature
        Optional ``feature_name -> bin count`` hint. When supplied, the
        marginal frequency arrays use this length even if the column never
        realises its highest bin in the sample, keeping shapes consistent
        with the MRMR screen's view.
    edge_cap
        Reject the clustering if more than this many above-threshold edges
        are produced. Mirrors the Pearson backend's safeguard against
        runaway pair density. Raises ``MemoryError`` if exceeded.
    use_parallel
        Route the O(f^2) pair scan through the numba prange kernel
        ``_pairwise_su_edges`` when ``n_features >= parallel_min_features``.
        Default ``True``. The serial path is kept as the fallback (and the
        chosen path at small widths where prange thread-spawn dominates).
    parallel_min_features
        Smallest ``n_features`` at which the parallel kernel is selected.
        ``None`` consults ``pyutilz.system.kernel_tuning_cache`` (key
        ``mlframe.shap_proxied_fs.cluster_su.parallel_min_features``);
        default 50.

    Returns
    -------
    labels : np.ndarray
        ``(n_features,)`` int64. Constant columns and features with no
        above-threshold partner become singleton clusters.
    """
    from mlframe.feature_selection.filters.info_theory import compute_su_from_classes

    if not isinstance(bins, dict):
        raise TypeError(
            f"cluster_correlated_features_su: expected bins dict, got {type(bins).__name__}"
        )
    names, arrays = _resolve_columns(bins, feature_names)
    f = len(names)
    if f == 0:
        return np.empty(0, dtype=np.int64)

    # Pre-compute per-column marginals once. SU(X_i, X_j) calls share the
    # marginal pass for each X_i and X_j independently, so caching across the
    # O(f^2) pair loop drops the marginal work from 2*f*(f-1) to f.
    marginals: list[tuple[np.ndarray, np.ndarray]] = []
    for name, col in zip(names, arrays):
        hint = None
        if nbins_per_feature is not None and name in nbins_per_feature:
            hint = int(nbins_per_feature[name])
        marginals.append(_column_marginal(col, n_bins_hint=hint))

    threshold = float(threshold)
    ei_parts: list[int] = []
    ej_parts: list[int] = []
    total = 0

    # Parallel path: when we have enough features for prange overhead to pay back,
    # build packed buffers once and let the numba kernel run the O(f^2) loop with
    # thread-local joint-count buffers. Falls back to the serial loop on small f
    # or when packing isn't safe (heterogeneous n_samples per column).
    pmin = parallel_min_features if parallel_min_features is not None else _resolve_parallel_min_features()
    use_kernel = bool(use_parallel) and f >= int(pmin)
    if use_kernel:
        packed = _pack_bins_for_kernel(arrays, marginals)
        if packed is not None:
            bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask = packed
            flags = _pairwise_su_edges(
                bins_packed, nbins, freqs_packed, freqs_offsets,
                h_marginals, constant_mask, threshold,
            )
            ei_arr, ej_arr = np.where(flags == 1)
            if ei_arr.size > edge_cap:
                raise MemoryError(
                    f"ShapProxiedFS SU clustering: >{edge_cap} edges at "
                    f"threshold={threshold}. Raise cluster_su_threshold to merge fewer features."
                )
            ei = ei_arr.astype(np.int64, copy=False)
            ej = ej_arr.astype(np.int64, copy=False)
            return _uf_labels(f, ei, ej)

    for i in range(f - 1):
        classes_i, freqs_i = marginals[i]
        if freqs_i.size == 0 or freqs_i.size == 1:
            # constant column -> SU=0 with every partner
            continue
        for j in range(i + 1, f):
            classes_j, freqs_j = marginals[j]
            if freqs_j.size == 0 or freqs_j.size == 1:
                continue
            # ``compute_su_from_classes`` is numba-jitted and reads the
            # int64 class arrays + float64 freq arrays we computed above.
            su = float(compute_su_from_classes(classes_i, freqs_i, classes_j, freqs_j))
            if su >= threshold:
                ei_parts.append(i)
                ej_parts.append(j)
                total += 1
                if total > edge_cap:
                    raise MemoryError(
                        f"ShapProxiedFS SU clustering: >{edge_cap} edges at "
                        f"threshold={threshold}. Raise cluster_su_threshold to merge fewer features."
                    )
    if not ei_parts:
        ei = np.empty(0, dtype=np.int64)
        ej = np.empty(0, dtype=np.int64)
    else:
        ei = np.asarray(ei_parts, dtype=np.int64)
        ej = np.asarray(ej_parts, dtype=np.int64)
    return _uf_labels(f, ei, ej)
