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
from typing import Iterable

import numpy as np

from mlframe.feature_selection._shap_proxy_cluster import _uf_labels

logger = logging.getLogger(__name__)


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
