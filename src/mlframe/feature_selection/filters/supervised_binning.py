"""Supervised discretisation methods (no leak when fit on train, applied to val).

Two backends:

* ``mdlp_bin_edges`` -- Fayyad-Irani 1993 entropy-based recursive splits. No ``n_bins`` hyperparameter; splits are chosen via the MDL
  principle. Simple pure-numpy implementation.
* ``optimal_bin_edges`` -- thin wrapper around ``optbinning.OptimalBinning`` (already a project dep). Production-grade, supports
  monotonic constraints + IV-based feature pre-selection.

Both produce ``bin_edges`` that the downstream ``np.searchsorted`` / ``np.digitize`` chain can consume identically to the unsupervised
paths in ``discretization.py``.

Leak-safe usage pattern::

    edges = mdlp_bin_edges(X_train[:, j], y_train)        # fit on train
    binned_train = np.searchsorted(edges[1:-1], X_train[:, j])
    binned_val   = np.searchsorted(edges[1:-1], X_val[:, j])

The helper does not call ``y`` on the val rows, so passing the same edges to both train and val is leak-safe.
"""
from __future__ import annotations

import numpy as np


def mdlp_bin_edges(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_split_size: int = 5,
    max_depth: int = 8,
) -> np.ndarray:
    """Fayyad-Irani MDLP discretisation. Returns sorted bin edges (includes ``-inf`` / ``+inf`` sentinels).

    Algorithm (recursive):
    1. Sort ``x``, with target ``y`` aligned.
    2. Find candidate split point that maximally reduces conditional entropy ``H(y | x <= split) + H(y | x > split)``.
    3. Apply MDL stopping criterion (Fayyad-Irani 1993): accept split iff ``Gain > log2(N - 1) / N + Delta(A, x, S) / N``.
    4. Recurse on each half.

    Notes
    -----
    Pure numpy. ``n=10000, p=20`` -> ~0.1s per column.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel().astype(np.int64)
    if len(x) != len(y):
        raise ValueError(f"len(x)={len(x)} != len(y)={len(y)}")

    sorter = np.argsort(x)
    x_sorted = x[sorter]
    y_sorted = y[sorter]

    splits: list[float] = []
    _mdlp_recurse(x_sorted, y_sorted, splits, depth=0,
                  min_split_size=min_split_size, max_depth=max_depth)
    splits.sort()
    edges = np.concatenate([[-np.inf], np.asarray(splits, dtype=np.float64), [np.inf]])
    return edges


def _entropy_from_labels(y: np.ndarray) -> float:
    """Shannon entropy of label distribution in nats."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log(p)).sum()


def _mdlp_recurse(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    depth: int,
    min_split_size: int,
    max_depth: int,
) -> None:
    n = len(x)
    if n < 2 * min_split_size or depth >= max_depth:
        return
    if len(np.unique(y)) <= 1:
        return  # already pure

    # Find candidate splits at class-boundary midpoints.
    boundary_idx = np.where(y[:-1] != y[1:])[0]
    if len(boundary_idx) == 0:
        return
    candidates = (x[boundary_idx] + x[boundary_idx + 1]) / 2.0
    candidates = np.unique(candidates)

    h_full = _entropy_from_labels(y)
    best_gain = -np.inf
    best_split = None
    best_left_idx = None

    for split in candidates:
        left_mask = x <= split
        if left_mask.sum() < min_split_size or (~left_mask).sum() < min_split_size:
            continue
        y_left = y[left_mask]
        y_right = y[~left_mask]
        h_left = _entropy_from_labels(y_left)
        h_right = _entropy_from_labels(y_right)
        n_l = len(y_left)
        n_r = len(y_right)
        h_split = (n_l * h_left + n_r * h_right) / n
        gain = h_full - h_split
        if gain > best_gain:
            best_gain = gain
            best_split = split
            best_left_idx = left_mask

    if best_split is None or best_gain <= 0:
        return

    # MDL stopping criterion (Fayyad-Irani 1993).
    n_classes_full = len(np.unique(y))
    n_classes_left = len(np.unique(y[best_left_idx]))
    n_classes_right = len(np.unique(y[~best_left_idx]))
    delta = (
        np.log2(3 ** n_classes_full - 2)
        - (
            n_classes_full * h_full
            - n_classes_left * _entropy_from_labels(y[best_left_idx])
            - n_classes_right * _entropy_from_labels(y[~best_left_idx])
        ) / np.log(2.0)
    )
    threshold = (np.log2(n - 1) + delta) / n
    if best_gain / np.log(2.0) <= threshold:
        return  # MDL says stop

    splits.append(float(best_split))
    _mdlp_recurse(x[best_left_idx], y[best_left_idx], splits, depth + 1,
                  min_split_size, max_depth)
    _mdlp_recurse(x[~best_left_idx], y[~best_left_idx], splits, depth + 1,
                  min_split_size, max_depth)


def optimal_bin_edges(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_n_bins: int = 10,
    monotonic_trend: str = "auto",
) -> np.ndarray:
    """Wrapper around ``optbinning.OptimalBinning``. Returns sorted bin edges with ``-inf / +inf`` sentinels.

    Parameters
    ----------
    monotonic_trend
        ``"auto"`` lets optbinning pick. Pass ``"ascending"`` / ``"descending"`` / ``None`` to override.

    Notes
    -----
    Optbinning is an existing project dep. Pricier than MDLP (~0.5s per column on n=10000) but produces monotonic bins which downstream
    GBM models prefer.

    **Compatibility note**: optbinning's metrics module uses the sklearn API ``check_array(force_all_finite=...)`` which was removed in
    sklearn 1.5+. If you hit ``TypeError: check_array() got an unexpected keyword argument 'force_all_finite'``, either pin
    ``sklearn<1.5`` or switch to ``mdlp_bin_edges``. Reported upstream; the issue is not in our code.
    """
    try:
        from optbinning import OptimalBinning
    except ImportError as e:
        raise ImportError(
            "optimal_bin_edges requires the `optbinning` package. "
            "Install via `pip install optbinning`."
        ) from e

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ob = OptimalBinning(
        name="x",
        dtype="numerical",
        max_n_bins=max_n_bins,
        monotonic_trend=monotonic_trend,
    )
    ob.fit(x, y)
    inner = ob.splits
    edges = np.concatenate([[-np.inf], np.asarray(inner, dtype=np.float64), [np.inf]])
    return edges


def apply_bin_edges(
    x: np.ndarray,
    edges: np.ndarray,
    dtype: object = np.int8,
) -> np.ndarray:
    """Apply pre-fit bin edges to discretise an array.

    Leak-safe: edges are computed once on train and used on both train + val without re-fitting.
    """
    return np.searchsorted(edges[1:-1], x, side="right").astype(dtype)
