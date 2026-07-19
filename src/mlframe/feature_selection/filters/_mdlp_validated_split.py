"""Holdout-validated MDLP: suppresses splits that overfit to noise.

Classic Fayyad-Irani MDLP (``supervised_binning.mdlp_bin_edges``) accepts every split that
clears the MDL threshold on the SAME data it was searched over -- on a genuinely irrelevant
or high-cardinality-noisy column that threshold is still cleared often enough by chance
(more so at small n / many candidate boundaries), producing spurious bins that fragment
noise instead of signal. This module holds out a validation slice and only accepts a
candidate split when it ALSO reduces conditional entropy on that unseen slice, i.e. the
split's information gain must generalise, not just fit.

Public entry point: ``mdlp_bin_edges_validated``. Mirrors ``mdlp_bin_edges``'s edge contract
(sorted array with ``-inf`` / ``+inf`` sentinels) so it drops into the same downstream
``np.searchsorted`` consumers.
"""
from __future__ import annotations

import math

import numpy as np

from .supervised_binning import _entropy_from_labels, mdlp_bin_edges


def _factorize_labels(y: np.ndarray) -> np.ndarray:
    """Coerce ``y`` to dense int64 class ids, factorizing non-numeric/categorical dtypes."""
    y_arr = np.asarray(y).ravel()
    if y_arr.dtype.kind in ("O", "U", "S") or y_arr.dtype.name in ("category", "string", "object"):
        try:
            import pandas as pd

            y_arr, _ = pd.factorize(y_arr, sort=True)
        except Exception:
            _, y_arr = np.unique(y_arr, return_inverse=True)
    return y_arr.astype(np.int64)


def _mdl_accepts(x: np.ndarray, y: np.ndarray, left_mask: np.ndarray, best_gain: float) -> bool:
    """Fayyad-Irani 1993 MDL acceptance test on an arbitrary (x, y, split-mask)."""
    n = x.size
    if n <= 1 or best_gain <= 0.0:
        return False
    h_full = _entropy_from_labels(y)
    y_left = y[left_mask]
    y_right = y[~left_mask]
    n_classes_full = int(np.unique(y).size)
    n_classes_left = int(np.unique(y_left).size)
    n_classes_right = int(np.unique(y_right).size)
    h_left = _entropy_from_labels(y_left)
    h_right = _entropy_from_labels(y_right)
    delta_arg = (3.0**n_classes_full) - 2.0
    if delta_arg <= 0.0:
        return False
    log2 = math.log(2.0)
    delta = (math.log(delta_arg) / log2) - (n_classes_full * h_full - n_classes_left * h_left - n_classes_right * h_right) / log2
    threshold = (math.log(float(n - 1)) / log2 + delta) / n
    return (best_gain / log2) > threshold


def _best_split(x: np.ndarray, y: np.ndarray, min_split_size: int) -> "tuple[float, float] | None":
    """Scan class-boundary candidates, return ``(split_value, gain)`` for the best one, or ``None``."""
    n = x.size
    if n < 2 * min_split_size:
        return None
    boundary_idx = np.where(y[:-1] != y[1:])[0]
    if boundary_idx.size == 0:
        return None
    candidates = np.unique(0.5 * (x[boundary_idx] + x[boundary_idx + 1]))
    h_full = _entropy_from_labels(y)
    best_gain = -np.inf
    best_split = None
    for split in candidates:
        left_mask = x <= split
        n_l = int(left_mask.sum())
        n_r = n - n_l
        if n_l < min_split_size or n_r < min_split_size:
            continue
        h_l = _entropy_from_labels(y[left_mask])
        h_r = _entropy_from_labels(y[~left_mask])
        h_split = (n_l * h_l + n_r * h_r) / n
        gain = h_full - h_split
        if gain > best_gain:
            best_gain = gain
            best_split = float(split)
    if best_split is None:
        return None
    return best_split, float(best_gain)


def _recurse_validated(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    splits: list,
    depth: int,
    min_split_size: int,
    max_depth: int,
    val_min_split_size: int,
) -> None:
    """Recursively find + accept splits on ``(x_fit, y_fit)`` gated by held-out ``(x_val, y_val)`` confirmation."""
    if depth >= max_depth:
        return
    found = _best_split(x_fit, y_fit, min_split_size)
    if found is None:
        return
    split_value, best_gain = found
    left_mask_fit = x_fit <= split_value
    if not _mdl_accepts(x_fit, y_fit, left_mask_fit, best_gain):
        return

    # Cross-validated confirmation: the same split must ALSO carry positive information
    # gain on the held-out slice. A split fit to noise generalises poorly (gain <= 0 or
    # too few val rows on one side to trust); a genuine signal split reduces entropy on
    # unseen rows too. This is the whole point of the module -- it is the one check
    # classic MDLP omits.
    left_mask_val = x_val <= split_value
    n_l_val = int(left_mask_val.sum())
    n_r_val = x_val.size - n_l_val
    if n_l_val < val_min_split_size or n_r_val < val_min_split_size:
        return
    h_full_val = _entropy_from_labels(y_val)
    h_l_val = _entropy_from_labels(y_val[left_mask_val])
    h_r_val = _entropy_from_labels(y_val[~left_mask_val])
    gain_val = h_full_val - (n_l_val * h_l_val + n_r_val * h_r_val) / x_val.size
    if gain_val <= 0.0:
        return

    splits.append(split_value)
    _recurse_validated(
        x_fit[left_mask_fit], y_fit[left_mask_fit], x_val[left_mask_val], y_val[left_mask_val],
        splits, depth + 1, min_split_size, max_depth, val_min_split_size,
    )
    _recurse_validated(
        x_fit[~left_mask_fit], y_fit[~left_mask_fit], x_val[~left_mask_val], y_val[~left_mask_val],
        splits, depth + 1, min_split_size, max_depth, val_min_split_size,
    )


def mdlp_bin_edges_validated(
    x: np.ndarray,
    y: np.ndarray,
    *,
    val_frac: float = 0.3,
    min_split_size: int = 5,
    val_min_split_size: int = 5,
    max_depth: int = 8,
    random_state: "int | np.random.Generator | None" = None,
) -> np.ndarray:
    """Fayyad-Irani MDLP with a held-out validation slice gating every accepted split.

    Args:
        x: 1-D continuous feature.
        y: 1-D class labels (any dtype; non-numeric/categorical is factorized).
        val_frac: Fraction of rows held out for split validation. Too small and the
            validation gate is itself noisy; too large starves the fit side of the
            recursion. 0.3 is a reasonable default for n in the hundreds-to-thousands range.
        min_split_size: Floor on fit-side samples per child node (mirrors classic MDLP).
        val_min_split_size: Floor on validation-side samples per child node -- a split
            confirmed on <5 held-out rows on one side is not meaningfully validated.
        max_depth: Recursion cap.
        random_state: Seed / Generator for the fit/val split.

    Returns:
        Sorted edge array with ``-inf`` / ``+inf`` sentinels, same contract as ``mdlp_bin_edges``.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = _factorize_labels(y)
    if x_arr.size != y_arr.size:
        raise ValueError(f"len(x)={x_arr.size} != len(y)={y_arr.size}")

    finite_mask = np.isfinite(x_arr)
    if not finite_mask.all():
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
    if x_arr.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)

    rng = np.random.default_rng(random_state)
    n = x_arr.size
    n_val = max(1, round(val_frac * n))
    n_val = min(n_val, n - 1) if n > 1 else 0
    if n_val < val_min_split_size * 2 or (n - n_val) < 2 * min_split_size:
        # Too few rows to carve out a meaningful validation slice -- fall back to classic MDLP
        # rather than silently degrading to zero splits (an empty edge array on a genuinely
        # informative small column would be worse than the un-validated classic result).
        return mdlp_bin_edges(x_arr, y_arr, min_split_size=min_split_size, max_depth=max_depth)

    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    fit_sorter = np.argsort(x_arr[fit_idx])
    x_fit = np.ascontiguousarray(x_arr[fit_idx][fit_sorter])
    y_fit = np.ascontiguousarray(y_arr[fit_idx][fit_sorter])
    val_sorter = np.argsort(x_arr[val_idx])
    x_val = np.ascontiguousarray(x_arr[val_idx][val_sorter])
    y_val = np.ascontiguousarray(y_arr[val_idx][val_sorter])

    splits: list = []
    _recurse_validated(x_fit, y_fit, x_val, y_val, splits, 0, min_split_size, max_depth, val_min_split_size)
    splits.sort()
    edges = np.concatenate([[-np.inf], np.asarray(splits, dtype=np.float64), [np.inf]])
    return edges


def edges_fayyad_irani_validated(
    x: np.ndarray,
    y: np.ndarray,
    *,
    val_frac: float = 0.3,
    max_depth: int = 8,
    min_split_size: int = 5,
    val_min_split_size: int = 5,
    random_state: "int | np.random.Generator | None" = None,
) -> np.ndarray:
    """``_adaptive_nbins.per_feature_edges``-compatible wrapper: returns INNER edges only."""
    full_edges = mdlp_bin_edges_validated(
        np.asarray(x), np.asarray(y),
        val_frac=val_frac, max_depth=max_depth, min_split_size=min_split_size,
        val_min_split_size=val_min_split_size, random_state=random_state,
    )
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    inner = full_edges[1:-1]
    inner = inner[np.isfinite(inner)]
    return np.asarray(inner, dtype=np.float64)
