"""Conditional quantile-rank feature (mrmr_audit_2026-07-20 fe_expansion.md "Extend conditional-
dispersion to the full conditional quantile").

Extends the existing conditional-dispersion family (z-within-group in ``_grouped_agg_fe.py`` /
``_composite_group_agg_fe.py``, and the z-score / |z| features in
``_extra_fe_families_dispersion.py``) from CONDITIONAL MEAN/STD ONLY to the full CONDITIONAL
QUANTILE: for a numeric column ``x_i`` and a binned conditioning column ``x_j``, compute
``q(row) = empirical_rank(x_i within bin(x_j))`` -- the row's percentile position WITHIN its
conditioning bin, not its z-score.

Why this catches a shape the catalog misses: on a heavy-tailed / skewed conditional distribution
(e.g. ``x_i | bin(x_j)`` is log-normal, common for financial/count data), a z-score
``(x-mu)/sigma`` badly misrepresents "how extreme" a row is -- the mean/std pair is not a
sufficient statistic for a skewed shape, so two rows with identical z-scores can sit at very
different TRUE percentiles. A target that depends on "is this row in the top-5% of its conditional
peer group" (e.g. fraud/outlier detection conditioned on a merchant category) is exactly the shape
z-score under-resolves and quantile-rank resolves directly.

The per-bin quantile edges are fit on TRAIN rows only (leak-safe, mirroring the existing K-fold-
fit-then-apply discipline used elsewhere in this codebase), then applied to ALL rows (train+test)
via ``searchsorted`` at apply time -- a row whose bin was never seen at fit time gets NaN rather
than a spurious extrapolated rank.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["conditional_quantile_rank_fe"]


def conditional_quantile_rank_fe(
    x_i: np.ndarray,
    x_j_bins: np.ndarray,
    *,
    x_i_fit: Optional[np.ndarray] = None,
    x_j_bins_fit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The empirical within-bin percentile rank of ``x_i``, conditioned on the discrete groups in
    ``x_j_bins``.

    Parameters
    ----------
    x_i : (n,) array
        The numeric column to rank.
    x_j_bins : (n,) array
        Discrete conditioning bin/category id per row (e.g. an already-quantile-binned column, or
        a categorical group id).
    x_i_fit, x_j_bins_fit : (n_fit,) arrays, optional
        The rows to fit the per-bin sorted-value reference on. ``None`` (default) fits on
        ``x_i``/``x_j_bins`` themselves. Pass the TRAIN rows explicitly for a leak-safe fit-once/
        apply-to-all-rows contract.

    Returns
    -------
    (n,) float64 array in ``[0, 1]``: the row's percentile position within its conditioning bin's
    fitted value distribution (``searchsorted(sorted_bin_values, x_i[row]) / len(sorted_bin_values)``).
    NaN for a row whose ``x_j_bins`` value was never seen at fit time, or for non-finite input.
    """
    x_i = np.asarray(x_i, dtype=np.float64).ravel()
    x_j_bins = np.asarray(x_j_bins).ravel()
    n = x_i.shape[0]
    if x_j_bins.shape[0] != n:
        raise ValueError(f"conditional_quantile_rank_fe: x_i has {n} rows but x_j_bins has {x_j_bins.shape[0]}")

    fit_x = np.asarray(x_i_fit, dtype=np.float64).ravel() if x_i_fit is not None else x_i
    fit_bins = np.asarray(x_j_bins_fit).ravel() if x_j_bins_fit is not None else x_j_bins
    if fit_bins.shape[0] != fit_x.shape[0]:
        raise ValueError("conditional_quantile_rank_fe: x_i_fit and x_j_bins_fit must have the same length")

    out = np.full(n, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(x_i)

    for b in np.unique(fit_bins):
        fit_vals = fit_x[fit_bins == b]
        fit_vals = fit_vals[np.isfinite(fit_vals)]
        if fit_vals.size == 0:
            continue
        sorted_vals = np.sort(fit_vals)
        row_mask = finite_mask & (x_j_bins == b)
        if not row_mask.any():
            continue
        ranks = np.searchsorted(sorted_vals, x_i[row_mask], side="right")
        out[row_mask] = ranks / sorted_vals.size

    return np.clip(out, 0.0, 1.0)
