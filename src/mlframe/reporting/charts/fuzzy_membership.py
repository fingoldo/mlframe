"""Soft-membership curves of a fitted fuzzy partition (PZAD fuzzy-set encoder, interpretability view).

The fuzzy encoder in ``mlframe.feature_engineering.fuzzy_features`` turns one numeric column into ``n_sets`` soft-membership
columns. A triangular (Ruspini) partition is a partition-of-unity: at every point the active memberships sum to 1, so a
reader can see that no probability mass is lost between sets. This chart fits such a partition on a feature and draws each
set's membership function over the feature range, making the overlap of neighbouring sets (the smooth alternative to hard
one-hot binning) directly visible.

The curve evaluation is ``grid * n_partitions`` (tiny); the length-``n`` work is the quantile fit, which is capped by
subsampling the feature to <=200k finite values (RAM-safe: quantile centres are stable under a random subsample, and no
copy of the caller's frame is made -- only a bounded 1-D array of at most 200k values is materialised).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from mlframe.feature_engineering.fuzzy_features import fuzzy_partition_fit, fuzzy_partition_names, fuzzy_partition_transform
from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, LinePanelSpec, PanelSpec

# Cap the length-n fit backing so the quantile sort stays bounded on 100+ GB frames; a random subsample leaves the centre
# quantiles statistically unchanged and never copies the caller's frame (only <=200k floats are pulled).
_FIT_SUBSAMPLE_CAP: int = 200_000


def fuzzy_membership_curves(x, *, n_partitions: int = 5, kind: str = "triangular", grid: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a fuzzy partition on ``x`` and evaluate each set's membership over a ``grid``-point linspace of ``x``'s range.

    Returns ``(grid_x (grid,), memberships (n_sets, grid))`` where ``n_sets`` is the number of fitted sets (equal to
    ``n_partitions`` unless duplicate quantiles collapsed on a low-cardinality column).
    """
    xx = np.asarray(x, dtype=np.float64).ravel()
    finite = xx[np.isfinite(xx)]
    if finite.size == 0:
        raise ValueError("fuzzy_membership_curves: no finite values to fit a partition on.")
    if finite.size > _FIT_SUBSAMPLE_CAP:
        rng = np.random.default_rng(seed)
        finite = finite[rng.integers(0, finite.size, _FIT_SUBSAMPLE_CAP)]  # bounded pull; duplicates are harmless for quantiles
    recipe = fuzzy_partition_fit(finite, n_sets=n_partitions, kind=kind)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        # A constant column has no range to partition. Widening it by an arbitrary 1.0 fabricated a plausible-looking
        # spread of curves over values the feature never takes; the caller is told instead.
        raise ValueError(f"the column is constant at {lo:.6g}, so there is no range over which to place fuzzy sets.")
    grid_x = np.linspace(lo, hi, int(grid))
    memberships = fuzzy_partition_transform(grid_x, recipe)  # (grid, n_sets)
    return grid_x, np.ascontiguousarray(memberships.T)


def fuzzy_membership_panel(x, *, n_partitions: int = 5, kind: str = "triangular", grid: int = 200, seed: int = 0, feature_name: str = "x") -> PanelSpec:
    """Multi-series ``LinePanelSpec`` of the fuzzy-partition membership curves (one series per set).

    ``feature_name`` names the sets in the legend. It used to be hardcoded to "x", so every chart in a report read
    ``x_low`` / ``x_high`` regardless of which feature was partitioned.
    """
    try:
        grid_x, memberships = fuzzy_membership_curves(x, n_partitions=n_partitions, kind=kind, grid=grid, seed=seed)
    except ValueError as exc:
        # Every sibling builder degrades to an annotation on empty input; this one alone propagated the exception
        # and took the caller's whole figure down with it.
        return AnnotationPanelSpec(text=f"Fuzzy partition unavailable: {exc}", title="Fuzzy partition")
    n_sets = memberships.shape[0]
    labels = tuple(fuzzy_partition_names(feature_name, n_sets))
    return LinePanelSpec(
        x=grid_x,
        y=tuple(memberships[j] for j in range(n_sets)),
        series_labels=labels,
        title=f"{kind.capitalize()} fuzzy partition ({n_sets} sets)",
        xlabel="feature value",
        ylabel="membership",
    )


def compose_fuzzy_membership_figure(
    x, *, n_partitions: int = 5, kind: str = "triangular", grid: int = 200, seed: int = 0,
    suptitle: str = "Fuzzy partition membership",
) -> FigureSpec:
    """One-panel ``FigureSpec`` wrapping :func:`fuzzy_membership_panel`."""
    panel = fuzzy_membership_panel(x, n_partitions=n_partitions, kind=kind, grid=grid, seed=seed)
    return FigureSpec(
        suptitle=suptitle,
        panels=((panel,),),
        figsize=(7.0, 4.5),
        caption=(
            "Each curve is one fuzzy set's membership function over the feature's range. A triangular (Ruspini) "
            "partition is a partition of unity: at every x the memberships sum to 1, so no mass is lost between "
            "sets. That is the smooth alternative to hard one-hot binning, and the OVERLAP between neighbouring "
            "curves is exactly the smoothing you buy -- wider overlap means a row near a bin edge contributes to "
            "both sets instead of flipping discontinuously from one to the other."
        ),
    )


__all__ = [
    "fuzzy_membership_curves",
    "fuzzy_membership_panel",
    "compose_fuzzy_membership_figure",
]
