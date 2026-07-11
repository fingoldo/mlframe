"""``binned_unique_count``: per-entity count of distinct value-bins visited ("cardinality of visited states").

Source: 3rd_amex-default-prediction.md -- "bin feature unique" (binned-feature unique counts per customer,
132 features). mlframe already covers statement-to-statement diffs (``entity_diff_features``) and
trailing-window aggregates (``multi_window_aggregate``) for panel/longitudinal data; this fills the remaining
genuinely distinct piece -- not a diff, not a rolling stat, but a CARDINALITY signal: how many DISTINCT
discretized states has this entity visited across its history (e.g. a customer whose balance has swung across
many different bins is behaviorally different from one who's stayed in a single band, even if their mean/std
look similar).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def binned_unique_count(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    n_bins: int = 10,
    bin_edges: Optional[np.ndarray] = None,
    column_prefix: str = "binned_unique",
    per_entity_bins: bool = False,
) -> pd.DataFrame:
    """Per-entity count of distinct quantile-bins ``value_col`` has visited across the entity's rows.

    Parameters
    ----------
    df
        Long-format panel: one row per ``(entity_col, value_col)`` observation.
    entity_col
        Grouping key.
    value_col
        Continuous column to discretize and count distinct visited bins for.
    n_bins
        Number of bins. For global binning, this is the number of quantile bins (ignored if ``bin_edges``
        is supplied). For ``per_entity_bins``, this is the number of equal-width bins each entity's own
        ``[min, max]`` range is split into.
    bin_edges
        Explicit bin edges (e.g. fitted on train, replayed on test to avoid recomputing quantiles per call);
        defaults to quantile-based edges computed on ``df[value_col]``. Ignored (must be ``None``) when
        ``per_entity_bins=True``, since bin edges are then computed per entity rather than globally.
    column_prefix
        Output column-name prefix.
    per_entity_bins
        Opt-in adaptive mode (default ``False`` preserves prior behavior exactly). When ``True``, bin edges
        are computed independently per entity from that entity's own observed value range (equal-width split
        into ``n_bins`` bins), instead of one global bin-edge set shared across all entities. This matters
        when entities operate on very different absolute scales of ``value_col`` (e.g. one entity's values
        span 1-5, another's span 1000-5000): with a single global bin-edge set (even quantile-based), bin
        resolution follows the POPULATION's density, so entities living in a sparsely-populated region of
        the overall range (typically the minority, e.g. the small-magnitude tail of a heavily right-skewed
        column) can all collapse into the same one or two global bins regardless of how much they
        individually varied -- destroying the "how many distinct states has this entity visited" signal
        exactly for the entities it matters most for. Per-entity bins make the visited-states-count feature
        meaningful for every entity's own dynamic range, independent of the column's global scale/skew.
    Returns
    -------
    pd.DataFrame
        One row per unique entity (first-seen order), columns ``entity_col`` and
        ``{column_prefix}_{value_col}`` (count of distinct bins visited, ``>= 1`` for any entity with at
        least one non-NaN observation).
    """
    values = df[value_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    entities = pd.unique(df[entity_col])
    entity_codes, _ = pd.factorize(df[entity_col], sort=False)

    if per_entity_bins:
        if bin_edges is not None:
            raise ValueError("bin_edges must be None when per_entity_bins=True (edges are computed per entity)")
        vals_for_range = np.where(valid, values, np.nan)
        grp = pd.Series(vals_for_range).groupby(entity_codes)
        entity_min = grp.transform("min").to_numpy()
        entity_max = grp.transform("max").to_numpy()
        width = entity_max - entity_min
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_idx = np.where(width > 0, (values - entity_min) / width * n_bins, 0.0)
        bin_codes = np.clip(np.floor(raw_idx), 0, n_bins - 1).astype(np.int64)
        n_bins_total = n_bins
    else:
        if bin_edges is None:
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            bin_edges = np.unique(np.nanquantile(values, quantiles))
            if len(bin_edges) < 2:
                bin_edges = np.array([np.nanmin(values), np.nanmax(values) + 1e-9])

        bin_codes = np.digitize(values, bin_edges[1:-1], right=False)
        n_bins_total = int(bin_codes.max()) + 1 if valid.any() else 1

    # A per-entity np.unique() Python-level loop was the earlier (rejected) design: n_entities small calls
    # instead of one. Single-pass vectorized equivalent: combine (entity_code, bin_code) into one integer key,
    # np.unique() the key array ONCE (drops duplicate (entity, bin) pairs globally), then count how many
    # surviving distinct-key rows belong to each entity via np.bincount -- O(n) total instead of O(n_entities)
    # separate reductions.
    combined_key = entity_codes[valid].astype(np.int64) * n_bins_total + bin_codes[valid].astype(np.int64)
    unique_keys = np.unique(combined_key)
    unique_entity_codes = unique_keys // n_bins_total
    counts = np.bincount(unique_entity_codes, minlength=len(entities)).astype(np.int64)

    return pd.DataFrame({entity_col: entities, f"{column_prefix}_{value_col}": counts})


__all__ = ["binned_unique_count"]
