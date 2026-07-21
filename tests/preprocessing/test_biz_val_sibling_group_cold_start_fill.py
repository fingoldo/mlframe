"""biz_value test for ``preprocessing.sibling_group_cold_start_fill.sibling_group_cold_start_fill``.

Source: dd_2nd_power-laws-forecasting.md, Future Steps -- "when a ForecastId has entirely missing values,
fall back to the last available value from the previous ForecastId rather than simple mean imputation." When
sibling groups (ordered by some sequence key) share a slowly drifting level, an entirely-missing group's true
value is much closer to its immediate neighbor than to the dataset-wide mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.sibling_group_cold_start_fill import sibling_group_cold_start_fill


def _make_drifting_sibling_groups(n_groups: int, rows_per_group: int, n_cold_start: int, seed: int):
    """Builds seeded synthetic test data; returns ``(df, true_level, mask_cold)``."""
    rng = np.random.default_rng(seed)
    levels = np.cumsum(rng.normal(scale=0.5, size=n_groups)) + 50
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    order_vals = np.repeat(np.arange(n_groups), rows_per_group)
    values = np.repeat(levels, rows_per_group) + rng.normal(scale=0.3, size=n_groups * rows_per_group)

    cold_start_groups = rng.choice(n_groups, size=n_cold_start, replace=False)
    mask_cold = np.isin(group_ids, cold_start_groups)
    values_with_missing = values.copy()
    values_with_missing[mask_cold] = np.nan

    df = pd.DataFrame({"group": group_ids, "order": order_vals, "value": values_with_missing})
    true_level = np.repeat(levels, rows_per_group)
    return df, true_level, mask_cold


def test_biz_val_sibling_fallback_beats_global_mean_for_cold_start_groups():
    """Sibling fallback beats global mean for cold start groups."""
    df, true_level, mask_cold = _make_drifting_sibling_groups(n_groups=100, rows_per_group=5, n_cold_start=15, seed=0)

    filled = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value")
    global_mean_fallback = np.full(len(df), df["value"].dropna().mean())

    mae_sibling = float(np.mean(np.abs(filled[mask_cold] - true_level[mask_cold])))
    mae_global = float(np.mean(np.abs(global_mean_fallback[mask_cold] - true_level[mask_cold])))

    assert (
        mae_sibling < mae_global * 0.5
    ), f"expected sibling-fallback to beat global-mean fallback by >=50% MAE, got sibling={mae_sibling:.4f} global={mae_global:.4f}"


def test_sibling_group_cold_start_fill_leaves_non_missing_groups_untouched():
    """Sibling group cold start fill leaves non missing groups untouched."""
    df = pd.DataFrame({"group": [0, 0, 1, 1, 2, 2], "order": [0, 0, 1, 1, 2, 2], "value": [10.0, 12.0, np.nan, np.nan, 30.0, 32.0]})
    filled = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value")
    # group 0's own last known value (12.0) is used for its own rows.
    assert filled[0] == 12.0
    assert filled[1] == 12.0
    # group 1 is entirely missing -> borrows group 0's last known value (nearest PRECEDING sibling).
    assert filled[2] == 12.0
    assert filled[3] == 12.0


def _make_drifting_sibling_groups_interior_missing(n_groups: int, rows_per_group: int, n_cold_start: int, seed: int):
    """Builds seeded synthetic test data; returns ``(df, true_level, mask_cold)``."""
    rng = np.random.default_rng(seed)
    levels = np.cumsum(rng.normal(scale=0.5, size=n_groups)) + 50
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    order_vals = np.repeat(np.arange(n_groups), rows_per_group)
    values = np.repeat(levels, rows_per_group) + rng.normal(scale=0.3, size=n_groups * rows_per_group)

    # sample cold-start groups strictly from the INTERIOR (never index 0 or n_groups-1) so every missing
    # group is sandwiched between a known preceding AND a known following sibling.
    cold_start_groups = rng.choice(np.arange(1, n_groups - 1), size=n_cold_start, replace=False)
    mask_cold = np.isin(group_ids, cold_start_groups)
    values_with_missing = values.copy()
    values_with_missing[mask_cold] = np.nan

    df = pd.DataFrame({"group": group_ids, "order": order_vals, "value": values_with_missing})
    true_level = np.repeat(levels, rows_per_group)
    return df, true_level, mask_cold


def test_biz_val_sibling_interpolate_beats_ffill_for_sandwiched_cold_start_groups():
    """Sibling interpolate beats ffill for sandwiched cold start groups."""
    df, true_level, mask_cold = _make_drifting_sibling_groups_interior_missing(n_groups=100, rows_per_group=5, n_cold_start=15, seed=0)

    filled_interp = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value", interpolate=True)
    filled_ffill = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value", interpolate=False)

    mae_interp = float(np.mean(np.abs(filled_interp[mask_cold] - true_level[mask_cold])))
    mae_ffill = float(np.mean(np.abs(filled_ffill[mask_cold] - true_level[mask_cold])))

    assert mae_interp < mae_ffill * 0.6, f"expected interpolate=True to beat forward-fill-only by >=40% MAE, got interp={mae_interp:.4f} ffill={mae_ffill:.4f}"


def test_sibling_group_cold_start_fill_interpolate_sandwiched_group_is_midpoint():
    """Sibling group cold start fill interpolate sandwiched group is midpoint."""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1, 2, 2],
            "order": [0, 0, 1, 1, 2, 2],
            "value": [10.0, 10.0, np.nan, np.nan, 30.0, 30.0],
        }
    )
    filled = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value", interpolate=True)
    # group 1 sits exactly between group 0 (10.0) and group 2 (30.0) -> linear interpolation gives 20.0.
    assert filled[2] == 20.0
    assert filled[3] == 20.0


def test_sibling_group_cold_start_fill_interpolate_tail_falls_back_to_ffill():
    """Sibling group cold start fill interpolate tail falls back to ffill."""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1, 2, 2],
            "order": [0, 0, 1, 1, 2, 2],
            "value": [10.0, 10.0, 20.0, 20.0, np.nan, np.nan],
        }
    )
    filled = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value", interpolate=True)
    # group 2 (tail) has no FOLLOWING sibling -> matches interpolate=False behavior: forward-fills from group 1.
    assert filled[4] == 20.0
    assert filled[5] == 20.0


def test_sibling_group_cold_start_fill_first_group_missing_uses_fallback():
    """Sibling group cold start fill first group missing uses fallback."""
    df = pd.DataFrame({"group": [0, 0, 1, 1], "order": [0, 0, 1, 1], "value": [np.nan, np.nan, 20.0, 22.0]})
    filled = sibling_group_cold_start_fill(df, group_col="group", order_col="order", value_col="value", fallback_value=99.0)
    assert filled[0] == 99.0
    assert filled[1] == 99.0
