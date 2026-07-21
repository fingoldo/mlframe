"""Cross-locale shrinkage extension of holiday-name target encoding.

Source: av_top3_xtreme_mlhack_datafest2017.md -- Mark Landry's Spanish holiday flags distinguished
national/local/observance TIERS; :func:`mlframe.feature_engineering.holiday_calendar_features.holiday_calendar_features`
(``include_nearest_name=True``) already emits a per-row holiday NAME so a leakage-safe target-encoder can
learn per-holiday magnitude. That existing recipe (``ordered_target_encode`` on a per-country composite key,
see ``tests/feature_engineering/test_biz_val_holiday_calendar_features_nearest_holiday_name.py``) is a genuine
cold-start trap: a country/locale with few or zero PAST observations of a given holiday (a new market, a
newly-onboarded region, a holiday recently added to the calendar) falls back to the flat global prior --
even when OTHER countries already carry rich history for a holiday of the SAME name (e.g. "Christmas Day" is
observed in dozens of ``holidays``-package locales). This module adds an opt-in blend toward a cross-country
per-NAME prior, controlled by ``cross_locale_shrinkage``, so sparse-history countries borrow strength from
richer ones instead of collapsing to the grand mean.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def holiday_name_target_encode_cross_locale(
    names: np.ndarray,
    countries: np.ndarray,
    y: np.ndarray,
    order: Optional[np.ndarray] = None,
    smoothing: float = 1.0,
    prior: Optional[float] = None,
    cross_locale_shrinkage: Optional[float] = None,
    causal_prior: bool = False,
) -> np.ndarray:
    """Ordered (causal, leakage-safe) target encoding of a per-(country, holiday-name) key.

    Same causal-expanding-mean contract as ``mlframe.training.feature_handling.ordered_target_encoder.ordered_target_encode``,
    specialized to a two-level (country, holiday-name) hierarchy. Default behavior (``cross_locale_shrinkage=None``)
    is SAME-COUNTRY-ONLY encoding: it is bit-identical to calling ``ordered_target_encode`` on the composite
    ``country + "\\x00" + name`` key directly -- i.e. exactly what the pre-existing single-locale recipe already
    does, just packaged as a two-array-input helper.

    Parameters
    ----------
    names
        ``(n,)`` holiday-name categories (e.g. ``holiday_calendar_features(..., include_nearest_name=True)``'s
        ``{prefix}_nearest_holiday_name`` column).
    countries
        ``(n,)`` country/locale codes aligned to ``names``.
    y
        ``(n,)`` target aligned to ``names``/``countries``.
    order
        Optional ``(n,)`` ordering key; rows are processed in ascending order of this array (same contract as
        ``ordered_target_encode``). Defaults to input row order.
    smoothing
        CatBoost-style "a" parameter pulling BOTH the local (country, name) mean and the cross-country
        per-name mean toward ``prior`` when their own running counts are small.
    prior
        Global prior for zero-count rows; defaults to the overall target mean (all countries, all names).
    cross_locale_shrinkage
        ``None`` (default) disables cross-locale blending entirely -- same-country-only encoding. When set to
        a positive float ``k``, each row's encoding is a count-weighted blend of its own (country, name)
        running mean and the cross-country running mean for that holiday NAME alone: ``(local_count *
        local_mean + k * global_name_mean) / (local_count + k)``. A country with zero prior local history of
        a holiday gets the cross-country name-level prior in full (weight ``k / (0 + k) == 1``); a country
        with abundant local history is barely pulled off its own mean (``local_count >> k``). This is the
        standard empirical-Bayes/James-Stein shrinkage-by-count recipe, applied causally.
    causal_prior
        Default ``False`` reproduces the original behaviour: ``global_prior`` is a single scalar computed
        over the FULL ``y`` array, including rows that occur causally AFTER the row being encoded (matches
        CatBoost's own published "ordered target statistics" design, which also uses a single
        global-average prior). Set ``True`` for a strictly zero-leakage prior: each row uses the EXPANDING
        mean of ``y`` over only the rows strictly before it in ``order`` (row 0 falls back to the explicit
        ``prior`` if given, else ``0.0``). See ``ordered_target_encode``'s identical parameter.

    Returns
    -------
    np.ndarray
        ``(n,)`` encoded values, same order as the input arrays.
    """
    names_arr = np.asarray(names)
    countries_arr = np.asarray(countries)
    y_arr = np.asarray(y, dtype=np.float64)
    n = names_arr.shape[0]
    if order is None:
        sort_idx = np.arange(n)
    else:
        sort_idx = np.argsort(np.asarray(order), kind="mergesort")

    sorted_names = names_arr[sort_idx]
    sorted_countries = countries_arr[sort_idx]
    sorted_y = y_arr[sort_idx]

    if causal_prior:
        # Strictly zero-leakage prior: see ordered_target_encode's causal_prior docstring.
        global_running_sum = np.cumsum(sorted_y) - sorted_y
        global_running_count = np.arange(n, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            global_prior_sorted = global_running_sum / global_running_count
        global_prior_sorted[0] = float(prior) if prior is not None else 0.0
    else:
        global_prior_sorted = np.full(n, float(np.mean(y_arr)) if prior is None else float(prior), dtype=np.float64)
    # NUL-joined composite key: holiday names/country codes are free-text strings that may themselves
    # contain any other separator, but never a NUL byte, so this can't accidentally collide two distinct
    # (country, name) pairs onto the same key.
    local_key = np.char.add(np.char.add(sorted_countries.astype(str), "\x00"), sorted_names.astype(str))

    df = pd.DataFrame({"local": local_key, "name": sorted_names, "y": sorted_y})

    local_grp = df.groupby("local", sort=False)["y"]
    local_running_sum = local_grp.cumsum() - sorted_y
    local_running_count = local_grp.cumcount()
    local_mean = (local_running_sum + smoothing * global_prior_sorted) / (local_running_count + smoothing)

    if cross_locale_shrinkage is None:
        encoded_sorted = local_mean
    else:
        k = float(cross_locale_shrinkage)
        if k <= 0.0:
            raise ValueError(f"cross_locale_shrinkage must be positive when not None, got {k}")
        name_grp = df.groupby("name", sort=False)["y"]
        name_running_sum = name_grp.cumsum() - sorted_y
        name_running_count = name_grp.cumcount()
        global_name_mean = (name_running_sum + smoothing * global_prior_sorted) / (name_running_count + smoothing)

        local_count_f = local_running_count.astype(np.float64)
        encoded_sorted = (local_count_f * local_mean + k * global_name_mean) / (local_count_f + k)

    encoded = np.empty(n, dtype=np.float64)
    encoded[sort_idx] = encoded_sorted.to_numpy()
    return encoded


__all__ = ["holiday_name_target_encode_cross_locale"]
