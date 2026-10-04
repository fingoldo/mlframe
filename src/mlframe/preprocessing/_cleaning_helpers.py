"""Helpers carved out of ``cleaning`` to keep that module under its size budget."""

from __future__ import annotations

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import Any

from gc import collect

import re
import numpy as np
import pandas as pd

from pyutilz.pandaslib import classify_column_types
from pyutilz.system import tqdmu  # lint: disable=ungrouped-imports,disable=wrong-import-order


from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from mlframe.config import THOUSANDS_SEPARATOR
from mlframe.utils.log_throttle import log_throttle
from types import SimpleNamespace as _SimpleNamespace

# *****************************************************************************************************************************************************
# INITS
# *****************************************************************************************************************************************************

NDIGITS = 10
DATEFRACTS_CODES = "h m s ms us ns".split(" ")  # list('HTSLUN') for Pandas
DATEFRACTS_MULTIPLIERS = [24, 60, 60, 1000, 1000, 1000]

# *****************************************************************************************************************************************************
# CODE
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------------------------------------------------------------------------


logger = logging.getLogger(__name__)


NDIGITS = 10


DATEFRACTS_CODES = "h m s ms us ns".split(" ")  # list('HTSLUN') for Pandas


DATEFRACTS_MULTIPLIERS = [24, 60, 60, 1000, 1000, 1000]


def _count_unique_fractional_parts(n_unique_fracts, fract_part, max_fract_digits, last_n_unique_fracts, min_fract_level_increase_perecent, n_unique_ints, nz_fract_digits):
    """Count the unique fractional parts at the current digit depth."""
    _sorted_fract: Any = None
    _count_rounded: Any = None
    from .cleaning import _get_count_distinct_rounded_njit  # lazy: looked up on the facade at call time
    if n_unique_fracts == 0:
        cur_fract_digits = 1
    else:
        # Sort the fractional part ONCE; the per-precision distinct-count below rounds inline over this single sorted array
        # (round is monotone, so sort-then-round == round-then-sort), avoiding an O(n log n) re-sort + a rounded-copy alloc per precision.
        # bench-attempt-rejected (2026-07): fusing all precisions into one kernel pass (running per-precision prev/count) was SLOWER at every
        # max_fract_digits {5,8,12,16}: -6% .. -14% (n=100k continuous). The per-precision kernel wins via its simpler inner loop + early break on
        # distinct-saturation; the fused pass recomputes np.rint for all precisions per element even after a precision saturates. See _benchmarks/bench_fused_precision_scan.py.
        if fract_part.dtype.kind == "f":
            _sorted_fract = np.sort(fract_part)
            _count_rounded = _get_count_distinct_rounded_njit()
        else:
            _sorted_fract = None
        cur_fract_digits, nz_fract_digits = _probe_fractional_digits(max_fract_digits, _sorted_fract, _count_rounded, fract_part, last_n_unique_fracts, min_fract_level_increase_perecent, n_unique_ints, nz_fract_digits)
        if cur_fract_digits == max_fract_digits - 1:
            if nz_fract_digits == 0:
                cur_fract_digits = 1
    return cur_fract_digits


def _log_continuity_verdict(verbose, var_is_numeric, cur_fract_digits, var_is_datetime, prev_date_fract, cont_ratio, use_quantiles, calculated_quantiles, sample_size, real_unique_values, nexpected_unique_values, max_scarceness, n_outliers, outliers_percent, variable_name):
    """Log the continuity verdict of the variable."""
    freq: Any = None
    if verbose:
        if var_is_numeric:
            freq = f"max_fract_digits={cur_fract_digits}"
        elif var_is_datetime:
            freq = f"min_freq={prev_date_fract}"
        mes = (
            f"{'Continuous' if cont_ratio >= 1.0 else 'Discrete'}"
            f": for {use_quantiles} quantiles {calculated_quantiles[0]} - {calculated_quantiles[1]} (sample_size={sample_size:{THOUSANDS_SEPARATOR}.0f}), "
            f"{real_unique_values:{THOUSANDS_SEPARATOR}.0f} unique values met, {nexpected_unique_values:{THOUSANDS_SEPARATOR}.0f} expected,"
            f" {freq}, continuity_ratio={cont_ratio:{THOUSANDS_SEPARATOR}.4f} with max_scarceness={max_scarceness:{THOUSANDS_SEPARATOR}}, overall n_outliers="
            f"{n_outliers:{THOUSANDS_SEPARATOR}}({outliers_percent*100:{THOUSANDS_SEPARATOR}.2f}%)."
        )

        if variable_name:
            mes = f"{variable_name}: " + mes
        logger.info(mes)


def _probe_fractional_digits(max_fract_digits, _sorted_fract, _count_rounded, fract_part, last_n_unique_fracts, min_fract_level_increase_perecent, n_unique_ints, nz_fract_digits):
    """Probe the number of fractional digits the values carry."""
    from .cleaning import _get_nunique  # lazy: the original module owns these (monkeypatch-visible)
    for cur_fract_digits in range(1, max_fract_digits):

        if _sorted_fract is not None:
            n_unique_fracts = _count_rounded(_sorted_fract, cur_fract_digits, 0.0, 1.0)
        else:
            n_unique_fracts = _get_nunique(vals=np.asarray(np.round(fract_part, cur_fract_digits)), skip_vals=(0.0, 1.0))
        if last_n_unique_fracts > 0:
            if (n_unique_fracts - last_n_unique_fracts) / last_n_unique_fracts < min_fract_level_increase_perecent or n_unique_fracts < 0.3 * (
                NDIGITS ** (cur_fract_digits)
            ) ** 0.95:  # <min_fract_fill_perecent * NDIGITS ** (cur_fract_digits)
                if n_unique_ints > 0 or nz_fract_digits > 0:
                    break
        last_n_unique_fracts = n_unique_fracts
        if n_unique_fracts > 0:
            nz_fract_digits = cur_fract_digits
    return cur_fract_digits, nz_fract_digits


def _resolve_quantile_cutoffs(calculated_quantiles, use_quantile, values, tukey_fences_multiplier, use_quantiles):
    """Resolve the quantile cutoffs, computing them when absent."""
    if calculated_quantiles is None:
        if use_quantile > 0.5:
            use_quantile = 1 - use_quantile
        # Wave 31 (2026-05-20): assert -> ValueError.
        if not (0 < use_quantile < 1.0):
            raise ValueError(f"use_quantile must be in (0, 1); got {use_quantile!r}.")

        use_quantiles = (use_quantile, 1 - use_quantile)
        calculated_quantiles = np.nanquantile(values, use_quantiles)
        tukey_fences_multiplier = get_tukey_fences_multiplier_for_quantile(
            quantile=use_quantile,
        )  # !TODO add sigma, dist+kwargs fields
    else:
        # Wave 31 (2026-05-20): assert -> ValueError.
        if tukey_fences_multiplier is None:
            raise ValueError("When calculated_quantiles is provided, " "tukey_fences_multiplier MUST also be supplied.")
    return calculated_quantiles, tukey_fences_multiplier, use_quantiles


def _analyse_and_clean__manyvalued_features_set(df, exclude_mask, update_data, cat_vars_clean_fcn, obj_vars_clean_fcn, cat_vars_replace, obj_vars_replace, verbose, analyse_mask, cont_use_quantile, cont_max_scarceness, cont_max_fract_digits, cont_min_fract_level_increase_perecent, cont_max_allowed_outliers_percent, potentially_outlying_features, min_fewlyvalued_rows_per_value, fewlyvalued_features, potentially_categorical_features, exclude_columns, clean_nonnumeric_rarevals, clean_numeric_continuous_rarevals, max_cont_col_nuniques_for_rarevals_cleaning, clean_numeric_discrete_rarevals, max_discrete_col_nuniques_for_rarevals_cleaning, max_rarevals_imbalance, default_na_val, features_transforms, default_float_type, features_dtypes, features_unique_values, features_ranges):
    """Block of _analyse_and_clean__manyvalued_features_set starting at ``manyvalued_features = set()``."""
    iterable_columns, st = _analyse_and_clean__m_step1_st_simplenamespace_long(df, exclude_mask, update_data, cat_vars_clean_fcn, obj_vars_clean_fcn, cat_vars_replace, obj_vars_replace, verbose, analyse_mask)

    _analyse_and_clean__m_step2_col_iterable_columns(iterable_columns, df, st, verbose, cont_use_quantile, cont_max_scarceness, cont_max_fract_digits, cont_min_fract_level_increase_perecent, cont_max_allowed_outliers_percent, potentially_outlying_features, min_fewlyvalued_rows_per_value, fewlyvalued_features, analyse_mask, potentially_categorical_features, exclude_columns, clean_nonnumeric_rarevals, clean_numeric_continuous_rarevals, max_cont_col_nuniques_for_rarevals_cleaning, clean_numeric_discrete_rarevals, max_discrete_col_nuniques_for_rarevals_cleaning, max_rarevals_imbalance, default_na_val, features_transforms, default_float_type, features_dtypes, update_data, features_unique_values, features_ranges)

    _analyse_and_clean__constant_features(st.constant_features, update_data, df)

    if verbose:
        logger.info("Analyzing & cleaning finished.")
    return st.constant_features, st.continuous_features, st.discrete_features, st.manyvalued_features


def _analyse_and_clean__m_step1_st_simplenamespace_long(df, exclude_mask, update_data, cat_vars_clean_fcn, obj_vars_clean_fcn, cat_vars_replace, obj_vars_replace, verbose, analyse_mask):
    """Step 1 of _analyse_and_clean__manyvalued_features_set: lines starting at ``st = _SimpleNamespace() # long-lived locals of this function (see the ``."""
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    from mlframe.preprocessing.cleaning import _clean_cat_and_obj_columns  # lazy: the original module owns these (monkeypatch-visible)
    st.sub_df = None
    st.manyvalued_features = set()
    st.constant_features = set()

    st.continuous_features = set()
    st.discrete_features = set()

    st.head = df.head(1)

    st.exclude_mask_regexp = None if not exclude_mask else re.compile(exclude_mask)

    if update_data:
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 1. Performs arbitrary values replacements in features of a dataframe (you'll know mistyped values after initial inspection via eda module).
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        _clean_cat_and_obj_columns(
            df=df,
            cat_vars_clean_fcn=cat_vars_clean_fcn,
            obj_vars_clean_fcn=obj_vars_clean_fcn,
            cat_vars_replace=cat_vars_replace,
            obj_vars_replace=obj_vars_replace,
            head=st.head,
            verbose=verbose,
        )
        collect()

    iterable_columns = df.columns
    if verbose:
        mes = f"Analyzing {len(iterable_columns)} features..."
        logger.info(mes)
        iterable_columns = tqdmu(iterable_columns, desc=mes, leave=True)

    if analyse_mask is None:
        st.sub_df = df
    else:
        st.sub_df = df.loc[analyse_mask, :]

    st.nrows = len(st.sub_df)
    return iterable_columns, st


def _analyse_and_clean__m_step2_col_iterable_columns(iterable_columns, df, st, verbose, cont_use_quantile, cont_max_scarceness, cont_max_fract_digits, cont_min_fract_level_increase_perecent, cont_max_allowed_outliers_percent, potentially_outlying_features, min_fewlyvalued_rows_per_value, fewlyvalued_features, analyse_mask, potentially_categorical_features, exclude_columns, clean_nonnumeric_rarevals, clean_numeric_continuous_rarevals, max_cont_col_nuniques_for_rarevals_cleaning, clean_numeric_discrete_rarevals, max_discrete_col_nuniques_for_rarevals_cleaning, max_rarevals_imbalance, default_na_val, features_transforms, default_float_type, features_dtypes, update_data, features_unique_values, features_ranges):
    """Step 2 of _analyse_and_clean__manyvalued_features_set: lines starting at ``for col in iterable_columns: # head.select_dtypes(include=["category",``."""
    from mlframe.preprocessing.cleaning import _update_sub_df_col

    for col in iterable_columns:  # head.select_dtypes(include=["category", "object", "number", "boolean"])

        col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)

        col_unique_values = st.sub_df[col].value_counts(dropna=False)
        nunique = len(col_unique_values)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 2. Divides numeric and date(time) features into discrete and continuous.
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        col_is_continuous, col_is_discrete = _analyse_and_clean__block(col_is_numeric, col_is_datetime, st.sub_df, col, verbose, cont_use_quantile, cont_max_scarceness, cont_max_fract_digits, cont_min_fract_level_increase_perecent, st.continuous_features, cont_max_allowed_outliers_percent, potentially_outlying_features, st.discrete_features)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Decides if a col is fewly- or manyvalued.
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        col_is_manyvalued = st.nrows < min_fewlyvalued_rows_per_value * nunique
        col_is_boolean, col_is_categorical, col_is_datetime, col_is_numeric, col_unique_values, nunique = _analyse_and_clean__m_step1_block(col_is_manyvalued, st, col, fewlyvalued_features, col_is_object, df, verbose, analyse_mask, col_unique_values, nunique, col_is_boolean, col_is_categorical, col_is_datetime, col_is_numeric)

        # 4. All discrete or fewly-valued (nrows/nunique_vals>=,say,100) features are potentially categorical.
        if col_is_discrete or col_is_categorical or not col_is_manyvalued:
            if not col_is_categorical:
                potentially_categorical_features.add(col)
            if (col in exclude_columns) or (st.exclude_mask_regexp and st.exclude_mask_regexp.search(col)):
                continue
            # 5. Optionally merges all under-presented categories into one RARE category (usually a NaN). Should this be a transformer suitable for a pipeline?
            if (
                ((clean_nonnumeric_rarevals and not col_is_numeric) and not (col_is_boolean or col_is_datetime))
                or (
                    clean_numeric_continuous_rarevals
                    and col_is_numeric
                    and col_is_continuous
                    and (max_cont_col_nuniques_for_rarevals_cleaning <= 0 or max_cont_col_nuniques_for_rarevals_cleaning >= nunique)
                )
                or (
                    clean_numeric_discrete_rarevals
                    and col_is_numeric
                    and col_is_discrete
                    and (max_discrete_col_nuniques_for_rarevals_cleaning <= 0 or max_discrete_col_nuniques_for_rarevals_cleaning >= nunique)
                )
            ):
                to_be_merged = col_unique_values[col_unique_values * nunique * max_rarevals_imbalance < st.nrows]
                nan_vals_already_in_index = col_unique_values.index.isna().astype(int).sum()
                nmerged = len(to_be_merged)
                if nmerged >= (nunique - nan_vals_already_in_index):
                    if verbose:
                        logger.info(
                            "Feature %s with %s unique vals is too scarcely populated: %s (head), so it will be removed.",
                            col,
                            nunique,
                            col_unique_values.head(),
                        )
                    st.constant_features.add(col)
                    continue  # next col
                else:
                    if nmerged > 0:
                        # (if there is a nan cat already, or if there are more than 1 such rare cats)
                        if nmerged > 1 or nan_vals_already_in_index > 0:
                            if verbose:
                                nrows_merged = to_be_merged.to_numpy().sum()
                                logger.info(
                                    "Merging %s values of feature %s into a single %s value due to being too rare (%s/%s [%s percent]): %s",
                                    nmerged,
                                    col,
                                    default_na_val,
                                    format(nrows_merged, THOUSANDS_SEPARATOR + "d"),
                                    format(st.nrows, THOUSANDS_SEPARATOR + "d"),
                                    round(nrows_merged / st.nrows * 100, 4),
                                    to_be_merged.index.to_list(),
                                )
                            repl_instructions = {}
                            for next_var in to_be_merged.index:
                                if next_var in features_transforms[col]:
                                    # Wave 63 (2026-05-20): collision-detection warning verified
                                    # in production logs; keep as honest WARN, drop the "remove
                                    # once checked" TODO.
                                    log_throttle(
                                        logger,
                                        "cleaning_features_transforms_key_collision",
                                        logging.WARNING,
                                        "Key %s of feature %s already in features_transforms with value %s!",
                                        next_var, col, features_transforms[col][next_var],
                                    )
                                repl_instructions[next_var] = default_na_val

                            if col_is_numeric and pd.isnull(default_na_val):
                                the_type = default_float_type  # to make sure ints are converted to float when NaNs are added
                            else:
                                # The CURRENT dtype, not `head`'s. `head = df.head(1)` was snapshotted before
                                # step 3's `astype("category")` ran, so restoring from it silently converted a
                                # just-categorised column back to object -- undoing the documented memory saving
                                # and leaving a 10M-row, 40-distinct-value column at full string-per-row cost,
                                # with `dtypes=df.dtypes` recording the regression as if intended.
                                the_type = df[col].dtype.name

                            features_transforms[col].update(repl_instructions)
                            features_dtypes[col] = str(the_type if isinstance(the_type, str) else np.dtype(the_type).name)
                            if update_data:
                                if col_is_categorical:
                                    df[col] = df[col].astype("object")
                                # Every rare value maps to the SAME default_na_val, so a single vectorized isin+mask pass replaces them in O(n)
                                # instead of pandas' per-cell dict lookup in .replace() which is O(n*k) for k rare keys (13.5x at n=10M, bit-identical).
                                rare_mask = df[col].isin(list(repl_instructions.keys()))
                                df[col] = df[col].mask(rare_mask, default_na_val).astype(the_type)
                                col_unique_values, nunique = _update_sub_df_col(
                                    df=df, sub_df=st.sub_df, analyse_mask=analyse_mask, col=col, col_unique_values=col_unique_values, nunique=nunique
                                )
                                col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)
                        else:
                            # nmerged=1 and nan_vals_already_in_index=0. No point in merging just one category.
                            pass
            if nunique == 2 and not col_is_datetime:
                # 6. Replaces nan with some other value when there is only one option except NAN. Like, for numerics, -real_val if real_val<>0, else real_val+1.
                # For category, "NOT "+option_name.
                real_val = None
                na_val = True
                na_val, real_val = _analyse_and_clean__category_option_name(col_unique_values, na_val, real_val)
                if (real_val is not None) and (na_val is not True):
                    if isinstance(real_val, str):
                        repl_value: Any = "not " + real_val
                    else:
                        if col_is_numeric:
                            if float(real_val) == 0.0:
                                repl_value = 1.0
                            else:
                                repl_value = real_val * -1
                        elif col_is_boolean:
                            repl_value = not real_val
                        else:
                            # Neither str/numeric/boolean (e.g. decimal.Decimal, pd.Timestamp): negate
                            # if the type supports arithmetic negation (covers Decimal), else fall back
                            # to a distinguishing string sentinel (mirrors the str branch's "not X" naming).
                            try:
                                repl_value = -real_val
                            except TypeError:
                                repl_value = f"not {real_val}"

                    if verbose:
                        logger.info("feature %s: %s->%s in %s.", col, na_val, repl_value, col_unique_values)

                    repl_instructions = {na_val: repl_value}

                    features_transforms[col].update(repl_instructions)
                    features_dtypes.setdefault(col, st.head[col].dtype.name)
                    if update_data:
                        if col_is_categorical:
                            df[col] = df[col].astype("object")
                        df[col] = df[col].replace(repl_instructions).astype(st.head[col].dtype.name)
                        col_unique_values, nunique = _update_sub_df_col(
                            df=df, sub_df=st.sub_df, analyse_mask=analyse_mask, col=col, col_unique_values=col_unique_values, nunique=nunique
                        )
                        col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)
                else:
                    _analyse_and_clean__real_val_none(real_val, verbose, col, col_unique_values, st.constant_features)
            if nunique == 1:
                # 7. Vars having only one unique value, after all, are constant and must be dropped.
                st.constant_features.add(col)

        _analyse_and_clean__vars_having_only_one(col, st.constant_features, st.manyvalued_features, col_is_numeric, col_unique_values, features_unique_values, features_ranges)

        collect()


def _analyse_and_clean__m_step1_block(col_is_manyvalued, st, col, fewlyvalued_features, col_is_object, df, verbose, analyse_mask, col_unique_values, nunique, col_is_boolean, col_is_categorical, col_is_datetime, col_is_numeric):
    """Step 1 of _analyse_and_clean__m_step2_col_iterable_columns: lines starting at ``if col_is_manyvalued:``."""
    from mlframe.preprocessing.cleaning import _update_sub_df_col

    if col_is_manyvalued:
        st.manyvalued_features.add(col)
    else:
        fewlyvalued_features.add(col)
        if col_is_object:
            # ---------------------------------------------------------------------------------------------------------------------------------------------
            # 3. Converts fewly-valued (ie sparse. say, >=100 rows per unique value on avg) object features into categorical, to save space &
            # increase processing speed.
            # ---------------------------------------------------------------------------------------------------------------------------------------------
            df[col] = df[col].astype("category")
            if verbose:
                logger.info("Feature  %s converted to category type.", col)
            col_unique_values, nunique = _update_sub_df_col(
                df=df, sub_df=st.sub_df, analyse_mask=analyse_mask, col=col, col_unique_values=col_unique_values, nunique=nunique
            )
            col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)
    return col_is_boolean, col_is_categorical, col_is_datetime, col_is_numeric, col_unique_values, nunique


def _analyse_and_clean__block(col_is_numeric, col_is_datetime, sub_df, col, verbose, cont_use_quantile, cont_max_scarceness, cont_max_fract_digits, cont_min_fract_level_increase_perecent, continuous_features, cont_max_allowed_outliers_percent, potentially_outlying_features, discrete_features):
    """Block of analyse_and_clean_features starting at ``if col_is_numeric or col_is_datetime:``."""
    from .cleaning import is_variable_truly_continuous  # lazy: the original module owns these (monkeypatch-visible)
    if col_is_numeric or col_is_datetime:
        col_is_continuous, outliers_percent = is_variable_truly_continuous(
            sub_df,
            col,
            verbose=verbose,
            use_quantile=cont_use_quantile,
            max_scarceness=cont_max_scarceness,
            max_fract_digits=cont_max_fract_digits,
            min_fract_level_increase_perecent=cont_min_fract_level_increase_perecent,
            var_is_numeric=col_is_numeric,
            var_is_datetime=col_is_datetime,
        )
        col_is_discrete = not col_is_continuous
        if col_is_continuous:
            continuous_features.add(col)
            if outliers_percent > cont_max_allowed_outliers_percent:
                potentially_outlying_features.add(col)
        else:
            discrete_features.add(col)
    else:
        col_is_continuous = None
        col_is_discrete = None
        outliers_percent = 0.0
    return col_is_continuous, col_is_discrete


def _analyse_and_clean__category_option_name(col_unique_values, na_val, real_val):
    """Block of analyse_and_clean_features starting at ``for val in col_unique_values.index:``."""
    for val in col_unique_values.index:
        if pd.isna(val):
            na_val = val
        else:
            real_val = val
    return na_val, real_val


def _analyse_and_clean__real_val_none(real_val, verbose, col, col_unique_values, constant_features):
    """Block of analyse_and_clean_features starting at ``if real_val is None:``."""
    if real_val is None:
        if verbose:
            log_throttle(logger, "cleaning_no_nonnull_in_2valued_feature", logging.WARNING, "Non-null value not found in a 2-valued feature %s: %s.", col, col_unique_values)
        constant_features.add(col)


def _analyse_and_clean__vars_having_only_one(col, constant_features, manyvalued_features, col_is_numeric, col_unique_values, features_unique_values, features_ranges):
    """Block of analyse_and_clean_features starting at ``if col not in constant_features:``."""
    if col not in constant_features:
        # 8. Tracks unique values of each feature (or feature ranges, for continuous vars) for future novelty detection.
        if (col not in manyvalued_features) or (not col_is_numeric):
            features_unique_values[col] = set(col_unique_values.index.to_numpy())
        else:
            """
            features_ranges[col]=df[col].describe().astype(np.float32).to_dict()
            {'count': 11706156.0,
             'mean': 840.458984375,
             'std': 592.664794921875,
             'min': 0.0,
             '25%': 339.0,
             '50%': 741.0,
             '75%': 1260.0,
             'max': 2594.0}
            """
            # `col_unique_values` is a `value_counts` Series: its INDEX holds the distinct values and its
            # VALUES hold the counts. min/max off the index are correct (the extremes are the same either
            # way), but the median was taken over the distinct-value SET with the counts ignored entirely --
            # for a monetary or count column concentrated near zero with a long sparse tail, that lands far
            # out in the tail rather than near zero, and every consumer of `features_ranges` (novelty
            # detection, range checks, imputation defaults) read a number labelled "median" that was nowhere
            # near the column's median. Weighting by the counts recovers the real one from the same summary,
            # with no extra pass over the column.
            _vals = np.asarray(col_unique_values.index, dtype=np.float64)
            _cnts = np.asarray(col_unique_values.to_numpy(), dtype=np.float64)
            _ok = np.isfinite(_vals) & (_cnts > 0)
            _median = float("nan")
            if _ok.any():
                _o = np.argsort(_vals[_ok], kind="stable")
                _sv, _sc = _vals[_ok][_o], _cnts[_ok][_o]
                _cum = np.cumsum(_sc)
                _median = float(_sv[int(np.searchsorted(_cum, _cum[-1] / 2.0, side="left"))])
            features_ranges[col] = dict(
                min=col_unique_values.index.min(),
                max=col_unique_values.index.max(),
                median=_median,
            )


def _analyse_and_clean__constant_features(constant_features, update_data, df):
    """Block of analyse_and_clean_features starting at ``if constant_features:``."""
    if constant_features:
        logger.info("%s columns are constant: %s.", len(constant_features), constant_features)
        if update_data:
            df.drop(columns=constant_features, inplace=True)  # noqa: PD002 -- update_data=True is documented as "mutate the caller's frame in place (legacy behaviour)"
            logger.info("Dropped %s columns.", len(constant_features))
