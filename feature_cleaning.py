"""Inspects and cleans features to be used in ML.

Works with pandas dataframe as input.
Also works with ndarrays.
"""

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
# Packages
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed  # lint: disable=ungrouped-imports,disable=wrong-import-order

ensure_installed("numpy pandas psutil")

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

from gc import collect
from collections import defaultdict

import re
import psutil
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from pyutilz.pandaslib import classify_column_types
from pyutilz.system import tqdmu, get_own_memory_usage  # lint: disable=ungrouped-imports,disable=wrong-import-order


from .stats import get_expected_unique_random_numbers_qty, get_tukey_fences_multiplier_for_quantile

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from .config import *

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


def _get_nunique(vals: np.ndarray, skip_nan: bool = True, skip_vals: tuple = None) -> int:
    unique_vals = np.unique(vals)
    if skip_nan:
        unique_vals = unique_vals[~np.isnan(unique_vals)]
    if skip_vals:
        for val in skip_vals:
            unique_vals = unique_vals[unique_vals != val]
    return len(unique_vals)


def _update_sub_df_col(
    df: pd.DataFrame, sub_df: pd.DataFrame, col: str, col_unique_values: pd.DataFrame, nunique: int, analyse_mask: np.ndarray = None
) -> tuple:
    if analyse_mask is not None:
        if not sub_df._is_view:
            sub_df[col] = df.loc[analyse_mask, col]
    col_unique_values = sub_df[col].value_counts(dropna=False)
    nunique = len(col_unique_values)
    collect()
    return col_unique_values, nunique


def _clean_cat_and_obj_columns(
    df: pd.DataFrame,
    cat_vars_clean_fcn: object = None,
    obj_vars_clean_fcn: object = None,
    cat_vars_replace: object = None,
    obj_vars_replace: object = None,
    head: pd.DataFrame = None,
    verbose: bool = True,
):
    if head is None:
        head = df.head(1)

    if cat_vars_clean_fcn:

        if verbose:
            logger.info("Cleaning categorical columns...")
        for col in tqdmu(head.select_dtypes("category").columns):
            df[col] = df[col].apply(cat_vars_clean_fcn).astype("category")
        collect()

    if obj_vars_clean_fcn:

        if verbose:
            logger.info("Cleaning object columns...")
        for col in tqdmu(head.select_dtypes("object").columns):
            df[col] = df[col].apply(obj_vars_clean_fcn)
        collect()

    if cat_vars_replace:

        if verbose:
            logger.info("Replacing vals in categorical columns: %s", cat_vars_replace)
        for col in tqdmu(head.select_dtypes("category").columns):
            df[col] = df[col].replace(cat_vars_replace).astype("category")
            collect()

    if obj_vars_replace:

        if verbose:
            logger.info("Replacing vals in object columns: %s", obj_vars_replace)
        for col in tqdmu(head.select_dtypes("object").columns):
            df[col] = df[col].replace(obj_vars_replace)
            collect()


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# CORE
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def is_variable_truly_continuous(
    df: pd.DataFrame = None,
    variable_name: str = "",
    values: np.ndarray = None,
    calculated_quantiles: np.ndarray = None,
    use_quantile: float = 0.1,
    max_scarceness: float = 10.0,
    max_fract_digits: int = 10,
    double_scarcenes_after: tuple = (1_000, 2_000, 5_000, 10_000),
    log_scarceness_divisor: int = 500,
    # min_fract_fill_perecent: float = 0.5,
    min_fract_level_increase_perecent: float = 0.15,
    tukey_fences_multiplier: float = None,
    var_is_datetime: bool = None,
    var_is_numeric: bool = None,
    verbose: bool = False,
):
    """Measures evidence that a variable with numeric type is continuous, given its span and number of unique values.

    mb float32 vars without fractional part, with big span and nunique<1000 must be categorical (nominal or ordinal-remapped)?
    kind of, it's unlikely if even an integer var ranging from -2 to +2 billions on 11M rows will have only 331 unique vals, right?
    not speaking of truly floating, with fractional digits

    what about var such as 10001.132456,10001.3464545,10001.26344432...?
    it can be impactful and very well floating, but will it be distinguished as numeric or not with default max_fract_digits=1?
    Probably not: span will be very narrow (like, 1 or even 0).
    so probably, for vars with fractional digits, need to try iteratively with increasing max_fract_digits=(1,2,.) as long as unique_fracts keeps growing.
    """
    if values is None:
        assert df is not None and variable_name
        values = df[variable_name].values

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # what is the smallest rounding after which variable stops changing?
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if var_is_datetime is None:
        var_is_datetime = is_datetime64_any_dtype(values)
    if var_is_numeric is None:
        var_is_numeric = is_numeric_dtype(values)

    assert var_is_numeric or var_is_datetime

    if var_is_numeric:
        nz_fract_digits = 0
        last_n_unique_fracts = 0
        fract_part, int_part = np.modf(values)

        n_unique_ints = _get_nunique(vals=int_part, skip_vals=(0.0))
        n_unique_fracts = _get_nunique(vals=fract_part, skip_vals=(0.0, 1.0))
        if n_unique_fracts == 0:
            cur_fract_digits = 1
        else:
            for cur_fract_digits in range(1, max_fract_digits):

                n_unique_fracts = _get_nunique(vals=np.round(fract_part, cur_fract_digits), skip_vals=(0.0, 1.0))
                # print(cur_fract_digits, n_unique_fracts, NDIGITS ** (cur_fract_digits))
                if last_n_unique_fracts > 0:
                    if (n_unique_fracts - last_n_unique_fracts) / last_n_unique_fracts < min_fract_level_increase_perecent or n_unique_fracts < 0.3 * (
                        NDIGITS ** (cur_fract_digits)
                    ) ** 0.95:  # <min_fract_fill_perecent * NDIGITS ** (cur_fract_digits)
                        if n_unique_ints > 0 or nz_fract_digits > 0:
                            break
                last_n_unique_fracts = n_unique_fracts
                if n_unique_fracts > 0:
                    nz_fract_digits = cur_fract_digits
            if cur_fract_digits == max_fract_digits - 1:
                if nz_fract_digits == 0:
                    cur_fract_digits = 1
        cur_fract_digits = cur_fract_digits - 1
    elif var_is_datetime:
        full_multiplier = 1
        prev_date_fract = "D"
        for date_fract, multiplier in zip(DATEFRACTS_CODES, DATEFRACTS_MULTIPLIERS):
            if np.all(values.astype(f"datetime64[{date_fract}]") == values):
                break
            else:
                full_multiplier *= multiplier
                prev_date_fract = date_fract

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # get quantiles and count outliers
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    use_quantiles = None

    if calculated_quantiles or use_quantile:

        if calculated_quantiles is None:
            if use_quantile > 0.5:
                use_quantile = 1 - use_quantile
            assert use_quantile > 0 and use_quantile < 1.0

            use_quantiles = (use_quantile, 1 - use_quantile)
            calculated_quantiles = np.nanquantile(values, use_quantiles)
            tukey_fences_multiplier = get_tukey_fences_multiplier_for_quantile(
                quantile=use_quantile,
            )  # !TODO add sigma, dist+kwargs fields
        else:
            assert tukey_fences_multiplier is not None

        values_in_span = values[(values >= calculated_quantiles[0]) & (values <= calculated_quantiles[1])]
        iqr = calculated_quantiles[1] - calculated_quantiles[0]

        n_outliers = (values < calculated_quantiles[0] - tukey_fences_multiplier * iqr).sum() + (
            values > calculated_quantiles[1] + tukey_fences_multiplier * iqr
        ).sum()

    else:
        calculated_quantiles = np.array([np.nanmin(values), np.nanmax(values)])
        values_in_span = values
        n_outliers = 0

    outliers_percent = n_outliers / len(values)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # compute span size, cont_ratio
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    sample_size = len(values_in_span)

    if var_is_numeric:
        span_size = int(n_unique_ints > 1) + (calculated_quantiles[1] - calculated_quantiles[0])

        if cur_fract_digits > 0:
            span_size = span_size * (NDIGITS**cur_fract_digits)
    elif var_is_datetime:
        span_size = (np.timedelta64(1, "D") + (calculated_quantiles[1] - calculated_quantiles[0])).astype("timedelta64[D]") / np.timedelta64(1, "D")
        span_size = span_size * full_multiplier

    if span_size > 0:
        nexpected_unique_values = get_expected_unique_random_numbers_qty(span_size=span_size, sample_size=sample_size)
    else:
        nexpected_unique_values = 0

    real_unique_values = len(np.unique(values_in_span))

    if nexpected_unique_values > 0:
        """
        for max_nunique in double_scarcenes_after:
            if real_unique_values > max_nunique:
                max_scarceness *= 2
        """
        max_scarceness = np.round(max_scarceness * (1.0 + np.max([0.0, np.log(sample_size / log_scarceness_divisor)])), 2)
        if real_unique_values <= 1:
            cont_ratio = 0.0
        else:
            cont_ratio = real_unique_values / (nexpected_unique_values / max_scarceness)
    else:
        cont_ratio = 0.0

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # report if needed
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

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

    return cont_ratio >= 1.0, outliers_percent


def fragment_df_on_ram_usage_increase(df: pd.DataFrame, prev_mem_usage: float, max_increase_percent: float = 0.5) -> tuple:
    new_mem_usage = get_own_memory_usage()
    if prev_mem_usage:
        if new_mem_usage >= prev_mem_usage * (1 + max_increase_percent):
            logger.warning("Trying to deframgent dataframe %s as RAM usage increased from %s to %s", df.columns.name, prev_mem_usage, new_mem_usage)
            collect()
            return df.copy(), prev_mem_usage
        return df, prev_mem_usage
    return df, new_mem_usage


def analyse_and_clean_features(
    df: pd.DataFrame,
    analyse_mask: np.ndarray = None,
    update_data: bool = False,
    cat_vars_clean_fcn: object = None,
    obj_vars_clean_fcn: object = None,
    cat_vars_replace: dict = None,
    obj_vars_replace: dict = None,
    max_rarevals_imbalance: int = 20,
    min_fewlyvalued_rows_per_value: int = 1000,
    clean_numeric_discrete_rarevals: bool = True,
    clean_numeric_continuous_rarevals: bool = True,
    max_cont_col_nuniques_for_rarevals_cleaning: int = 100,
    max_discrete_col_nuniques_for_rarevals_cleaning: int = 0,
    clean_nonnumeric_rarevals: bool = True,
    exclude_columns: tuple = (),
    exclude_mask:str=None,
    default_na_val=np.nan,
    default_float_type=np.float32,
    verbose: bool = True,
    cont_use_quantile: float = 0.1,
    cont_max_scarceness: int = 10,
    cont_max_fract_digits: int = 10,
    cont_max_allowed_outliers_percent: float = 0.1 / 100,
    cont_min_fract_level_increase_perecent: float = 0.15,
) -> dict:
    """Make raw features suitable for machine learning, learn necessary transformations.

    1. Performs arbitrary values replacements in features of a dataframe (you'll know mistyped values after initial inspection via eda module).
    2. Divides numeric and date(time) features into discrete and continuous. (sp. attention to datetime features)
    3. Converts fewly-valued (ie sparse. say, >=100 rows per unique value on avg) object features into categorical, to save space & increase processing speed.
    4. All discrete or fewly-valued (nrows/nunique_vals>=100) features are potentially categorical.
    5. Optionally merges all under-presented categories into one RARE category (usually, a NaN). (Should this be a transformer suitable for a pipeline?)
    6. Replaces nan with some other value when there is only one option except NAN. Like, for numerics, -real_val if real_val<>0, else real_val+1.
    For category, "NOT "+option_name.
    7. Vars having only one unique value, after all, are constant and must be dropped.
    8. Tracks unique values of each feature (or feature ranges, for continuous vars) for future novelty detection.
    Divides all features into constant, fewly- and manyvalued (sparse and dense), possibly categorical and possibly outlying (for future feeding to kBins).
    Merges multiple kinds of Nan values into one? (probably it's possible only in non-numeric columns) !TODO (None,np.nan,pd.NaN)

    Processing of categorical or discrete vars when rarevals cleaning is enabled:
        every category with nvals < nrows/nunique_vals/max_rarevals_imbalance becomes nan (if there is a nan cat already,
        or if there are more than 1 such rare cats)

    possible processing of continuous features:
    1) as is
    2) kbins (converts to categorical, basically)
    3) univariate outliers trimming
    4) embeddings

    possible processing of categorical features:
    1) directly, as "categorical" to estimator with native support (ie, catboost) for internal cat treatment
    2) CE package
    3) converted to numeric by some property, say, textual length of category name (only if not converted TO categorical on the prev step)
    can be also a mapping of intensity for
    4) embeddings

    possible processing of discrete features:
    1) as is - as continuous (kbins only for some minimal ncats at least)
    2) as categorical

    Rareveals cleaning takes into account possible clusters of variables: !TODO
        namely, occupation for each of cluster's vars is taken totalled for the entire cluster, therefore preventing
        dropping of values that are rare in some var but not rare in other vars of the cluster (?).

    When working with imbalanced classification or non-equal error cost (e.g., regression task with RMSE metric, cleaning of features must be done on subset of
    data where target takes most precious values (ie, rare class-balanced). Cause alternatively some variable filled by only 1% could be dropped while
    still being extremely predictive of our target in its rare (and most valuable) class. For this, analyse_mask parameter must be used. Construct the mask in
    a way that a row is chosen according to the value/cost of its target.

    exclude_columns usually contains Target column, to be protected from rarevals merging.

    Multiprocessing support !TODO
    Auto-fragment df if own mem usage keeps growing and system free RAM becomes scarce.
    """
    features_transforms: Dict[str, Dict[Any, Any]] = defaultdict(dict)

    potentially_categorical_features = set()  # all discrete+all fewly-valued (e.g.,<1/1000 population ration)
    potentially_outlying_features = set()

    features_unique_values = {}
    features_ranges = {}

    fewlyvalued_features = set()
    manyvalued_features = set()
    constant_features = set()

    continuous_features = set()
    discrete_features = set()

    head = df.head(1)

    exclude_mask_regexp = None if not exclude_mask else re.compile(exclude_mask)

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
            head=head,
            verbose=verbose,
        )
        collect()

    iterable_columns = df.columns
    if verbose:
        mes=f"Analyzing {len(iterable_columns)} features..."
        logger.info(mes)
        iterable_columns = tqdmu(iterable_columns,desc=mes,leave=True)

    if analyse_mask is None:
        sub_df = df
    else:
        sub_df = df.loc[analyse_mask, :]

    nrows = len(sub_df)

    for col in iterable_columns:  # head.select_dtypes(include=["category", "object", "number", "boolean"])

        col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)

        col_unique_values = sub_df[col].value_counts(dropna=False)
        nunique = len(col_unique_values)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 2. Divides numeric and date(time) features into discrete and continuous.
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
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

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Decides if a col is fewly- or manyvalued.
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        col_is_manyvalued = nrows < min_fewlyvalued_rows_per_value * nunique
        if col_is_manyvalued:
            manyvalued_features.add(col)
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
                    df=df, sub_df=sub_df, analyse_mask=analyse_mask, col=col, col_unique_values=col_unique_values, nunique=nunique
                )
                col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)

        # 4. All discrete or fewly-valued (nrows/nunique_vals>=,say,100) features are potentially categorical.
        if col_is_discrete or col_is_categorical or not col_is_manyvalued:
            if not col_is_categorical:
                potentially_categorical_features.add(col)
            if (col in exclude_columns) or (exclude_mask_regexp and exclude_mask_regexp.search(col)):
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
                to_be_merged = col_unique_values[col_unique_values * nunique * max_rarevals_imbalance < nrows]
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
                    constant_features.add(col)
                    continue  # next col
                else:
                    if nmerged > 0:
                        # (if there is a nan cat already, or if there are more than 1 such rare cats)
                        if nmerged > 1 or nan_vals_already_in_index > 0:
                            if verbose:
                                nrows_merged = to_be_merged.values.sum()
                                logger.info(
                                    "Merging %s values of feature %s into a single %s value due to being too rare (%s/%s [%s percent]): %s",
                                    nmerged,
                                    col,
                                    default_na_val,
                                    format(nrows_merged, THOUSANDS_SEPARATOR + "d"),
                                    format(nrows, THOUSANDS_SEPARATOR + "d"),
                                    round(nrows_merged / nrows * 100, 4),
                                    to_be_merged.index.values.tolist(),
                                )
                            repl_instructions = {}
                            for next_var in to_be_merged.index:
                                if next_var in features_transforms[col]:
                                    logger.warning(
                                        "Key %s of feature %s already in features_transforms with value %s!", next_var, col, features_transforms[col][next_var]
                                    )  # !TODO remove this once checked
                                repl_instructions[next_var] = default_na_val

                            if col_is_numeric and pd.isnull(default_na_val):
                                the_type = default_float_type  # to make sure ints are converted to float when NaNs are added
                            else:
                                the_type = head[col].dtype.name

                            features_transforms[col].update(repl_instructions)
                            if update_data:
                                if col_is_categorical:
                                    df[col] = df[col].astype("object")
                                df[col] = df[col].replace(repl_instructions).astype(the_type)
                                col_unique_values, nunique = _update_sub_df_col(
                                    df=df, sub_df=sub_df, analyse_mask=analyse_mask, col=col, col_unique_values=col_unique_values, nunique=nunique
                                )
                                col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)
                        else:
                            #nmerged=1 and nan_vals_already_in_index=0. No point in merging just one category.
                            pass
            if nunique == 2 and not col_is_datetime:
                # 6. Replaces nan with some other value when there is only one option except NAN. Like, for numerics, -real_val if real_val<>0, else real_val+1.
                # For category, "NOT "+option_name.
                real_val = None
                na_val = True
                for val in col_unique_values.index:
                    if pd.isna(val):
                        na_val = val
                    else:
                        real_val = val
                if (real_val is not None) and (na_val is not True):
                    if type(real_val) == str:
                        repl_value: Any = "not " + real_val
                    else:
                        if col_is_numeric:
                            if float(real_val) == 0.0:
                                repl_value = 1.0
                            else:
                                repl_value = real_val * -1
                        elif col_is_boolean:
                            repl_value = not real_val

                    if verbose:
                        logger.info("feature %s: %s->%s in %s.", col, na_val, repl_value, col_unique_values)

                    repl_instructions = {na_val: repl_value}

                    features_transforms[col].update(repl_instructions)
                    if update_data:
                        if col_is_categorical:
                            df[col] = df[col].astype("object")
                        df[col] = df[col].replace(repl_instructions).astype(head[col].dtype.name)
                        col_unique_values, nunique = _update_sub_df_col(
                            df=df, sub_df=sub_df, analyse_mask=analyse_mask, col=col, col_unique_values=col_unique_values, nunique=nunique
                        )
                        col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=df, col=col)
                else:
                    if real_val is None:
                        if verbose:
                            logger.warning("Non-null value not found in a 2-valued feature %s: %s.", col, col_unique_values)
                        constant_features.add(col)
            if nunique == 1:
                # 7. Vars having only one unique value, after all, are constant and must be dropped.
                constant_features.add(col)

        if col not in constant_features:
            # 8. Tracks unique values of each feature (or feature ranges, for continuous vars) for future novelty detection.
            if (col not in manyvalued_features) or (not col_is_numeric):
                features_unique_values[col] = set(col_unique_values.index.values)
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
                features_ranges[col] = dict(
                    min=col_unique_values.index.min(),
                    max=col_unique_values.index.max(),
                    median=np.nanmedian(col_unique_values.index),
                )

        collect()

    if constant_features:
        logger.info("%s columns are constant: %s.", len(constant_features), constant_features)
        if update_data:
            df.drop(columns=constant_features, inplace=True)
            logger.info("Dropped %s columns.", len(constant_features))

    if verbose:
        logger.info("Analyzing & cleaning finished.")

    return dict(
        dtypes=df.dtypes,
        features_ranges=features_ranges,
        constant_features=constant_features,
        discrete_features=discrete_features,
        continuous_features=continuous_features,
        manyvalued_features=manyvalued_features,
        features_transforms=features_transforms,
        fewlyvalued_features=fewlyvalued_features,
        features_unique_values=features_unique_values,
        potentially_outlying_features=potentially_outlying_features,
        potentially_categorical_features=potentially_categorical_features,
        params=dict(
            update_data=update_data,
            cat_vars_clean_fcn=cat_vars_clean_fcn,
            obj_vars_clean_fcn=obj_vars_clean_fcn,
            cat_vars_replace=cat_vars_replace,
            obj_vars_replace=obj_vars_replace,
            min_fewlyvalued_rows_per_value=min_fewlyvalued_rows_per_value,
            max_rarevals_imbalance=max_rarevals_imbalance,
            clean_numeric_discrete_rarevals=clean_numeric_discrete_rarevals,
            clean_numeric_continuous_rarevals=clean_numeric_continuous_rarevals,
            max_cont_col_nuniques_for_rarevals_cleaning=max_cont_col_nuniques_for_rarevals_cleaning,
            clean_nonnumeric_rarevals=clean_nonnumeric_rarevals,
            default_na_val=default_na_val,
            default_float_type=default_float_type,
            cont_use_quantile=cont_use_quantile,
            cont_max_scarceness=cont_max_scarceness,
            cont_max_fract_digits=cont_max_fract_digits,
            cont_max_allowed_outliers_percent=cont_max_allowed_outliers_percent,
            cont_min_fract_level_increase_perecent=cont_min_fract_level_increase_perecent,
        ),
    )


def fix_doublespaces_and_strip(text: str) -> str:
    try:
        text = text.strip().replace("  ", " ")
    except:
        pass

    return text


def apply_features_cleaning(df: pd.DataFrame, features_cleaning: dict):
    """Apply learned features cleaning to an already cleaned dataframe MUST NOT CHANGE it.

    Basically it's applying learned replacements and dropping columns known as constant.
    Novelty detection? !TODO
    """
    head = df.head(1)
    for col, repl_instructions in features_cleaning["features_transforms"]:
        df[col] = df[col].replace(repl_instructions).astype(head[col].dtype.name)

    constant_features = features_cleaning["constant_features"].copy()
    constant_features = [col for col in constant_features if col in head]
    if constant_features:
        df.drop(columns=constant_features, inplace=True)
