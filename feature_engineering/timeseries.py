# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

# ensure_installed("numpy pandas") #  PyWavelets

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import psutil
from numba import njit
from mlframe.ewma import ewma

from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names

from pyutilz.numpylib import smart_ratios

import pandas as pd, numpy as np

from pyutilz.system import tqdmu
from pyutilz.strings import slugify
from pyutilz.pythonlib import get_human_readable_set_size
from pyutilz.parallel import applyfunc_parallel
from functools import partial

from scipy.signal import welch
from scipy.fftpack import fft
from scipy import signal
import pywt

cCOMPACT_WAVELETS = True

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Inits
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_numaggs_metadata(numaggs_kwds: dict = {}, numaggs_names: list = []):

    if not numaggs_names:
        numaggs_names = list(get_numaggs_names(**numaggs_kwds))

    try:
        q1_idx = numaggs_names.index("q0.25")
        q3_idx = numaggs_names.index("q0.75")
    except ValueError as e:
        q1_idx, q3_idx = None, None

    return numaggs_names, q1_idx, q3_idx


default_countaggs_names = list(get_countaggs_names())
default_numaggs_names, default_q1_idx, default_q3_idx = get_numaggs_metadata()

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------------------------------------------------------------------------


@njit()
def find_next_cumsum_left_index(window_var_values: np.ndarray, amount: float, right_index: int = None, min_samples: int = 1, use_abs: bool = False) -> tuple:
    """Calculating windows having required turnovers."""
    total = 0.0
    if right_index <= 0:
        return 0, total

    if right_index is None:
        right_index = len(window_var_values)
    for i in range(1, right_index):
        if not np.isnan(window_var_values[right_index - i]):
            total += window_var_values[right_index - i]
            if use_abs:
                if np.abs(total) >= amount and i >= min_samples:
                    return right_index - i, total
            else:
                if total >= amount and i >= min_samples:
                    return right_index - i, total

    return 0, total


@njit()
def find_next_cumsum_right_index(window_var_values: np.ndarray, amount: float, left_index: int = None, min_samples: int = 1, use_abs: bool = False) -> tuple:
    """Calculating windows having required turnovers."""
    total = 0.0
    l = len(window_var_values)
    if left_index >= l - 1:
        return l - 1, total

    if left_index is None:
        left_index = 0
    for i in range(1, (l - left_index)):
        if not np.isnan(window_var_values[left_index + i]):
            total += window_var_values[left_index + i]
            if use_abs:
                if np.abs(total) >= amount and i >= min_samples:
                    return left_index + i, total
            else:
                if total >= amount and i >= min_samples:
                    return left_index + i, total

    return l - 1, total


def get_nwindows_expected(windows: dict) -> int:
    """How many windows are expected from the entire set?"""
    r = 0
    for window_var, windows_lengths in windows.items():
        r += len(windows_lengths)
    return r


def get_ts_window_name(window_var: str, window_size: float, window_index_name: str = "") -> str:
    """Give a timeseries window a human readable short name."""

    if window_var == "":  # just index
        dataset_name = str(window_size) + window_index_name
    else:
        dataset_name = window_var + ":" + get_human_readable_set_size(window_size)

    return dataset_name


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Windows and ML features over them
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def create_aggregated_features(
    window_df: pd.DataFrame,
    row_features: list,
    create_features_names: bool,
    features_names: list,
    dataset_name: str,
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # common settings
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    vars_mask_regexp: object = None,
    vars_mask_exclude_regexp: object = None,
    captions_vars_sep: str = "-",
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # numericals
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    differences_features: bool = False,
    ratios_features: bool = False,
    robust_features: bool = False,
    weighting_vars: Sequence = (),
    na_fills: dict = {"": 1e3},
    span_corrections: dict = {"": 1e2},
    ewma_alphas: Sequence = (),
    rolling: Sequence = (),  # method_params can also include engine="numba", engine_kwargs={"parallel": True}
    nonlinear_transforms=[np.cbrt],
    nonnormal_vars: Sequence = (),
    waveletnames="",  # "rbio3.1",
    numaggs_kwds: dict = {},
    splitting_vars: dict = {},
    drawdown_vars: dict = {},
    groupby_vars: dict = {},  # {'ticker':['Volume']}=deals.groupby("ticker").Volume.agg("sum").values / deals.Volume.sum()
    return_n_finite: bool = False,
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # categoricals
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    process_categoricals: bool = False,  # categoricals will be processed as counts data
    counts_processing_mask_regexp: object = None,  # separate variables can be processed as counts as well
    countaggs_kwds: dict = {},
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # subsets
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    subsets: dict = {},
    checked_subsets: list = [],
    subset_token: str = "_",
    nested_subsets: bool = False,
):
    """
    Each suitable variable of a dataframe gets numerical aggregates computed over a number of transformations over the set of its raw values
        1) as is: raw_vals
        2) ratios: div0(raw_vals[1:], raw_vals[:-1], fill=0.0) **ordered feature
        3) wavelets of raw_vals
        4) raw_vals weighted by second var, if the main var is not related to second var (has no second var in its name)
        5) exponentially weighted row_wals, for example with alphas=[0.6]: ewma(prices.Price.values, 0.6) **ordered feature
        5.1) rolling with optional scipy windows: rolling=[(dict(window=2,win_type='parzen'),'mean',dict(sym=False))]; can also include engine="numba", engine_kwargs={"parallel": True}
        6) log, or cubic root, or some other non-linear transform (yeo-johnson) of raw_vals (benefitial for non-normally distributed vars)
        7) robust subset of raw_vals, ie, within 0.1 and 0.9 quantiles
        8) for some variables, especially with many repeated values, or categorical, we can do value_counts(normalize=True or False). Further we can return
            1) Top N highest/lowest values along with their counts (missing are padded with NaNs)
            2) numaggs over counts data
            3) if variable is numeric, numaggs(timeseries_features=True) for values series sorted by counts (timeseries_features=True leaves only aggregates depending on the order of values)
        9*) possibly, ratios of numfeatures over raw values of a smaller window compared to bigger windows
        10*) lags: closest same day of month, week, year. at least for Price! pd.offsets.DateOffset(months=1)
    """
    # assert len(window_df)>1

    if numaggs_kwds:
        numaggs_names, q1_idx, q3_idx = get_numaggs_metadata(numaggs_kwds)
    else:
        numaggs_names, q1_idx, q3_idx = default_numaggs_names, default_q1_idx, default_q3_idx

    if not countaggs_kwds:
        countaggs_names = default_countaggs_names
    else:
        countaggs_names = list(get_countaggs_names(**countaggs_kwds))

    for var in window_df.columns:
        if var in subsets or var in checked_subsets:
            continue
        if (vars_mask_regexp is None or vars_mask_regexp.search(var)) and (vars_mask_exclude_regexp is None or not vars_mask_exclude_regexp.search(var)):

            if var in groupby_vars:
                groupby = window_df.groupby(var, observed=True)  # TODO! investigate: observed=True leads to missing blocks??
                for sum_var in groupby_vars[var]:
                    if sum_var in window_df:
                        total = window_df[sum_var].sum()
                        raw_vals = groupby[sum_var].agg("sum").values
                        if total:
                            raw_vals = raw_vals / total
                        row_features.extend(compute_numaggs(raw_vals, **numaggs_kwds))
                        if create_features_names:
                            features_names.extend((captions_vars_sep.join((dataset_name, sum_var, "grpby", var, feat)) for feat in numaggs_names))

            # is this categorical?
            if window_df[var].dtype.name in (
                "category",
            ):  # we do not list "object", "str" to exclude pure textual columns. They can be added explicilty though.
                if process_categoricals or (counts_processing_mask_regexp and counts_processing_mask_regexp.search(var)):
                    row_features.extend(compute_countaggs(window_df[var], **countaggs_kwds))
                    if create_features_names:
                        features_names.extend((captions_vars_sep.join((dataset_name, var, "vlscnt", feat)) for feat in countaggs_names))
            else:
                if not (window_df[var].dtype.name in ("object", "str")):
                    if "datetime" in window_df[var].dtype.name:
                        raw_vals = window_df[var].diff(1).dt.total_seconds().values
                    else:
                        raw_vals = window_df[var].values

                    # safe values without nans and infs

                    idx = np.isfinite(raw_vals)
                    raw_vals = raw_vals[idx]

                    if return_n_finite:
                        row_features.append(idx.sum())
                        if create_features_names:
                            features_names.append(captions_vars_sep.join((dataset_name, var, "n_finite")))

                    # 1) as is: numaggs of raw_vals

                    if var in drawdown_vars:
                        custom_numaggs_kwds = numaggs_kwds.copy()
                        custom_numaggs_kwds["return_drawdown_stats"] = True
                        simple_numaggs_names = get_numaggs_names(**custom_numaggs_kwds)
                    else:
                        custom_numaggs_kwds = numaggs_kwds
                        simple_numaggs_names = numaggs_names

                    simple_numerical_features = compute_numaggs(raw_vals, **custom_numaggs_kwds)
                    row_features.extend(simple_numerical_features)
                    if create_features_names:
                        features_names.extend((captions_vars_sep.join((dataset_name, var, feat)) for feat in simple_numaggs_names))

                    if splitting_vars and (custom_numaggs_kwds.get("return_unsorted_stats") != False):
                        if splitting_vars and var in splitting_vars:
                            compute_splitting_stats(
                                window_df=window_df,
                                dataset_name=dataset_name,
                                splitting_vars=splitting_vars,
                                var=var,
                                numaggs_names=simple_numaggs_names,
                                numaggs_values=simple_numerical_features,
                                row_features=row_features,
                                features_names=features_names,
                                create_features_names=create_features_names,
                                captions_vars_sep=captions_vars_sep,
                            )

                    if differences_features:
                        differences = np.diff(raw_vals, 1)

                        custom_numaggs_kwds = numaggs_kwds.copy()
                        custom_numaggs_kwds["return_profit_factor"] = True
                        row_features.extend(compute_numaggs(differences, **custom_numaggs_kwds))
                        if create_features_names:
                            custom_numaggs_names = list(get_numaggs_names(**custom_numaggs_kwds))
                            features_names.extend((captions_vars_sep.join((dataset_name, var, "dif", feat)) for feat in custom_numaggs_names))

                    if ratios_features:
                        # 2) ratios: div0(raw_vals[1:], raw_vals[:-1], fill=0.0)
                        # print(var)
                        ratios = smart_ratios(
                            raw_vals[1:],
                            raw_vals[:-1],
                            span_correction=span_corrections.get(var, span_corrections.get("", 1e2)),
                            na_fill=na_fills.get(var, na_fills.get("")),
                        )

                        custom_numaggs_kwds = numaggs_kwds.copy()
                        custom_numaggs_kwds["return_profit_factor"] = True
                        row_features.extend(compute_numaggs(ratios, **custom_numaggs_kwds))
                        if create_features_names:
                            custom_numaggs_names = list(get_numaggs_names(**custom_numaggs_kwds))
                            features_names.extend((captions_vars_sep.join((dataset_name, var, "rat", feat)) for feat in custom_numaggs_names))

                    # 3) wavelets of raw_vals
                    if waveletnames:
                        custom_numaggs_kwds = numaggs_kwds.copy()
                        custom_numaggs_kwds["return_hurst"] = False
                        custom_numaggs_kwds["return_entropy"] = False  # they all are constant anyways ()
                        for waveletname in waveletnames:
                            all_coeffs = []
                            for i, coeffs in enumerate(pywt.wavedec(raw_vals, waveletname)):
                                all_coeffs.append(coeffs)
                            all_coeffs = np.hstack(all_coeffs)
                            row_features.extend(compute_numaggs(all_coeffs, **custom_numaggs_kwds))
                            if create_features_names:
                                custom_numaggs_names = list(get_numaggs_names(**custom_numaggs_kwds))
                                features_names.extend((captions_vars_sep.join((dataset_name, var, waveletname, feat)) for feat in custom_numaggs_names))

                    # 4) raw_vals weighted by second var, if the main var is not related to second var (has no second var in its name)
                    for weighting_var in weighting_vars:
                        if weighting_var not in var and weighting_var in window_df:
                            weighting_values = window_df.loc[idx, weighting_var].values
                            row_features.extend(compute_numaggs((raw_vals / weighting_values.sum()) * weighting_values, **numaggs_kwds))
                            if create_features_names:
                                features_names.extend((captions_vars_sep.join((dataset_name, var, "wgt", weighting_var, feat)) for feat in numaggs_names))

                    # 5) exponentially weighted raw_vals with some alphas, like [0.6, 0.9]:
                    for alpha in ewma_alphas:
                        row_features.extend(compute_numaggs(ewma(raw_vals, alpha), **numaggs_kwds))
                        if create_features_names:
                            features_names.extend((captions_vars_sep.join((dataset_name, var, "ewma", str(alpha), feat)) for feat in numaggs_names))

                    # 5.1) rolling
                    for window, method, method_params in rolling:
                        if "datetime" in window_df[var].dtype.name:
                            # logger.warning(f"Cant't use rolling features for var {var} 'cause of datetime-like dtype.")
                            break
                        vals = getattr(window_df[var].rolling(**window), method)(**method_params).values
                        safe_idx = np.isfinite(vals)
                        vals = vals[safe_idx]
                        row_features.extend(compute_numaggs(vals, **numaggs_kwds))
                        if create_features_names:
                            specs = ";".join(
                                (f"{key}={value}" for key, value in dict(**window, m=method, **method_params).items())
                            )  # comma , refused by lgbm: lightgbm.basic.LightGBMError: Do not support special JSON characters in feature name.
                            specs = specs.replace("win_type", "t").replace("window", "w")
                            features_names.extend((captions_vars_sep.join((dataset_name, var, "rol", specs, feat)) for feat in numaggs_names))

                    # 6) log, or cubic root, or some other non-linear transform (yeo-johnson) of raw_vals
                    if var in nonnormal_vars:
                        for nonlinear_func in nonlinear_transforms:
                            transform_name = nonlinear_func.__name__
                            row_features.extend(compute_numaggs(nonlinear_func(raw_vals), **numaggs_kwds))
                            if create_features_names:
                                features_names.extend((captions_vars_sep.join((dataset_name, var, transform_name, feat)) for feat in numaggs_names))

                    # 7) robust subset of raw_vals, ie, within 0.1 and 0.9 quantiles. Or, better, using Tukey fences to identify outliers.
                    if robust_features:
                        _, q1_idx, q3_idx = get_numaggs_metadata(numaggs_names=simple_numaggs_names)
                        if q1_idx and q3_idx:
                            Q3 = simple_numerical_features[q3_idx]
                            Q1 = simple_numerical_features[q1_idx]
                            IQR = Q3 - Q1
                            Lower_Bound = Q1 - 1.5 * IQR
                            Upper_Bound = Q3 + 1.5 * IQR
                            robust_subset = raw_vals[(raw_vals >= Lower_Bound) & (raw_vals <= Upper_Bound)]
                            # logger.info(f"Var={var}, Q1={Q1}, Q3={Q3}, len(robust_subset)={len(robust_subset)}")

                            if len(robust_subset) == 0:
                                row_features.extend([np.nan] * len(numaggs_names))
                            else:
                                row_features.extend(compute_numaggs(robust_subset, **numaggs_kwds))
                            if create_features_names:
                                features_names.extend((captions_vars_sep.join((dataset_name, var, "rbst", feat)) for feat in numaggs_names))
                # 8) for some variables, especially with many repeated values, or categorical, we can do value_counts(normalize=True or False).
                if counts_processing_mask_regexp and counts_processing_mask_regexp.search(var):
                    row_features.extend(compute_countaggs(window_df[var], **countaggs_kwds))
                    if create_features_names:
                        features_names.extend((captions_vars_sep.join((dataset_name, var, "vlscnt", feat)) for feat in countaggs_names))

    if subsets:
        for subset_var, subset_var_values in subsets.items():
            if subset_var in checked_subsets or subset_var not in window_df:
                continue

            for subset_var_value in subset_var_values:
                idx = window_df[subset_var] == subset_var_value
                subset_df = window_df[idx]
                subset_direct = True
                if len(subset_df) <= 1:
                    # logger.warning(f"Empty subset {subset_var}={subset_var_value} (size={len(subset_df)}). let's use reverse to keep ndims. Block length before was {len(row_features)}")
                    subset_df = window_df[~idx]
                    subset_direct = False

                row_features.append(subset_direct)
                subset_dataset_name = None if dataset_name is None else dataset_name + subset_token + subset_var + "=" + str(subset_var_value)
                if create_features_names:
                    features_names.append(subset_dataset_name + captions_vars_sep + "subset_direct")

                create_aggregated_features(
                    window_df=subset_df,
                    row_features=row_features,
                    create_features_names=create_features_names,
                    features_names=features_names,
                    dataset_name=subset_dataset_name,
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    # common settings
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    vars_mask_regexp=vars_mask_regexp,
                    vars_mask_exclude_regexp=vars_mask_exclude_regexp,
                    captions_vars_sep=captions_vars_sep,
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    # numericals
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    differences_features=differences_features,
                    ratios_features=ratios_features,
                    robust_features=robust_features,
                    weighting_vars=weighting_vars,
                    na_fills=na_fills,
                    span_corrections=span_corrections,
                    ewma_alphas=ewma_alphas,
                    rolling=rolling,
                    nonlinear_transforms=nonlinear_transforms,
                    nonnormal_vars=nonnormal_vars,
                    waveletnames=waveletnames,
                    numaggs_kwds=numaggs_kwds,
                    splitting_vars=splitting_vars,
                    drawdown_vars=drawdown_vars,
                    groupby_vars=groupby_vars,
                    return_n_finite=return_n_finite,
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    # categoricals
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    process_categoricals=process_categoricals,
                    counts_processing_mask_regexp=counts_processing_mask_regexp,
                    countaggs_kwds=countaggs_kwds,
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    # subsets
                    # -----------------------------------------------------------------------------------------------------------------------------------------------------
                    subsets={} if not nested_subsets else subsets,
                    checked_subsets=checked_subsets + [subset_var],
                    subset_token=subset_token,
                    nested_subsets=nested_subsets,
                )

                # if not subset_direct:
                #    logger.warning(f"Block length after subset processing became {len(row_features)}")


def compute_splitting_stats(
    window_df: pd.DataFrame,
    dataset_name: str,
    splitting_vars: dict,
    var: str,
    numaggs_names: list,
    numaggs_values: list,
    row_features: list,
    features_names: list,
    create_features_names: bool,
    captions_vars_sep: str = "-",
) -> None:
    splitting_vals = []
    if create_features_names:
        splitting_ratios_names = []
    subvars = splitting_vars[var]
    for col in ("minr", "maxr"):
        try:
            col_idx = numaggs_names.index(col)
        except Exception as e:
            logger.warning(f"Could not find col={col} in numagg fields {numaggs_names}")
        else:
            index = int(numaggs_values[col_idx] * len(window_df)) - 1
            for subvar in subvars:
                if subvar in window_df:
                    if "datetime" in window_df[subvar].dtype.name:
                        pre_sum = (window_df[subvar].iloc[index] - window_df[subvar].iloc[0]).total_seconds()
                        post_sum = (window_df[subvar].iloc[-1] - window_df[subvar].iloc[index]).total_seconds()
                    else:
                        pre_sum = window_df[subvar].iloc[:index].sum()
                        post_sum = window_df[subvar].iloc[index:].sum()
                    tot = pre_sum + post_sum
                    splitting_vals.append(pre_sum / tot if tot else 0)
                    if create_features_names:
                        splitting_ratios_names.append(captions_vars_sep.join([dataset_name, var, col, subvar, "split"]))

    row_features.extend(splitting_vals)
    if create_features_names:
        features_names.extend(splitting_ratios_names)


def create_windowed_features(
    start_index: int = 0,
    end_index: int = 0,
    df: pd.DataFrame = None,
    past_processing_fcn: object = None,
    future_processing_fcn: object = None,
    features_creation_fcn: object = None,
    targets_creation_fcn: object = None,
    step_size: int = 1,
    nrecords_per_period: int = 1,  # use when fixed number of records belongs to the same period, for example, features contains values for 24 consecutive hours of a day.
    past_windows: dict = {},  # example: {"": [7, 31, 31 * 12], "Load_West": [200e3, 1e6, 5e6]},
    future_windows: dict = {},  # example: {"": [7, 31, 31 * 12], "Load_West": [200e3, 1e6, 5e6]},
    window_index_name: str = "",  # example: D for days, or T for Ticks
    overlapping: bool = False,
    dtype=np.float32,
    verbose: bool = False,
):
    """Creates, for a given range of indexes/time moments, start_index to end_index over step_size, a number of past and future windows.
    Past window(s) will be used to compute ML features, future window(s) - to compute ML target(s).
    Windows can be overlapping or not. They can be of a fixed size in records or in units of any numerical variable from the dataframe.
    Resulting features & targets are merged into a pandas dataframe.

    How to support a scenario where we want as features the RATIOS of aggregated features over 2 consecutive windows?
    What if we want as a target the ratio of some parameter in the future window by its average over 3 consecutive past windows?
    For that, we need the ability to combine calculated windows features in an arbitrary way.

    Another advanced scenario:
        we need to take 1 big past and 1 big future window for the entire market, compute features (past and future, they are different) for it.
        then from that 2 big windows we need to partition by instrument code and create the same features for each instrument.
        Finally, as ML features we need the ratio of instrument past features to market past features.
        As the ML targets we need one future feature of the instrument divided by the same past feature of the same instrument.

    There are 2 modes. In legacy one, features/captions over different windows are simply merged together in the downstream aggregates calculation funct.
    In the advanced mode, ML features and targets have to be constructed manually from the dicts of individually computed windows variables.


    """
    logger.info("got ranges from %s to %s", start_index, end_index)

    targets = []
    features = []

    targets_names = []
    features_names = []

    past_vars_names = []
    future_vars_names = []

    if not end_index:
        end_index = len(df)

    past_nwindows_expected = get_nwindows_expected(future_windows)
    future_nwindows_expected = get_nwindows_expected(future_windows)

    for index in tqdmu(range(start_index, end_index, step_size), desc="dataset range", leave=False):

        # one step means only one features line will be added to features and targets lists.

        row_targets = []
        row_features = []
        base_point = index * nrecords_per_period

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Compute future features (targets constituents) first: they are usually less expensive
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        future_windows_features = create_and_process_windows(
            df=df,
            base_point=base_point,
            windows=future_windows,
            apply_fcn=future_processing_fcn,
            window_index_name=window_index_name,
            overlapping=overlapping,
            forward_direction=True,
            window_features_names=future_vars_names,
            window_features=None if targets_creation_fcn else row_targets,
            create_features_names=targets,
            verbose=verbose,
        )

        if (len(row_targets) < future_nwindows_expected) and len(future_windows_features) == 0:
            # targets could not be fully created, skipping
            continue

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Compute past features
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        past_windows_features = create_and_process_windows(
            df=df,
            base_point=base_point,
            windows=past_windows,
            apply_fcn=past_processing_fcn,
            window_index_name=window_index_name,
            overlapping=overlapping,
            forward_direction=False,
            window_features_names=past_vars_names,
            window_features=None if features_creation_fcn else row_features,
            create_features_names=targets,
            verbose=verbose,
        )

        if row_features or past_windows_features:

            features.append(row_features)

            if targets_creation_fcn:
                targets.append(
                    targets_creation_fcn(past_windows=past_windows_features if features_creation_fcn else row_features, future_windows=future_windows_features)
                )
            else:
                targets.append(row_targets)
        else:
            if not targets:
                features_names = []
                targets_names = []

    if features:

        if targets_creation_fcn:
            targets_names = [targets_creation_fcn.__name__]

        X = pd.DataFrame(data=features, columns=features_names if features_creation_fcn else past_vars_names, dtype=dtype)
        Y = pd.DataFrame(data=targets, columns=targets_names if targets_creation_fcn else future_vars_names, dtype=dtype)

        if Y.shape[1] == 1:
            Y = Y.iloc[:, 0]

        logger.info("computed the features")

        return X, Y
    else:
        return None, None


def create_and_process_windows(
    df: pd.DataFrame,
    base_point: int,
    apply_fcn: object,
    windows: dict,
    window_features_names: list,
    window_features: list,
    targets: list,
    forward_direction: bool = True,
    window_index_name: str = "",  # example: D for days, or T for Ticks, R for rows
    nrecords_per_period: int = 1,  # use when fixed number of records belongs to the same period, for example, data contains values for 24 consecutive hours of a day.
    overlapping: bool = False,
    verbose: bool = False,
):
    """Build all required windows (target or features), from the current base_point of the dataframe.
    Apply function to compute features to each window, return dict with window scpecification as a key, list of features as a value.
    """
    res = {}
    for window_var, windows_lengths in windows.items():  # (p_window_var := tqdmu(windows.items(), desc="window var", leave=False)):
        # p_window_var.set_description("window var=" + window_var)
        # windows_lengths must be sorted

        if forward_direction:
            windows_l = base_point
        else:
            windows_r = base_point

        if window_var:
            if window_var not in df:
                break
            if forward_direction:
                window_var_values = df[window_var].values[base_point:]
            else:
                window_var_values = df[window_var].values[:base_point]

        for window_order, window_size in enumerate(windows_lengths):  # (p_window_size := tqdmu(enumerate(windows_lengths), desc=f"window size", leave=False)):
            # p_window_size.set_description("window size=" + str(window_size))
            if window_var == "":  # just index
                dataset_name = str(window_size) + window_index_name
                if forward_direction:
                    windows_r = min(windows_l + window_size * nrecords_per_period, len(df))
                else:
                    windows_l = max(windows_r - window_size * nrecords_per_period, 0)
            else:
                dataset_name = window_var + ":" + get_human_readable_set_size(window_size)
                # binary search along that var's cumsum until it reaches the required size
                if forward_direction:
                    windows_r, accumulated_amount = find_next_cumsum_right_index(window_var_values=window_var_values, amount=window_size, left_index=windows_l)
                else:
                    windows_l, accumulated_amount = find_next_cumsum_left_index(window_var_values=window_var_values, amount=window_size, right_index=windows_r)
                if accumulated_amount > 0 and accumulated_amount * 2 < window_size:
                    logger.warning("Insufficient data for window %s of size %s: real size=%s", window_var, window_size, accumulated_amount)
                    continue

            # Window established. Now need to create features for it.

            window_df = df.iloc[windows_l:windows_r]
            if len(window_df):

                if verbose:
                    logger.info(
                        "%s, acc.size %s, l=%s, r=%s (%s to %s)",
                        dataset_name + " " + ("future" if forward_direction else "past"),
                        accumulated_amount if window_var else "",
                        windows_l,
                        windows_r,
                        window_df.index[0],
                        window_df.index[-1],
                    )
                if window_features is not None:
                    apply_fcn(df=window_df, row_features=window_features, targets=targets, features_names=window_features_names, dataset_name=dataset_name)
                else:
                    temp_window_features = []
                    apply_fcn(df=window_df, row_features=temp_window_features, targets=targets, features_names=window_features_names, dataset_name=dataset_name)
                    res[dataset_name] = temp_window_features

                # print(
                #    window_var, window_order, window_size, time_from_ts(window_df.MOMENT.values[0]), time_from_ts(window_df.MOMENT.values[-1])
                # )  # ,"->",time_from_ts(future.MOMENT.values[0]),time_from_ts(future.MOMENT.values[-1]),)

                if not overlapping:
                    if forward_direction:
                        windows_l = windows_r
                    else:
                        windows_r = windows_l
    return res


def create_ts_features_parallel(
    start_index: int, end_index: int = None, ts_func: object = None, n_cores: int = None, logical: bool = False, n_chunks: int = None, **kwargs
):
    """
    Divides a span (end_index-start_index) into n_chunks non-overlapping consecutive chunks.
    Submits FE pipeline execution in parallel, joins results.
    """
    nrecords_per_period = kwargs.get("nrecords_per_period", 1)

    if not end_index:
        end_index = len(kwargs.get("df", []))
        if not end_index:
            return None, None
        end_index = end_index // nrecords_per_period

    if not n_chunks or n_chunks < 0:
        n_chunks = min(int(psutil.cpu_count(logical=True) * 1.5), (end_index - start_index))

    args = []
    step = (end_index - start_index) // n_chunks
    if step < 1:
        return None, None

    l = start_index
    for i in range(n_chunks):
        r = min(l + step, end_index)
        if i == n_chunks - 1:
            r = end_index

        args.append((l, r))

        l = r

    logger.info("starting applyfunc_parallel using args %s", args)
    res = applyfunc_parallel(args, partial(ts_func, **kwargs), return_dataframe=False, logical=logical, n_cores=n_cores)
    X_parts, Y_parts = [], []
    for X_part, Y_part in res:
        X_parts.append(X_part)
        Y_parts.append(Y_part)

    del res
    return pd.concat(X_parts, ignore_index=True), pd.concat(Y_parts, ignore_index=True)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Detecting optimal lags for FE
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def compute_corr(dependent_vals: np.ndarray, independent_vals: np.ndarray, deciding_func: object, absolutize: bool = True):
    if deciding_func is np.corrcoef:
        corr = deciding_func(dependent_vals, independent_vals)[0][1]
    else:
        corr = deciding_func(dependent_vals.reshape(-1, 1), independent_vals)[0]

    if absolutize:
        corr = np.abs(corr)

    return corr


def general_acf(
    Y: np.ndarray, X: np.ndarray = None, windows: dict = {}, deciding_func: object = np.corrcoef, lag_len: int = 30, min_samples=500, absolutize: bool = True
):
    """Advanced ACF(nonlinear, +over variables with non-fixed offsets).
    windows={var: {from,to,nsteps]}, ie {"Load":{"from":40_000,"to":1e6,"nsteps":100}}
    """
    res = {}

    if lag_len:
        acfs_vals = [1.0]
        acfs_index = [1.0]
        for i in tqdmu(range(lag_len), desc="Fixed offsets"):
            if len(Y) - i >= min_samples:
                dependent_vals = Y[i + 1 :]
                independent_vals = Y[: -(i + 1)]

                corr = compute_corr(dependent_vals=dependent_vals, independent_vals=independent_vals, deciding_func=deciding_func, absolutize=absolutize)

                acfs_vals.append(corr)
                acfs_index.append(i + 1)

        res["fixed_offsets"] = pd.Series(data=acfs_vals, name="fixed_offsets", index=acfs_index)

    if windows:
        for window_var, windows_params in tqdmu(windows.items(), desc="Flexible windows"):  # windows_lengths must be sorted
            window_var_values = X[window_var].values
            acfs_vals = [1.0]
            acfs_index = [0.0]
            window_sizes = np.linspace(
                start=windows_params.get("from", max(1, window_var_values.min())),
                stop=windows_params.get("to", window_var_values.sum()),
                num=windows_params.get("nsteps", 100),
            )

            for window_size in tqdmu(window_sizes, desc=window_var):
                dependent_vals = []
                independent_vals = []
                windows_r = len(window_var_values) - 1

                while True:
                    # binary search along that var's cumsum until it reaches required size
                    windows_l, accumulated_amount = find_next_cumsum_left_index(window_var_values=window_var_values, amount=window_size, right_index=windows_r)
                    if accumulated_amount * 2 < window_size:  # or (windows_r - windows_l) < min_samples
                        break
                    else:
                        dependent_vals.append(Y[windows_r])
                        independent_vals.append(Y[windows_l])

                    windows_r = windows_l
                    if windows_l <= 0:
                        break
                if dependent_vals:
                    corr = compute_corr(
                        dependent_vals=np.array(dependent_vals), independent_vals=np.array(independent_vals), deciding_func=deciding_func, absolutize=absolutize
                    )

                    acfs_vals.append(corr)
                    acfs_index.append(window_size)

            res[window_var] = pd.Series(data=acfs_vals, name=window_var, index=acfs_index)
    return res
