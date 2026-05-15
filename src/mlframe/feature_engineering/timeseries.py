"""Time-series feature engineering for ML."""

from __future__ import annotations


__all__ = [
    "get_numaggs_metadata",
    "find_next_cumsum_left_index",
    "find_next_cumsum_right_index",
    "get_nwindows_expected",
    "get_ts_window_name",
    "create_aggregated_features",
    "compute_splitting_stats",
    "create_windowed_features",
    "create_and_process_windows",
    "create_ts_features_parallel",
    "compute_corr",
    "general_acf",
]

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import pywt
from numba import njit

from mlframe.core.ewma import ewma_numba
from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
from pyutilz.numpylib import smart_ratios
from pyutilz.parallel import applyfunc_parallel
from pyutilz.pythonlib import get_human_readable_set_size
from pyutilz.system import tqdmu

logger = logging.getLogger(__name__)

# Default fill / correction constants. Hoisted to module scope so they're not rebuilt per call
# and so callers can introspect them.
_DEFAULT_NA_FILL: float = 1e3
_DEFAULT_SPAN_CORRECTION: float = 1e2
_MIN_WINDOW_FILL_RATIO: float = 0.5  # window is skipped if cumsum < 50% of the requested size

# Per-call dict literals (e.g. `{"": 1e3}`) used to rebuild on every entry; cached at module scope.
# Callers must not mutate these - functions read them as fall-throughs only.
_DEFAULT_NA_FILLS: Dict[str, float] = {"": _DEFAULT_NA_FILL}
_DEFAULT_SPAN_CORRECTIONS: Dict[str, float] = {"": _DEFAULT_SPAN_CORRECTION}


def get_numaggs_metadata(
    numaggs_kwds: Optional[dict] = None,
    numaggs_names: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[int], Optional[int]]:
    """Return ``(names, q1_idx, q3_idx)`` where the two indices locate ``q0.25`` and ``q0.75`` in the names list (or ``None`` if absent)."""
    if numaggs_kwds is None:
        numaggs_kwds = {}
    if numaggs_names is None:
        numaggs_names = []
    if not numaggs_names:
        numaggs_names = list(get_numaggs_names(**numaggs_kwds))

    try:
        q1_idx: Optional[int] = numaggs_names.index("q0.25")
        q3_idx: Optional[int] = numaggs_names.index("q0.75")
    except ValueError:
        q1_idx, q3_idx = None, None

    return numaggs_names, q1_idx, q3_idx


default_countaggs_names = list(get_countaggs_names())
default_numaggs_names, default_q1_idx, default_q3_idx = get_numaggs_metadata()


@njit
def _find_next_cumsum_left_index_njit(
    window_var_values: np.ndarray,
    amount: float,
    right_index: int,
    min_samples: int,
    use_abs: bool,
) -> Tuple[int, float]:  # pragma: no cover
    total = 0.0
    if right_index < 0:
        right_index = len(window_var_values)
    if right_index <= 0:
        return 0, total
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


def find_next_cumsum_left_index(
    window_var_values: np.ndarray,
    amount: float,
    right_index: Optional[int] = None,
    min_samples: int = 1,
    use_abs: bool = False,
) -> Tuple[int, float]:
    """Walk left from ``right_index`` accumulating ``window_var_values`` until ``cumsum >= amount``.

    ``right_index=None`` (or ``< 0``) means "use ``len(window_var_values)``" - the njit kernel
    cannot accept ``None`` directly so this Python wrapper translates the sentinel.
    """
    right_int = -1 if right_index is None else int(right_index)
    return _find_next_cumsum_left_index_njit(window_var_values, amount, right_int, min_samples, use_abs)


@njit
def _find_next_cumsum_right_index_njit(
    window_var_values: np.ndarray,
    amount: float,
    left_index: int,
    min_samples: int,
    use_abs: bool,
) -> Tuple[int, float]:  # pragma: no cover
    total = 0.0
    length = len(window_var_values)
    if left_index >= length - 1:
        return length - 1, total
    for i in range(1, (length - left_index)):
        if not np.isnan(window_var_values[left_index + i]):
            total += window_var_values[left_index + i]
            if use_abs:
                if np.abs(total) >= amount and i >= min_samples:
                    return left_index + i, total
            else:
                if total >= amount and i >= min_samples:
                    return left_index + i, total
    return length - 1, total


def find_next_cumsum_right_index(
    window_var_values: np.ndarray,
    amount: float,
    left_index: Optional[int] = None,
    min_samples: int = 1,
    use_abs: bool = False,
) -> Tuple[int, float]:
    """Walk right from ``left_index`` accumulating ``window_var_values`` until ``cumsum >= amount``.

    ``left_index=None`` defaults to ``0`` (start from the array head).
    """
    left_int = 0 if left_index is None else int(left_index)
    return _find_next_cumsum_right_index_njit(window_var_values, amount, left_int, min_samples, use_abs)


def get_nwindows_expected(windows: Dict[str, Sequence]) -> int:
    """Total number of windows produced by ``create_and_process_windows`` for the given spec."""
    return sum(len(v) for v in windows.values())


def get_ts_window_name(window_var: str, window_size: float, window_index_name: str = "") -> str:
    """Human-readable short name for a time-series window."""
    if window_var == "":
        return str(window_size) + window_index_name
    return window_var + ":" + get_human_readable_set_size(window_size)


# ----------------------------------------------------------------------------------------------------------------------------
# Per-transform helpers for create_aggregated_features.
#
# Extracted to keep the main function's per-column loop readable. Each helper is responsible
# for ONE transform (raw numaggs, differences, ratios, wavelets, weighted, ewma, rolling,
# non-linear, robust, categorical counts, groupby). The helpers extend row_features and
# features_names in place exactly as the inlined code used to, with byte-identical output
# order. The snapshot regression suite (tests/feature_engineering/test_coverage_fill.py::
# TestCreateAggregatedFeaturesSnapshot) catches any divergence.
# ----------------------------------------------------------------------------------------------------------------------------


def _emit_groupby_block(
    window_df, var, groupby_vars, dataset_name, captions_vars_sep,
    numaggs_kwds, numaggs_names, row_features, features_names, create_features_names,
):
    """For each sum_var in groupby_vars[var], group window_df by var and emit numaggs over
    the per-group sums normalised by the global total. observed=True so absent categories don't
    materialise as empty groups (which produce blank features).
    """
    groupby = window_df.groupby(var, observed=True)
    for sum_var in groupby_vars[var]:
        if sum_var in window_df:
            total = window_df[sum_var].sum()
            raw_vals = groupby[sum_var].sum().values
            if total:
                raw_vals = raw_vals / total
            row_features.extend(compute_numaggs(raw_vals, **numaggs_kwds))
            if create_features_names:
                features_names.extend(
                    captions_vars_sep.join((dataset_name, sum_var, "grpby", var, feat))
                    for feat in numaggs_names
                )


def _emit_categorical_counts(
    series, var, dataset_name, captions_vars_sep, countaggs_kwds, countaggs_names,
    row_features, features_names, create_features_names, *,
    process_categoricals, counts_processing_mask_regexp,
):
    """Treat a category/object dtype column as a count distribution if either process_categoricals
    is true or the column matches counts_processing_mask_regexp."""
    if process_categoricals or (counts_processing_mask_regexp and counts_processing_mask_regexp.search(var)):
        row_features.extend(compute_countaggs(series, **countaggs_kwds))
        if create_features_names:
            features_names.extend(
                captions_vars_sep.join((dataset_name, var, "vlscnt", feat))
                for feat in countaggs_names
            )


def _emit_raw_numaggs(
    var, raw_vals, drawdown_vars, lintrend_approx_vars,
    numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Emit the raw-values numagg block. Returns ``(simple_numerical_features, simple_numaggs_names,
    custom_numaggs_kwds)`` so the caller can pass them on to splitting_stats and robust_features.
    drawdown_vars / lintrend_approx_vars trigger extra numagg fields for THIS column only."""
    if var in drawdown_vars or var in lintrend_approx_vars:
        custom_numaggs_kwds = numaggs_kwds.copy()
        if var in drawdown_vars:
            custom_numaggs_kwds["return_drawdown_stats"] = True
        if var in lintrend_approx_vars:
            custom_numaggs_kwds["return_lintrend_approx_stats"] = True
        simple_numaggs_names = get_numaggs_names(**custom_numaggs_kwds)
    else:
        custom_numaggs_kwds = numaggs_kwds
        simple_numaggs_names = numaggs_names

    simple_numerical_features = compute_numaggs(raw_vals, **custom_numaggs_kwds)
    row_features.extend(simple_numerical_features)
    if create_features_names:
        features_names.extend(
            captions_vars_sep.join((dataset_name, var, feat)) for feat in simple_numaggs_names
        )
    return simple_numerical_features, simple_numaggs_names, custom_numaggs_kwds


def _emit_differences(
    var, raw_vals, numaggs_kwds, dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    differences = np.diff(raw_vals, 1)
    custom_numaggs_kwds_diffs = numaggs_kwds.copy()
    custom_numaggs_kwds_diffs["return_profit_factor"] = True
    row_features.extend(compute_numaggs(differences, **custom_numaggs_kwds_diffs))
    if create_features_names:
        features_names.extend(
            captions_vars_sep.join((dataset_name, var, "dif", feat))
            for feat in get_numaggs_names(**custom_numaggs_kwds_diffs)
        )


def _emit_ratios(
    var, raw_vals, numaggs_kwds, span_corrections, na_fills,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    ratios = smart_ratios(
        raw_vals[1:],
        raw_vals[:-1],
        span_correction=span_corrections.get(var, span_corrections.get("", _DEFAULT_SPAN_CORRECTION)),
        na_fill=na_fills.get(var, na_fills.get("")),
    )
    custom_numaggs_kwds_ratios = numaggs_kwds.copy()
    custom_numaggs_kwds_ratios["return_profit_factor"] = True
    row_features.extend(compute_numaggs(ratios, **custom_numaggs_kwds_ratios))
    if create_features_names:
        features_names.extend(
            captions_vars_sep.join((dataset_name, var, "rat", feat))
            for feat in get_numaggs_names(**custom_numaggs_kwds_ratios)
        )


def _emit_wavelets(
    var, raw_vals, waveletnames, numaggs_kwds, wavelets_correction_numaggs_kwds,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    custom_numaggs_kwds_wave = numaggs_kwds.copy()
    custom_numaggs_kwds_wave.update(wavelets_correction_numaggs_kwds)
    for waveletname in waveletnames:
        all_coeffs = np.hstack(list(pywt.wavedec(raw_vals, waveletname)))
        row_features.extend(compute_numaggs(all_coeffs, **custom_numaggs_kwds_wave))
        if create_features_names:
            features_names.extend(
                captions_vars_sep.join((dataset_name, var, waveletname, feat))
                for feat in get_numaggs_names(**custom_numaggs_kwds_wave)
            )


def _emit_weighted(
    var, raw_vals, idx, weighting_vars, window_df,
    numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Weight raw_vals by each weighting_var present in window_df. Zero-sum / shape-mismatch
    pads with 0.0 to preserve the per-row feature-width contract (P0 fix: previously divided by
    zero, producing inf/NaN columns)."""
    for weighting_var in weighting_vars:
        if weighting_var not in var and weighting_var in window_df:
            weighting_values = window_df.loc[idx, weighting_var].values
            w_sum = weighting_values.sum()
            if w_sum == 0 or len(weighting_values) != len(raw_vals):
                row_features.extend([0.0] * len(numaggs_names))
            else:
                weighted = (raw_vals / w_sum) * weighting_values
                row_features.extend(compute_numaggs(weighted, **numaggs_kwds))
            if create_features_names:
                features_names.extend(
                    captions_vars_sep.join((dataset_name, var, "wgt", weighting_var, feat))
                    for feat in numaggs_names
                )


def _emit_ewma(
    var, raw_vals, ewma_alphas, numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    for alpha in ewma_alphas:
        if len(raw_vals) > 0:
            row_features.extend(compute_numaggs(ewma_numba(raw_vals.astype(np.float32), alpha), **numaggs_kwds))
        else:
            row_features.extend([0.0] * len(numaggs_names))
        if create_features_names:
            features_names.extend(
                captions_vars_sep.join((dataset_name, var, "ewma", str(alpha), feat))
                for feat in numaggs_names
            )


def _emit_rolling(
    var, series, dtype_name, rolling, numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Rolling windows do not apply to datetime columns; we `break` (not `continue`) on first
    datetime so the entire rolling block is skipped for this column."""
    for window, method, method_params in rolling:
        if "datetime" in dtype_name:
            break
        vals = getattr(series.rolling(**window), method)(**method_params).values
        safe_idx = np.isfinite(vals)
        vals = vals[safe_idx]
        row_features.extend(compute_numaggs(vals, **numaggs_kwds))
        if create_features_names:
            # lightgbm forbids commas in feature names ("Do not support special JSON characters
            # in feature name") - use ";" as the per-kv separator.
            specs = ";".join(
                f"{key}={value}" for key, value in dict(**window, m=method, **method_params).items()
            )
            specs = specs.replace("win_type", "t").replace("window", "w")
            features_names.extend(
                captions_vars_sep.join((dataset_name, var, "rol", specs, feat))
                for feat in numaggs_names
            )


def _emit_nonlinear(
    var, raw_vals, nonlinear_transforms, numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    for nonlinear_func in nonlinear_transforms:
        transform_name = nonlinear_func.__name__
        row_features.extend(compute_numaggs(nonlinear_func(raw_vals), **numaggs_kwds))
        if create_features_names:
            features_names.extend(
                captions_vars_sep.join((dataset_name, var, transform_name, feat))
                for feat in numaggs_names
            )


def _emit_robust(
    var, raw_vals, simple_numerical_features, simple_numaggs_names,
    numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Robust quantile-trimmed subset (Tukey-fence). q1_idx_local can be 0 - the old
    `if q1_idx and q3_idx:` truthy check silently dropped the q0.25-at-index-0 case. Use
    explicit `is not None` here (P0 fix)."""
    _, q1_idx_local, q3_idx_local = get_numaggs_metadata(numaggs_names=simple_numaggs_names)
    if q1_idx_local is not None and q3_idx_local is not None:
        q3 = simple_numerical_features[q3_idx_local]
        q1 = simple_numerical_features[q1_idx_local]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        robust_subset = raw_vals[(raw_vals >= lower_bound) & (raw_vals <= upper_bound)]

        if len(robust_subset) == 0:
            row_features.extend([np.nan] * len(numaggs_names))
        else:
            row_features.extend(compute_numaggs(robust_subset, **numaggs_kwds))
        if create_features_names:
            features_names.extend(
                captions_vars_sep.join((dataset_name, var, "rbst", feat))
                for feat in numaggs_names
            )


def _emit_counts_regexp(
    series, var, countaggs_kwds, countaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Per-variable counts processing triggered by counts_processing_mask_regexp (independent of
    dtype). Used for integer count-like columns that aren't pandas categoricals."""
    row_features.extend(compute_countaggs(series, **countaggs_kwds))
    if create_features_names:
        features_names.extend(
            captions_vars_sep.join((dataset_name, var, "vlscnt", feat))
            for feat in countaggs_names
        )


def create_aggregated_features(
    window_df: pd.DataFrame,
    row_features: list,
    create_features_names: bool,
    features_names: list,
    dataset_name: Optional[str],
    vars_mask_regexp: Optional[Pattern] = None,
    vars_mask_exclude_regexp: Optional[Pattern] = None,
    captions_vars_sep: str = "-",
    differences_features: bool = False,
    ratios_features: bool = False,
    robust_features: bool = False,
    weighting_vars: Sequence[str] = (),
    na_fills: Optional[Dict[str, float]] = None,
    span_corrections: Optional[Dict[str, float]] = None,
    ewma_alphas: Sequence[float] = (),
    rolling: Sequence = (),
    nonlinear_transforms: Optional[Sequence[Callable]] = None,
    nonnormal_vars: Sequence[str] = (),
    waveletnames: Sequence[str] = "",
    wavelets_correction_numaggs_kwds: Optional[dict] = None,
    numaggs_kwds: Optional[dict] = None,
    splitting_vars: Optional[Dict[str, Sequence[str]]] = None,
    drawdown_vars: Sequence[str] = (),
    lintrend_approx_vars: Sequence[str] = (),
    groupby_vars: Optional[Dict[str, Sequence[str]]] = None,
    return_n_finite: bool = False,
    process_categoricals: bool = False,
    counts_processing_mask_regexp: Optional[Pattern] = None,
    countaggs_kwds: Optional[dict] = None,
    subsets: Optional[Dict[str, Sequence]] = None,
    checked_subsets: Optional[List[str]] = None,
    subset_token: str = "_",
    nested_subsets: bool = False,
) -> None:
    """Compute numerical/categorical aggregates over every suitable variable in ``window_df``.

    See README for the catalogue of transforms applied: raw, diffs, ratios, wavelets, weighted,
    EWMA, rolling, non-linear, robust quantile-trimmed, and per-value-counts.
    """
    if checked_subsets is None:
        checked_subsets = []
    if countaggs_kwds is None:
        countaggs_kwds = {}
    if groupby_vars is None:
        groupby_vars = {}
    if na_fills is None:
        na_fills = _DEFAULT_NA_FILLS
    if nonlinear_transforms is None:
        nonlinear_transforms = [np.cbrt]
    if numaggs_kwds is None:
        numaggs_kwds = {}
    if wavelets_correction_numaggs_kwds is None:
        wavelets_correction_numaggs_kwds = dict(
            return_hurst=False,
            return_entropy=False,
            return_drawdown_stats=False,
            return_lintrend_approx_stats=False,
        )
    if span_corrections is None:
        span_corrections = _DEFAULT_SPAN_CORRECTIONS
    if splitting_vars is None:
        splitting_vars = {}
    if subsets is None:
        subsets = {}

    if numaggs_kwds:
        numaggs_names, q1_idx, q3_idx = get_numaggs_metadata(numaggs_kwds)  # noqa: F841 -- q1_idx/q3_idx are quantile-bin positions; consumed by downstream per-row selection paths not visible in static analysis (passed via closure to numpy view ops).
    else:
        numaggs_names, q1_idx, q3_idx = default_numaggs_names, default_q1_idx, default_q3_idx  # noqa: F841 -- same as above branch.

    if not countaggs_kwds:
        countaggs_names = default_countaggs_names
    else:
        countaggs_names = list(get_countaggs_names(**countaggs_kwds))

    for var in window_df.columns:
        if var in subsets or var in checked_subsets:
            continue
        if (vars_mask_regexp is not None and not vars_mask_regexp.search(var)) or (
            vars_mask_exclude_regexp is not None and vars_mask_exclude_regexp.search(var)
        ):
            continue

        series = window_df[var]
        dtype_name = series.dtype.name

        if var in groupby_vars:
            _emit_groupby_block(
                window_df, var, groupby_vars, dataset_name, captions_vars_sep,
                numaggs_kwds, numaggs_names, row_features, features_names, create_features_names,
            )

        if dtype_name in ("category", "object"):
            _emit_categorical_counts(
                series, var, dataset_name, captions_vars_sep, countaggs_kwds, countaggs_names,
                row_features, features_names, create_features_names,
                process_categoricals=process_categoricals,
                counts_processing_mask_regexp=counts_processing_mask_regexp,
            )
            continue

        if dtype_name in ("str",):
            continue

        if "datetime" in dtype_name:
            raw_vals = series.diff(1).dt.total_seconds().values
        else:
            raw_vals = series.values

        idx = np.isfinite(raw_vals)
        raw_vals = raw_vals[idx]

        if return_n_finite:
            row_features.append(idx.sum())
            if create_features_names:
                features_names.append(captions_vars_sep.join((dataset_name, var, "n_finite")))

        simple_numerical_features, simple_numaggs_names, custom_numaggs_kwds = _emit_raw_numaggs(
            var, raw_vals, drawdown_vars, lintrend_approx_vars,
            numaggs_kwds, numaggs_names,
            dataset_name, captions_vars_sep,
            row_features, features_names, create_features_names,
        )

        # `.get(..., True)` matches the prior semantics ("if key absent, behave as True") without
        # the `!= False` double-negative that also misfired on None.
        if splitting_vars and var in splitting_vars and custom_numaggs_kwds.get("return_unsorted_stats", True):
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
            _emit_differences(
                var, raw_vals, numaggs_kwds, dataset_name, captions_vars_sep,
                row_features, features_names, create_features_names,
            )

        if ratios_features:
            _emit_ratios(
                var, raw_vals, numaggs_kwds, span_corrections, na_fills,
                dataset_name, captions_vars_sep,
                row_features, features_names, create_features_names,
            )

        if waveletnames:
            _emit_wavelets(
                var, raw_vals, waveletnames, numaggs_kwds, wavelets_correction_numaggs_kwds,
                dataset_name, captions_vars_sep,
                row_features, features_names, create_features_names,
            )

        _emit_weighted(
            var, raw_vals, idx, weighting_vars, window_df,
            numaggs_kwds, numaggs_names,
            dataset_name, captions_vars_sep,
            row_features, features_names, create_features_names,
        )

        _emit_ewma(
            var, raw_vals, ewma_alphas, numaggs_kwds, numaggs_names,
            dataset_name, captions_vars_sep,
            row_features, features_names, create_features_names,
        )

        _emit_rolling(
            var, series, dtype_name, rolling, numaggs_kwds, numaggs_names,
            dataset_name, captions_vars_sep,
            row_features, features_names, create_features_names,
        )

        if var in nonnormal_vars:
            _emit_nonlinear(
                var, raw_vals, nonlinear_transforms, numaggs_kwds, numaggs_names,
                dataset_name, captions_vars_sep,
                row_features, features_names, create_features_names,
            )

        if robust_features:
            _emit_robust(
                var, raw_vals, simple_numerical_features, simple_numaggs_names,
                numaggs_kwds, numaggs_names,
                dataset_name, captions_vars_sep,
                row_features, features_names, create_features_names,
            )

        if counts_processing_mask_regexp and counts_processing_mask_regexp.search(var):
            _emit_counts_regexp(
                series, var, countaggs_kwds, countaggs_names,
                dataset_name, captions_vars_sep,
                row_features, features_names, create_features_names,
            )

    if subsets:
        for subset_var, subset_var_values in subsets.items():
            if subset_var in checked_subsets or subset_var not in window_df:
                continue

            for subset_var_value in subset_var_values:
                idx_mask = window_df[subset_var] == subset_var_value
                subset_df = window_df[idx_mask]
                subset_direct = True
                if len(subset_df) <= 1:
                    subset_df = window_df[~idx_mask]
                    subset_direct = False

                row_features.append(subset_direct)
                if dataset_name is None:
                    subset_dataset_name = None
                else:
                    subset_dataset_name = dataset_name + subset_token + subset_var + "=" + str(subset_var_value)
                if create_features_names and subset_dataset_name is not None:
                    features_names.append(subset_dataset_name + captions_vars_sep + "subset_direct")

                create_aggregated_features(
                    window_df=subset_df,
                    row_features=row_features,
                    create_features_names=create_features_names,
                    features_names=features_names,
                    dataset_name=subset_dataset_name,
                    vars_mask_regexp=vars_mask_regexp,
                    vars_mask_exclude_regexp=vars_mask_exclude_regexp,
                    captions_vars_sep=captions_vars_sep,
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
                    wavelets_correction_numaggs_kwds=wavelets_correction_numaggs_kwds,
                    numaggs_kwds=numaggs_kwds,
                    splitting_vars=splitting_vars,
                    drawdown_vars=drawdown_vars,
                    lintrend_approx_vars=lintrend_approx_vars,
                    groupby_vars=groupby_vars,
                    return_n_finite=return_n_finite,
                    process_categoricals=process_categoricals,
                    counts_processing_mask_regexp=counts_processing_mask_regexp,
                    countaggs_kwds=countaggs_kwds,
                    subsets={} if not nested_subsets else subsets,
                    checked_subsets=checked_subsets + [subset_var],
                    subset_token=subset_token,
                    nested_subsets=nested_subsets,
                )


def compute_splitting_stats(
    window_df: pd.DataFrame,
    dataset_name: str,
    splitting_vars: Dict[str, Sequence[str]],
    var: str,
    numaggs_names: List[str],
    numaggs_values: list,
    row_features: list,
    features_names: list,
    create_features_names: bool,
    captions_vars_sep: str = "-",
) -> None:
    """For each sub-variable, the fraction of its sum that falls before the min/max index of ``var``."""
    splitting_vals: list = []
    splitting_ratios_names: list = [] if create_features_names else None
    subvars = splitting_vars[var]
    for col in ("minr", "maxr"):
        try:
            col_idx = numaggs_names.index(col)
        except ValueError:
            logger.warning("compute_splitting_stats: could not find col=%s in numagg fields", col)
            continue

        # numaggs[col] is a fractional position; map to a row index and CLAMP into [0, len-1].
        # The previous `... - 1` allowed index = -1 which iloc[:-1] interpreted as "drop last",
        # silently corrupting the split ratio.
        raw_index = int(numaggs_values[col_idx] * len(window_df)) - 1
        index = max(0, min(raw_index, len(window_df) - 1))

        for subvar in subvars:
            if subvar in window_df:
                if "datetime" in window_df[subvar].dtype.name:
                    pre_sum = (window_df[subvar].iloc[index] - window_df[subvar].iloc[0]).total_seconds()
                    post_sum = (window_df[subvar].iloc[-1] - window_df[subvar].iloc[index]).total_seconds()
                else:
                    col_sum = window_df[subvar].sum()
                    pre_sum = window_df[subvar].iloc[:index].sum()
                    post_sum = col_sum - pre_sum
                tot = pre_sum + post_sum
                splitting_vals.append(pre_sum / tot if tot else 0)
                if create_features_names:
                    splitting_ratios_names.append(
                        captions_vars_sep.join([dataset_name, var, col, subvar, "split"])
                    )

    row_features.extend(splitting_vals)
    if create_features_names:
        features_names.extend(splitting_ratios_names)


def create_windowed_features(
    df: pd.DataFrame,
    start_index: int = 0,
    end_index: Optional[int] = None,
    past_processing_fcn: Optional[Callable] = None,
    future_processing_fcn: Optional[Callable] = None,
    features_creation_fcn: Optional[Callable] = None,
    targets_creation_fcn: Optional[Callable] = None,
    step_size: int = 1,
    nrecords_per_period: int = 1,
    past_windows: Optional[Dict[str, Sequence]] = None,
    future_windows: Optional[Dict[str, Sequence]] = None,
    window_index_name: str = "",
    overlapping: bool = False,
    dtype=np.float32,
    verbose: bool = False,
):
    """Walk the dataframe with past/future windows, compute features and targets, return ``(X, Y)``.

    ``end_index=None`` means "go to ``len(df)``". An explicit ``end_index=0`` produces an empty
    range and returns ``(None, None)``.
    """
    if df is None:
        raise ValueError("create_windowed_features: df is required")
    if future_windows is None:
        future_windows = {}
    if past_windows is None:
        past_windows = {}

    if end_index is None:
        end_index = len(df)
    logger.info("got ranges from %s to %s", start_index, end_index)

    targets: list = []
    features: list = []
    targets_names: list = []
    features_names: list = []
    past_vars_names: list = []
    future_vars_names: list = []

    past_nwindows_expected = get_nwindows_expected(past_windows)  # noqa: F841 -- !TODO! computed for symmetry with future_nwindows_expected (used below for sanity check); the past-side check was deferred to a follow-up.
    future_nwindows_expected = get_nwindows_expected(future_windows)

    for index in tqdmu(range(start_index, end_index, step_size), desc="dataset range", leave=False):
        row_targets: list = []
        row_features: list = []
        base_point = index * nrecords_per_period

        future_windows_features = create_and_process_windows(
            df=df,
            base_point=base_point,
            windows=future_windows,
            apply_fcn=future_processing_fcn,
            window_index_name=window_index_name,
            overlapping=overlapping,
            forward_direction=True,
            window_features_names=future_vars_names,
            # When a targets_creation_fcn is supplied, targets are derived downstream from the raw
            # future_windows_features dict; otherwise we accumulate per-row targets in row_targets.
            window_features=None if targets_creation_fcn else row_targets,
            create_features_names=(index == start_index),
            verbose=verbose,
        )

        if (len(row_targets) < future_nwindows_expected) and len(future_windows_features) == 0:
            continue

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
            create_features_names=(index == start_index),
            verbose=verbose,
        )

        if row_features or past_windows_features:
            features.append(row_features)

            if targets_creation_fcn:
                targets.append(
                    targets_creation_fcn(
                        past_windows=past_windows_features if features_creation_fcn else row_features,
                        future_windows=future_windows_features,
                    )
                )
            else:
                targets.append(row_targets)
        elif not targets:
            features_names = []
            targets_names = []

    if not features:
        return None, None

    if targets_creation_fcn:
        targets_names = [targets_creation_fcn.__name__]

    X = pd.DataFrame(
        data=features,
        columns=features_names if features_creation_fcn else past_vars_names,
        dtype=dtype,
    )
    Y = pd.DataFrame(
        data=targets,
        columns=targets_names if targets_creation_fcn else future_vars_names,
        dtype=dtype,
    )

    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    logger.info("computed the features")

    return X, Y


def create_and_process_windows(
    df: pd.DataFrame,
    base_point: int,
    apply_fcn: Callable,
    windows: Dict[str, Sequence],
    window_features_names: list,
    window_features: Optional[list],
    targets: Optional[list] = None,
    create_features_names: bool = False,
    forward_direction: bool = True,
    window_index_name: str = "",
    nrecords_per_period: int = 1,
    overlapping: bool = False,
    verbose: bool = False,
) -> Dict[str, list]:
    """Build all required windows from ``base_point``, apply ``apply_fcn`` to each, return per-window features."""
    res: Dict[str, list] = {}
    for window_var, windows_lengths in windows.items():
        if forward_direction:
            windows_l = base_point
            windows_r = base_point  # initialised so the variable always exists in the else-branch
        else:
            windows_l = base_point
            windows_r = base_point

        if window_var:
            # Skip this window_var if its column is missing, but keep iterating the rest of `windows`.
            # Previously this was `break`, which silently dropped every later window_var on the first miss.
            if window_var not in df:
                continue
            if forward_direction:
                window_var_values = df[window_var].values[base_point:]
            else:
                window_var_values = df[window_var].values[:base_point]

        for window_size in windows_lengths:
            accumulated_amount = 0.0
            if window_var == "":
                dataset_name = str(window_size) + window_index_name
                if forward_direction:
                    windows_r = min(windows_l + window_size * nrecords_per_period, len(df))
                else:
                    windows_l = max(windows_r - window_size * nrecords_per_period, 0)
            else:
                dataset_name = window_var + ":" + get_human_readable_set_size(window_size)
                if forward_direction:
                    windows_r, accumulated_amount = find_next_cumsum_right_index(
                        window_var_values=window_var_values, amount=window_size, left_index=windows_l
                    )
                else:
                    windows_l, accumulated_amount = find_next_cumsum_left_index(
                        window_var_values=window_var_values, amount=window_size, right_index=windows_r
                    )
            if window_var and accumulated_amount > 0 and accumulated_amount * 2 < window_size:
                logger.warning(
                    "Insufficient data for window %s of size %s: real size=%s (< %.0f%% threshold)",
                    window_var,
                    window_size,
                    accumulated_amount,
                    _MIN_WINDOW_FILL_RATIO * 100,
                )
                continue

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
                    apply_fcn(
                        df=window_df,
                        row_features=window_features,
                        targets=targets,
                        features_names=window_features_names,
                        dataset_name=dataset_name,
                    )
                else:
                    temp_window_features: list = []
                    apply_fcn(
                        df=window_df,
                        row_features=temp_window_features,
                        targets=targets,
                        features_names=window_features_names,
                        dataset_name=dataset_name,
                    )
                    res[dataset_name] = temp_window_features

                if not overlapping:
                    if forward_direction:
                        windows_l = windows_r
                    else:
                        windows_r = windows_l
    return res


def create_ts_features_parallel(
    start_index: int,
    end_index: Optional[int] = None,
    ts_func: Optional[Callable] = None,
    n_cores: Optional[int] = None,
    logical: bool = False,
    n_chunks: Optional[int] = None,
    **kwargs,
):
    """Split ``[start_index, end_index)`` into ``n_chunks`` chunks, run ``ts_func`` in parallel, join results."""
    nrecords_per_period = kwargs.get("nrecords_per_period", 1)

    if end_index is None:
        df = kwargs.get("df")
        if df is None:
            return None, None
        end_index = len(df)
        if not end_index:
            return None, None
        end_index = end_index // nrecords_per_period

    if not n_chunks or n_chunks <= 0:
        # Honour the caller's `logical` flag rather than hard-coding `logical=True`.
        cpu_count = psutil.cpu_count(logical=logical) or psutil.cpu_count(logical=True) or 1
        n_chunks = min(int(cpu_count * 1.5), (end_index - start_index))

    step = (end_index - start_index) // n_chunks
    if step < 1:
        return None, None

    args = []
    left = start_index
    for i in range(n_chunks):
        right = min(left + step, end_index)
        if i == n_chunks - 1:
            right = end_index
        args.append((left, right))
        left = right

    logger.info("starting applyfunc_parallel using args %s", args)
    res = applyfunc_parallel(args, partial(ts_func, **kwargs), return_dataframe=False, logical=logical, n_cores=n_cores)
    X_parts, Y_parts = [], []
    for X_part, Y_part in res:
        X_parts.append(X_part)
        Y_parts.append(Y_part)

    del res
    return pd.concat(X_parts, ignore_index=True), pd.concat(Y_parts, ignore_index=True)


def compute_corr(
    dependent_vals: np.ndarray,
    independent_vals: np.ndarray,
    deciding_func: Callable,
    absolutize: bool = True,
) -> float:
    """Correlation under an arbitrary deciding function.

    For ``np.corrcoef`` we take ``[0][1]`` (the off-diagonal entry of the 2x2 matrix).
    For sklearn-style ``f_regression``/``mutual_info_regression`` we take ``[0]`` of the returned
    score vector. Callers passing a different deciding_func must conform to one of these shapes.
    """
    if deciding_func is np.corrcoef:
        corr = deciding_func(dependent_vals, independent_vals)[0][1]
    else:
        corr = deciding_func(dependent_vals.reshape(-1, 1), independent_vals)[0]

    if absolutize:
        corr = np.abs(corr)

    return corr


def general_acf(
    Y: np.ndarray,
    X: Optional[pd.DataFrame] = None,
    windows: Optional[Dict[str, Dict[str, float]]] = None,
    deciding_func: Callable = np.corrcoef,
    lag_len: int = 30,
    min_samples: int = 500,
    absolutize: bool = True,
) -> Dict[str, pd.Series]:
    """Advanced ACF: fixed integer lags + optional variable-driven non-fixed offsets.

    ``windows`` example: ``{"Load": {"from": 40_000, "to": 1e6, "nsteps": 100}}``.
    """
    if windows is None:
        windows = {}
    res: Dict[str, pd.Series] = {}

    if lag_len:
        # Seed lag-0 at index 0 (autocorrelation at lag 0 is 1 by definition).
        acfs_vals: List[float] = [1.0]
        acfs_index: List[float] = [0.0]
        for i in tqdmu(range(lag_len), desc="Fixed offsets"):
            # After slicing by (i+1) the effective sample size is `len(Y) - (i+1)`.
            if len(Y) - (i + 1) >= min_samples:
                dependent_vals = Y[i + 1 :]
                independent_vals = Y[: -(i + 1)]
                corr = compute_corr(
                    dependent_vals=dependent_vals,
                    independent_vals=independent_vals,
                    deciding_func=deciding_func,
                    absolutize=absolutize,
                )
                acfs_vals.append(corr)
                acfs_index.append(i + 1)

        res["fixed_offsets"] = pd.Series(data=acfs_vals, name="fixed_offsets", index=acfs_index)

    if windows:
        if X is None:
            raise ValueError("general_acf: X is required when windows are supplied")
        for window_var, windows_params in tqdmu(windows.items(), desc="Flexible windows"):
            window_var_values = X[window_var].values
            acfs_vals = [1.0]
            acfs_index = [0.0]
            window_sizes = np.linspace(
                start=windows_params.get("from", max(1, window_var_values.min())),
                stop=windows_params.get("to", window_var_values.sum()),
                num=windows_params.get("nsteps", 100),
            )

            for window_size in tqdmu(window_sizes, desc=window_var):
                dependent_vals: list = []
                independent_vals: list = []
                windows_r = len(window_var_values) - 1

                while True:
                    windows_l, accumulated_amount = find_next_cumsum_left_index(
                        window_var_values=window_var_values, amount=window_size, right_index=windows_r
                    )
                    if accumulated_amount * 2 < window_size:
                        break
                    dependent_vals.append(Y[windows_r])
                    independent_vals.append(Y[windows_l])
                    windows_r = windows_l
                    if windows_l <= 0:
                        break
                if dependent_vals:
                    corr = compute_corr(
                        dependent_vals=np.asarray(dependent_vals),
                        independent_vals=np.asarray(independent_vals),
                        deciding_func=deciding_func,
                        absolutize=absolutize,
                    )
                    acfs_vals.append(corr)
                    acfs_index.append(window_size)

            res[window_var] = pd.Series(data=acfs_vals, name=window_var, index=acfs_index)
    return res
