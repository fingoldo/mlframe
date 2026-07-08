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
from typing import Callable, Dict, List, Optional, Pattern, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import psutil
from numba import njit

from mlframe.feature_engineering.categorical import get_countaggs_names
from mlframe.feature_engineering.numerical import get_numaggs_names
from pyutilz.parallel import applyfunc_parallel
from pyutilz.pythonlib import get_human_readable_set_size
from pyutilz.system import tqdmu

logger = logging.getLogger(__name__)

# Default fill / correction constants. Hoisted to module scope so they're not rebuilt per call
# and so callers can introspect them.
_DEFAULT_NA_FILL: float = 1e3
_DEFAULT_SPAN_CORRECTION: float = 1e2
_MIN_WINDOW_FILL_RATIO: float = 0.5  # window is skipped if cumsum < 50% of the requested size

# Per-call dict / list literals (e.g. `{"": 1e3}`, `[np.cbrt]`) used to rebuild on every entry; cached at module scope.
# Callers must not mutate these - functions read them as fall-throughs only.
_DEFAULT_NA_FILLS: Dict[str, float] = {"": _DEFAULT_NA_FILL}
_DEFAULT_SPAN_CORRECTIONS: Dict[str, float] = {"": _DEFAULT_SPAN_CORRECTION}
_DEFAULT_NONLINEAR_TRANSFORMS: tuple = (np.cbrt,)


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


@njit(cache=True)
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
    return cast(Tuple[int, float], _find_next_cumsum_left_index_njit(window_var_values, amount, right_int, min_samples, use_abs))


@njit(cache=True)
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
    return cast(Tuple[int, float], _find_next_cumsum_right_index_njit(window_var_values, amount, left_int, min_samples, use_abs))


def get_nwindows_expected(windows: Dict[str, Sequence]) -> int:
    """Total number of windows produced by ``create_and_process_windows`` for the given spec."""
    return sum(len(v) for v in windows.values())


def get_ts_window_name(window_var: str, window_size: float, window_index_name: str = "") -> str:
    """Human-readable short name for a time-series window."""
    if window_var == "":
        return str(window_size) + window_index_name
    return window_var + ":" + str(get_human_readable_set_size(window_size))


# Wave 96 (2026-05-21): the 11 _emit_* per-transform helpers moved to
# sibling file _timeseries_emit.py to drop this file below the 1k-line
# monolith threshold. Re-exported below so existing callers
# (, etc.)
# keep working.
from ._timeseries_emit import (
    _emit_groupby_block,
    _emit_categorical_counts,
    _emit_raw_numaggs,
    _emit_differences,
    _emit_ratios,
    _emit_wavelets,
    _emit_weighted,
    _emit_ewma,
    _emit_rolling,
    _emit_nonlinear,
    _emit_robust,
    _emit_counts_regexp,
)


def create_aggregated_features(  # nosec B107 - default is a separator/label token, not a credential
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
        nonlinear_transforms = list(_DEFAULT_NONLINEAR_TRANSFORMS)
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
                features_names.append(captions_vars_sep.join((cast(str, dataset_name), var, "n_finite")))

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

            # Pull the subset column into a numpy view ONCE per ``subset_var``; each value comparison then operates on that view rather than re-aligning a pandas boolean Series. ``to_numpy()`` is a view for numeric dtypes and a one-column decode for Categorical -- either is O(N) once, not O(N*|values|).
            _subset_vals = window_df[subset_var].to_numpy()
            for subset_var_value in subset_var_values:
                # Compute the boolean mask ONCE on the numpy view (was: two pandas-aligned mask evaluations on the direct + complement path). ``np.flatnonzero`` materialises a positional-index array used to drive the single ``.iloc`` slice below, avoiding the pre-fix "copy on the direct path, then maybe copy AGAIN on the complement when ``len(subset_df)<=1``" pattern -- now we know which set to keep BEFORE the BlockManager copy fires.
                _bm = _subset_vals == subset_var_value
                _pos_idx = np.flatnonzero(_bm)
                subset_direct = True
                if len(_pos_idx) <= 1:
                    _pos_idx = np.flatnonzero(~_bm)
                    subset_direct = False
                subset_df = window_df.iloc[_pos_idx]

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
                )  # nosec B107 - default value is a plain separator/label string, not a credential


def compute_splitting_stats(
    window_df: pd.DataFrame,
    dataset_name: Optional[str],
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
    # Wave 39 (2026-05-20): empty window after upstream isfinite filter is reachable;
    # iloc[0]/iloc[-1] on empty frame raises IndexError. Treat as no-op.
    if len(window_df) == 0:
        return
    splitting_vals: list = []
    splitting_ratios_names: Optional[list] = [] if create_features_names else None
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
                if create_features_names and splitting_ratios_names is not None:
                    splitting_ratios_names.append(captions_vars_sep.join([cast(str, dataset_name), var, col, subvar, "split"]))

    row_features.extend(splitting_vals)
    if create_features_names and splitting_ratios_names is not None:
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

    # Wave 69 (2026-05-20): symmetric past/future window-count expectation; the
    # past-side check fires below after past_windows_features is computed.
    past_nwindows_expected = get_nwindows_expected(past_windows)
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

        # Wave 69 (2026-05-20): past-side window-count sanity check, symmetric
        # with the future-side check at the future-windows branch above. Skip
        # the row when past windows didn't produce the expected count (data
        # boundary -- not enough history yet at this base_point).
        if past_nwindows_expected and not past_windows_features and (features_creation_fcn or not row_features):
            continue

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

    # bench-attempt-rejected (2026-05-24): conditional skip of the dtype= kwarg (skip when features ndarray dtype already matches) is moot here -- features
    # arrives as a list-of-lists from the outer accumulation loop, with no .dtype attribute. Constructing without dtype= gives object dtype and forces a
    # downstream astype anyway; passing dtype= upfront is the correct shape. Measured at N=10k D=50: list-of-rows + dtype= = 105 ms, no dtype = 97 ms +
    # mandatory astype = parity. Keep dtype= passed.
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
    apply_fcn: Optional[Callable],
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
                    windows_r, accumulated_amount = find_next_cumsum_right_index(window_var_values=window_var_values, amount=window_size, left_index=windows_l)
                else:
                    windows_l, accumulated_amount = find_next_cumsum_left_index(window_var_values=window_var_values, amount=window_size, right_index=windows_r)
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
                if apply_fcn is None:
                    raise ValueError(f"create_and_process_windows: apply_fcn is required to process non-empty window '{dataset_name}'")
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
    if ts_func is None:
        raise ValueError("create_ts_features_parallel: ts_func is required")

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

    return float(corr)


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

            # Each window_size re-walks window_var_values from the right (find_next_cumsum_left_index) to carve the
            # flexible cumsum windows -- an O(L) rescan per scale, O(L * nsteps) overall. Not trivially cacheable: the
            # window boundaries are a function of window_size (which varies per iteration) AND the running right anchor,
            # so the partition is different every step and no prefix-sum table reuses across scales without changing
            # the carved windows. Left as-is; revisit only if this becomes a measured hotspot on real flexible-window FE.
            for window_size in tqdmu(window_sizes, desc=window_var):
                dependent_vals_list: list = []
                independent_vals_list: list = []
                windows_r = len(window_var_values) - 1

                while True:
                    windows_l, accumulated_amount = find_next_cumsum_left_index(window_var_values=window_var_values, amount=window_size, right_index=windows_r)
                    if accumulated_amount * 2 < window_size:
                        break
                    dependent_vals_list.append(Y[windows_r])
                    independent_vals_list.append(Y[windows_l])
                    windows_r = windows_l
                    if windows_l <= 0:
                        break
                if dependent_vals_list:
                    corr = compute_corr(
                        dependent_vals=np.asarray(dependent_vals_list),
                        independent_vals=np.asarray(independent_vals_list),
                        deciding_func=deciding_func,
                        absolutize=absolutize,
                    )
                    acfs_vals.append(corr)
                    acfs_index.append(window_size)

            res[window_var] = pd.Series(data=acfs_vals, name=window_var, index=acfs_index)
    return res
