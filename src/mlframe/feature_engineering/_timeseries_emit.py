"""Per-transform emit helpers for `timeseries.create_aggregated_features`.

Wave 96 (2026-05-21): split out from `timeseries.py` to keep that file
below the 1k-line threshold. Behaviour preserved bit-for-bit; every
helper is re-exported from `timeseries` so existing
``from mlframe.feature_engineering.timeseries import _emit_robust``
imports continue to work.

Each helper is responsible for ONE transform (raw numaggs, differences,
ratios, wavelets, weighted, ewma, rolling, non-linear, robust,
categorical counts, groupby). They extend ``row_features`` and
``features_names`` in place with byte-identical output order to the
prior inlined code.
"""
from __future__ import annotations

import numpy as np
import pywt

from mlframe.core.ewma import ewma_numba
from mlframe.feature_engineering.categorical import compute_countaggs
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
from pyutilz.numpylib import smart_ratios

# Mirror the parent module's default-fill constants for ratio helpers.
_DEFAULT_NA_FILL: float = 1e3
_DEFAULT_SPAN_CORRECTION: float = 1e2


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
                features_names.extend(captions_vars_sep.join((dataset_name, sum_var, "grpby", var, feat)) for feat in numaggs_names)


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
            features_names.extend(captions_vars_sep.join((dataset_name, var, "vlscnt", feat)) for feat in countaggs_names)


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
        features_names.extend(captions_vars_sep.join((dataset_name, var, feat)) for feat in simple_numaggs_names)
    return simple_numerical_features, simple_numaggs_names, custom_numaggs_kwds


def _emit_differences(
    var, raw_vals, numaggs_kwds, dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """First-difference block. return_profit_factor is forced on since differences are the
    natural series for gain/loss ratio stats (unlike the raw level series)."""
    differences = np.diff(raw_vals, 1)
    custom_numaggs_kwds_diffs = numaggs_kwds.copy()
    custom_numaggs_kwds_diffs["return_profit_factor"] = True
    row_features.extend(compute_numaggs(differences, **custom_numaggs_kwds_diffs))
    if create_features_names:
        features_names.extend(captions_vars_sep.join((dataset_name, var, "dif", feat)) for feat in get_numaggs_names(**custom_numaggs_kwds_diffs))


def _emit_ratios(
    var, raw_vals, numaggs_kwds, span_corrections, na_fills,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Lag-1 ratio block (raw_vals[t]/raw_vals[t-1]) via smart_ratios, which handles
    zero/near-zero denominators with span_correction and na_fill instead of producing inf/NaN."""
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
        features_names.extend(captions_vars_sep.join((dataset_name, var, "rat", feat)) for feat in get_numaggs_names(**custom_numaggs_kwds_ratios))


def _emit_wavelets(
    var, raw_vals, waveletnames, numaggs_kwds, wavelets_correction_numaggs_kwds,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Emit one numagg block per wavelet name in waveletnames, over the flattened
    multi-level wavedec coefficients (approximation + all detail levels concatenated)."""
    custom_numaggs_kwds_wave = numaggs_kwds.copy()
    custom_numaggs_kwds_wave.update(wavelets_correction_numaggs_kwds)
    for waveletname in waveletnames:
        all_coeffs = np.hstack(list(pywt.wavedec(raw_vals, waveletname)))
        row_features.extend(compute_numaggs(all_coeffs, **custom_numaggs_kwds_wave))
        if create_features_names:
            features_names.extend(captions_vars_sep.join((dataset_name, var, waveletname, feat)) for feat in get_numaggs_names(**custom_numaggs_kwds_wave))


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
                features_names.extend(captions_vars_sep.join((dataset_name, var, "wgt", weighting_var, feat)) for feat in numaggs_names)


def _emit_ewma(
    var, raw_vals, ewma_alphas, numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Exponentially-weighted moving average block, one numagg group per alpha in ewma_alphas.
    Empty raw_vals pads with zeros rather than calling ewma_numba on a zero-length array."""
    for alpha in ewma_alphas:
        if len(raw_vals) > 0:
            row_features.extend(compute_numaggs(ewma_numba(raw_vals.astype(np.float32), alpha), **numaggs_kwds))
        else:
            row_features.extend([0.0] * len(numaggs_names))
        if create_features_names:
            features_names.extend(captions_vars_sep.join((dataset_name, var, "ewma", str(alpha), feat)) for feat in numaggs_names)


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
            specs = ";".join(f"{key}={value}" for key, value in dict(**window, m=method, **method_params).items())
            specs = specs.replace("win_type", "t").replace("window", "w")
            features_names.extend(captions_vars_sep.join((dataset_name, var, "rol", specs, feat)) for feat in numaggs_names)


def _emit_nonlinear(
    var, raw_vals, nonlinear_transforms, numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Apply each callable in nonlinear_transforms (e.g. log, sqrt, square) to raw_vals and emit
    a numagg block per transform; the transform's __name__ becomes part of the feature caption."""
    for nonlinear_func in nonlinear_transforms:
        transform_name = nonlinear_func.__name__
        row_features.extend(compute_numaggs(nonlinear_func(raw_vals), **numaggs_kwds))
        if create_features_names:
            features_names.extend(captions_vars_sep.join((dataset_name, var, transform_name, feat)) for feat in numaggs_names)


def _emit_robust(
    var, raw_vals, simple_numerical_features, simple_numaggs_names,
    numaggs_kwds, numaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Robust quantile-trimmed subset (Tukey-fence). q1_idx_local can be 0 - the old
    `if q1_idx and q3_idx:` truthy check silently dropped the q0.25-at-index-0 case. Use
    explicit `is not None` here (P0 fix)."""
    # Lazy import: get_numaggs_metadata lives in the parent module; importing it at module-top
    # would create a circular load (timeseries imports from us, we import from timeseries).
    from .timeseries import get_numaggs_metadata

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
            features_names.extend(captions_vars_sep.join((dataset_name, var, "rbst", feat)) for feat in numaggs_names)


def _emit_counts_regexp(
    series, var, countaggs_kwds, countaggs_names,
    dataset_name, captions_vars_sep,
    row_features, features_names, create_features_names,
):
    """Per-variable counts processing triggered by counts_processing_mask_regexp (independent of
    dtype). Used for integer count-like columns that aren't pandas categoricals."""
    row_features.extend(compute_countaggs(series, **countaggs_kwds))
    if create_features_names:
        features_names.extend(captions_vars_sep.join((dataset_name, var, "vlscnt", feat)) for feat in countaggs_names)
