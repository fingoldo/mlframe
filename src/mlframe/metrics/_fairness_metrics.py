"""Fairness / robustness subgrouping + metrics for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import create_fairness_subgroups`` (and the
other moved names) imports continue to work.

What lives here:
  - ``create_fairness_subgroups`` (auto-bin numeric + categorical features
    with low-pop merge/exclude policy)
  - ``create_fairness_subgroups_indices`` (project bins into per-split index
    arrays)
  - ``create_robustness_standard_bins`` (the ``**ORDER**`` / ``**RANDOM**``
    pseudo-features)
  - ``compute_fairness_metrics`` (per-bin metric values + Tukey-fence
    outlier-flagging + top-N best/worst formatting)
  - ``robust_mlperf_metric`` (boosting-callable scoring fn that combines
    whole-set metric with per-subgroup mean +/- stdev)
  - Backward-compatible aliases: ``create_robustness_subgroups`` etc.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd

from pyutilz.pythonlib import sort_dict_by_value
from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile

logger = logging.getLogger(__name__)


def create_fairness_subgroups(
    df: pd.DataFrame,
    features: Sequence[Union[str, pd.Series]],
    cont_nbins: int = 3,
    min_pop_cat_thresh: Union[float, int] = 1000,
    merge_lowpop_cats: bool = True,
    exclude_terminal_lowpop_cats: bool = True,
    rare_group_name: str = "*RARE*",
) -> dict:
    """Create subgroups for fairness evaluation across demographic/categorical features.

    Fairness analysis evaluates model performance consistency across different
    demographic groups (e.g., age, gender, region) or categorical segments to
    ensure the model doesn't discriminate against specific subpopulations.

    Subgrouping splits observations into bins for which ML metrics are calculated separately.
    Use this when you need consistent & fair performance across subgroups - different geographical
    regions, client types, demographic segments, etc.

    For categorical variables, each category forms a natural bin.
    Low-populated categories (<min_pop_cat_thresh) are merged into a single 'rarevals' bin or excluded.
    Subgroups can have different weights (by default equal).

    How metrics are adjusted: From/to the original metric on entire dataset, weighted sum of stdevs over
    subgroups is deducted/added (depending on greater_is_better). An ideally fair model has zero stdevs.

    Final ML report includes: subgroup name, nbins, metric stdev, outliers, best/worst bins & performance."""

    # Bad values used to slip past as ``assert`` under -O and crashed
    # deeper in pandas grouping with an opaque message.
    if isinstance(min_pop_cat_thresh, float):
        if not (0 < min_pop_cat_thresh < 1.0):
            raise ValueError(
                f"min_pop_cat_thresh (float) must be in (0, 1); got {min_pop_cat_thresh!r}."
            )
        min_pop_cat_thresh = int(len(df) * min_pop_cat_thresh)  # convert to abs value
    elif isinstance(min_pop_cat_thresh, int):
        if not (0 < min_pop_cat_thresh <= len(df) // 2):
            raise ValueError(
                f"min_pop_cat_thresh (int) must be in (0, len(df)//2={len(df) // 2}]; "
                f"got {min_pop_cat_thresh!r}."
            )

    subgroups = {}
    for feature_name in features:

        if feature_name in ("**ORDER**", "**RANDOM**"):
            subgroups[feature_name] = feature_name
            continue

        if isinstance(feature_name, pd.Series):
            feature_vals = feature_name
            feature_name = feature_vals.name
        else:
            feature_vals = df[feature_name]

        val_cnts = feature_vals.value_counts()

        # qcut requires a NUMERIC dtype (it computes quantile cut-points).
        # A prior blacklist ``not in ("category","object","date","datetime")``
        # silently let pandas StringDtype / pyarrow large_string slip past
        # ("string" is none of those names) and pd.qcut then crashed deep in
        # pyarrow with ``ArrowNotImplementedError: Function 'quantile' has
        # no kernel matching input types (large_string)`` (observed on S:
        # in fuzz_3way c0133_7257609a-cb_hgb_lgb_linear-pl_utf8-n5000;
        # the ``pl_utf8`` input type converted to pyarrow-backed large_string
        # on the pandas side). Switch to ``is_numeric_dtype`` so the gate is
        # whitelist-style: only int/float/bool/numeric extension types reach
        # qcut; everything else uses the value_counts categorical path below.
        if pd.api.types.is_numeric_dtype(feature_vals):
            if len(val_cnts) > cont_nbins:
                feature_vals = pd.qcut(feature_vals, q=cont_nbins, labels=None)  # use qcut for equipopulated binning
                val_cnts = feature_vals.value_counts()  # this needs recalculation now

        # use categories as natural bins. ensure that low-populated cats are merged if possible (merge_lowpop_cats)
        # or excluded (exclude_terminal_lowpop_cats).

        rarecats = val_cnts[val_cnts < min_pop_cat_thresh]
        if len(rarecats) > 0:
            cats = rarecats.index.values.tolist()
            if merge_lowpop_cats and rarecats.sum() >= min_pop_cat_thresh:
                # merging is possible
                feature_vals = feature_vals.copy().replace({cat: rare_group_name for cat in cats})
                val_cnts = feature_vals.value_counts()  # this needs recalculation now
                cats_to_use = val_cnts.index.values.tolist()
                logger.info(f"For feature {feature_name}, had to merge {len(cats):_} bins {','.join(map(str,cats))}, {rarecats.sum():_} records.")
            else:
                if exclude_terminal_lowpop_cats:
                    cats_to_use = val_cnts[val_cnts >= min_pop_cat_thresh].index.values.tolist()
                    logger.info(f"For feature {feature_name}, had to exclude {len(cats):_} bins {','.join(map(str,cats))}, {rarecats.sum():_} records.")
        else:
            cats_to_use = val_cnts.index.values.tolist()

        if len(cats_to_use) > 1:
            subgroups[feature_name] = dict(bins=feature_vals, bins_names=cats_to_use)
        else:
            logger.warning(f"Feature {feature_name} can't particiate in subgrouping: it has only one bin.")

    return subgroups


def create_fairness_subgroups_indices(
    subgroups: dict, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, group_weights: dict = None, cont_nbins: int = 3
) -> dict:
    """Create index mappings for fairness subgroups across train/val/test splits.

    Converts fairness subgroups (demographic/categorical bins) into index arrays
    for each data split, enabling per-subgroup metric computation.
    """
    if group_weights is None:
        group_weights = {}
    res = {}
    if len(val_idx) == len(test_idx):
        logger.warning("Validation and test sets have the same size. Fairness subgroups estimation will be incorrect.")
    for arr in (train_idx, test_idx, val_idx):
        npoints = len(arr)
        fairness_subgroups_indices = {}
        for group_name, group_params in subgroups.items():
            group_indices = {}
            if group_name in ("**ORDER**", "**RANDOM**"):
                bins, unique_bins = create_robustness_standard_bins(group_name=group_name, npoints=npoints, cont_nbins=cont_nbins)
            else:
                bins = group_params.get("bins")
                # SILENT-CORRECTNESS bug under -O when this was an ``assert``:
                # a non-unique bins.index makes ``bins.loc[arr]`` return
                # multiple rows per key, silently corrupting the fairness-eval
                # output with duplicated/oversized partitions. Loud failure
                # required.
                if not bins.index.is_unique:
                    raise ValueError(
                        "create_fairness_subgroups_indices: group_params['bins'] "
                        "must have a unique index; pre-fix duplicate index "
                        "caused bins.loc[arr] to return multiple rows per key, "
                        "silently corrupting fairness-eval output."
                    )
                bins = bins.loc[arr]
                unique_bins = None

            if unique_bins is None:
                if isinstance(bins, pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)

            # The per-bin ``np.where(bins == bin_name)[0]`` loop is O(n*B): for a high-card categorical (B bins) over object dtype it runs B full element-wise
            # string comparisons over all n rows (B=200 over n=1M was ~12 s/call, dominated by pandas ``comp_method_OBJECT_ARRAY``). A single factorize +
            # stable argsort partitions all groups in one O(n log n) pass; per-group positional index arrays are contiguous slices of the sorted order and stay
            # ascending (stable sort) -- bit-identical to the per-bin ``np.where`` result.
            bins_arr = bins.to_numpy() if isinstance(bins, pd.Series) else np.asarray(bins)
            codes, uniques = pd.factorize(bins_arr, sort=False)
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            boundaries = np.searchsorted(sorted_codes, np.arange(len(uniques)), side="left")
            ends = np.searchsorted(sorted_codes, np.arange(len(uniques)), side="right")
            code_of = {bin_name: i for i, bin_name in enumerate(uniques)}
            for bin_name in unique_bins:
                # A NaN bin_name (degenerate bin) factorizes to code -1 and is absent from ``uniques``; ``bins == NaN`` matched nothing, so emit an empty array.
                c = code_of.get(bin_name, -1)
                if c < 0:
                    group_indices[bin_name] = np.empty(0, dtype=np.intp)
                else:
                    group_indices[bin_name] = order[boundaries[c]:ends[c]]

            fairness_subgroups_indices[group_name] = dict(bins=group_indices, weight=group_weights.get(group_name, 1.0))

        res[npoints] = fairness_subgroups_indices

    return res


def create_robustness_standard_bins(group_name: str, npoints: int, cont_nbins: int, seed: int = 0) -> tuple:

    step_size = npoints // cont_nbins
    # int16 wraps if cont_nbins > 32767. Use a range-aware narrowest dtype
    # that fits the caller's cont_nbins-1 max so unusual callers don't wrap
    # silently.
    if cont_nbins - 1 <= np.iinfo(np.int8).max:
        _bin_dtype = np.int8
    elif cont_nbins - 1 <= np.iinfo(np.int16).max:
        _bin_dtype = np.int16
    else:
        _bin_dtype = np.int32
    bins = np.empty(shape=npoints, dtype=_bin_dtype)
    start = 0
    unique_bins = range(cont_nbins)
    for i in unique_bins:
        bins[start : start + step_size] = i
        start += step_size
    if group_name == "**RANDOM**":
        # local seeded generator: the global np.random.shuffle is unseeded -> the random
        # fairness baseline differed run-to-run; same inputs+seed must give the same bins.
        np.random.default_rng(seed).shuffle(bins)

    return bins, unique_bins


def compute_fairness_metrics(
    metrics: dict,
    metrics_higher_is_better: dict,
    subgroups: dict,
    subset_index: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cont_nbins: int = 3,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compute fairness metrics across demographic/categorical subgroups.

    Evaluates model performance consistency across different subpopulations
    to identify potential bias or discrimination. * is added to the bin name
    if bin's metric is an outlier (computed using Tukey's fence & IQR)."""

    # Signature declares np.ndarray for y_true / y_pred but upstream
    # report_regression_model_perf threads through whatever the model
    # returned, which can be a polars Series for native polars-fastpath
    # models. The bin loop below builds idx as a pandas boolean Series
    # (when ``bins`` is a pandas Series) and then does y_pred[idx]; polars
    # Series.__getitem__ rejects pandas-Series keys with TypeError. Coerce
    # to numpy at the boundary so the bin loop sees a uniform indexable
    # surface regardless of caller-side carrier type.
    if hasattr(y_true, "to_numpy"):
        y_true = y_true.to_numpy()
    else:
        y_true = np.asarray(y_true)
    if hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()
    else:
        y_pred = np.asarray(y_pred)

    if subgroups:

        res = []
        quantile = 0.25
        quantiles_to_compute = [0.5 - quantile, 0.5, 0.5 + quantile]
        tukey_mult = get_tukey_fences_multiplier_for_quantile(quantile=quantile, sd_sigma=2.7)

        for group_name, group_params in subgroups.items():
            if group_name in ("**ORDER**", "**RANDOM**"):
                bins, unique_bins = create_robustness_standard_bins(group_name=group_name, npoints=len(y_true), cont_nbins=cont_nbins)
            else:
                bins = group_params.get("bins")
                if bins is not None:
                    if subset_index is None:
                        raise RuntimeError(
                            "compute_fairness_metrics: bins is set but "
                            "subset_index is None; the function's state "
                            "machine reached an unreachable branch."
                        )
                    bins = bins.loc[subset_index]
                bins_names = group_params.get("bins_names")  # noqa: F841 -- preserved for parity with original
                unique_bins = None

            npoints = []
            perfs = defaultdict(dict)
            if unique_bins is None:
                if isinstance(bins, pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)
            for bin_name in unique_bins:
                idx = np.asarray(bins == bin_name)
                n_points = idx.sum()
                if n_points:
                    npoints.append(n_points)
                    for metric_name, metric_func in metrics.items():
                        if y_pred.ndim == 2:
                            metric_value = metric_func(y_true[idx], y_pred[idx, :])
                        else:
                            metric_value = metric_func(y_true[idx], y_pred[idx])
                        perfs[metric_name][f"{bin_name} [{n_points}]"] = metric_value

            for metric_name, metric_perf in perfs.items():

                metric_perf = sort_dict_by_value(metric_perf)
                npoints = np.array(npoints)
                line = dict(
                    factor=group_name,
                    metric=metric_name,
                    nbins=len(unique_bins),
                    npoints_from=npoints.min(),
                    npoints_median=int(np.median(npoints)),
                    npoints_to=npoints.max(),
                )

                # ``np.array(list(metric_perf.values()))`` materialised an
                # intermediate Python list (one allocation + one ndarray
                # alloc). ``np.fromiter`` consumes the dict_values view
                # directly: one buffer alloc, ~2x less peak memory on
                # large groupings. Performance is sub-second here, but
                # the cleaner shape also makes the dtype contract
                # explicit (was object-dtype-by-accident if any metric
                # returned non-float).
                # ``numba.njit`` was considered but is not applicable:
                # ``metric_func`` is a Python callable / closure (sklearn
                # metric), ``metric_perf`` is a sorted Python dict, and
                # the dominant cost upstream is the per-bin metric call
                # itself, not the quantile compute. Numba would not
                # touch the bottleneck.
                performances = np.fromiter(
                    metric_perf.values(),
                    dtype=np.float64,
                    count=len(metric_perf),
                )
                # ``nanquantile`` so a NaN bin (degenerate metric) doesn't
                # poison the Tukey outlier thresholds below.
                quantiles = np.nanquantile(performances, q=quantiles_to_compute)
                iqr = quantiles[-1] - quantiles[0]
                min_boundary = quantiles[0] - tukey_mult * iqr
                max_boundary = quantiles[-1] + tukey_mult * iqr

                # nanmean / nanstd so the metric_mean / metric_std columns
                # aren't silently NaN whenever a single bin emits NaN.
                line["metric_mean"] = float(np.nanmean(performances))
                line["metric_std"] = float(np.nanstd(performances))

                l = len(metric_perf)
                real_top_n = min(l // 2, top_n)

                for i, (bin_name, metric_value) in enumerate(metric_perf.items()):
                    if metric_value < min_boundary or metric_value > max_boundary:
                        postfix = "*"
                    else:
                        postfix = ""
                    if i < real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-worst-" + str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-best-" + str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                    elif i >= l - real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-best-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-worst-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"

                res.append(line)
        if res:
            res = pd.DataFrame(res).set_index(["factor", "nbins", "npoints_from", "npoints_median", "npoints_to", "metric"])
            return res.reindex(sorted(res.columns), axis=1)


# Backward-compatible aliases for renamed fairness functions
create_robustness_subgroups = create_fairness_subgroups
create_robustness_subgroups_indices = create_fairness_subgroups_indices
compute_robustness_metrics = compute_fairness_metrics


def robust_mlperf_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable,
    higher_is_better: bool,
    subgroups: dict = None,
    whole_set_weight: float = 0.5,
    min_group_size: int = 100,
) -> float:
    """Bins idices need to be aware of arr sizes: boostings can call the metric on
    multiple sets of differnt lengths - train, val, etc. Arrays will be pure numpy, so no other means to
    distinguish except the arr size."""

    weights_sum = whole_set_weight
    total_metric_value = metric(y_true, y_score) * whole_set_weight

    l = len(y_true)
    if subgroups and l in subgroups:

        for group_name, group_params in subgroups[l].items():

            bins = group_params.get("bins")
            bin_weight = group_params.get("weight", 1.0)

            perfs = []
            for bin_name, bin_indices in bins.items():
                if len(bin_indices) < min_group_size:
                    continue
                if isinstance(y_score, Sequence):
                    if len(y_score) == 2:
                        metric_value = metric(y_true[bin_indices], [el[bin_indices] for el in y_score])
                    else:
                        metric_value = metric(y_true[bin_indices], y_score[1][bin_indices])
                else:
                    if y_score.ndim == 2:
                        metric_value = metric(y_true[bin_indices], y_score[bin_indices, :])
                    else:
                        metric_value = metric(y_true[bin_indices], y_score[bin_indices])
                perfs.append(metric_value)

            if perfs:
                perfs = np.array(perfs)
                bin_metric_value = perfs.mean()
                if higher_is_better:
                    bin_metric_value -= perfs.std()
                else:
                    bin_metric_value += perfs.std()

                weights_sum += bin_weight
                total_metric_value += bin_metric_value * bin_weight

    return total_metric_value / weights_sum
