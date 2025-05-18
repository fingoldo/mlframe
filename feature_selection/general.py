# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numpy as np, pandas as pd
import polars as pl, polars.selectors as cs

import psutil
import textwrap
from timeit import default_timer as timer
from collections import defaultdict

from pyutilz.system import clean_ram, tqdmu
from pyutilz.polarslib import bin_numerical_columns
from pyutilz.benchmarking import benchmark_algos_by_runtime

from mlframe.feature_selection.mi import grok_compute_mutual_information, chatgpt_compute_mutual_information, deepseek_compute_mutual_information

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def benchmark_mi_algos(base_mi_algos: list, verbose: int = 0) -> list:

    target_indices = np.array([0, 10, 20], dtype=np.int64)

    # prewarm
    arr = np.random.randint(0, 15, size=(10, 200), dtype=np.int8)
    for func in base_mi_algos:
        _ = func(data=arr, target_indices=target_indices)

    # main
    arr = np.random.randint(0, 15, size=(1_000_000, 200), dtype=np.int8)
    base_mi_algos, durations = benchmark_algos_by_runtime(
        implementations=base_mi_algos, algo_name="MI", n_reps=2, verbose=verbose, data=arr, target_indices=target_indices
    )

    return base_mi_algos


def estimate_features_relevancy(
    # data
    bins: pl.DataFrame,
    target_columns: list,
    # precomputed info
    mi_algorithms_ranking: list = None,  # ltr
    benchmark_mi_algorithms: bool = True,
    permuted_mutual_informations: dict = None,
    # working params
    min_mi_prevalence: float = 2,  # 10 is too high for a weak target
    permuted_max_mi_quantile: float = None,
    min_permuted_mi_evaluations: int = 1000,
    min_randomized_permutations: int = 1,
    max_permuted_prevalence_percent: float = 0.0,
    # stopping criteria
    max_runtime_mins: float = None,
    # style
    leave_progressbar: bool = False,
    max_log_text_width: int = 300,
    verbose: int = 1,
):
    """Computes relevancy of all features to the targets, using integer bins computed at previous step.
    Suggests for droppping columns that have no firm impact on any of the targets.

    We think that a feature has impact on target if:
        it's MI with original target is higher than MI with permuted target in (1-max_permuted_prevalence_percent)
            (default ALL) permutations we tried;
        it's MI with original target is at least min_mi_prevalence times higher than permuted_max_mi_quantile
            (default MAXIMUM) MI with permuted target across all features and permutaions we tried.

    Either min_randomized_permutations or max_runtime_mins should be specified.

    Reports:

    """
    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    start_time = timer()
    ran_out_of_time = False

    columns_to_drop = []

    assert min_randomized_permutations >= 1

    # ----------------------------------------------------------------------------------------------------------------------------
    # What MI implementation is the fastest for current machine?
    # ----------------------------------------------------------------------------------------------------------------------------

    if not mi_algorithms_ranking:
        base_mi_algos = [chatgpt_compute_mutual_information, grok_compute_mutual_information, deepseek_compute_mutual_information]

        if benchmark_mi_algorithms:
            mi_algorithms_ranking = benchmark_mi_algos(base_mi_algos=base_mi_algos, verbose=verbose)
        else:
            mi_algorithms_ranking = base_mi_algos

    # ----------------------------------------------------------------------------------------------------------------------------
    # For each of the targets, compute joint freqs and then MI for each of the "normal" columns:
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Computing original MIs...")

    arr = bins.to_numpy(allow_copy=True)
    target_indices = [bins.columns.index(target_col) for target_col in target_columns]

    original_mi_results = mi_algorithms_ranking[0](arr, target_indices)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Start randomly shuffling targets, and computing MIs of original features with such shuffled targets.
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Permutation testing...")

    # How many times should we evaluate permuted MIs to have a baseline?

    feature_columns = cs.expand_selector(bins, cs.all() - cs.by_name(target_columns))

    expected_evaluations_num = 0
    if permuted_mutual_informations:
        expected_evaluations_num += len(permuted_mutual_informations[target_columns[0]])
    expected_evaluations_num += min_randomized_permutations * len(feature_columns)

    num_randomized_permutations = min_randomized_permutations
    if expected_evaluations_num < min_permuted_mi_evaluations:
        num_randomized_permutations += int(np.ceil((min_permuted_mi_evaluations - expected_evaluations_num) / len(feature_columns)))

    all_permuted_mis = defaultdict(list)
    current_permuted_mis = defaultdict(list)

    if permuted_mutual_informations:
        for target_name, permuted_mis in permuted_mutual_informations.items():
            all_permuted_mis[target_name].append(permuted_mis)

    # Actual permutations

    for permutation_id in tqdmu(range(num_randomized_permutations), desc="Permutation", leave=leave_progressbar):

        for idx in target_indices:
            np.random.shuffle(arr[:, idx])

        permuted_mi_results = mi_algorithms_ranking[0](arr, target_indices)

        for target_idx, target_col_idx in enumerate(target_indices):
            target_name = target_columns[target_idx]
            target_mis = permuted_mi_results[target_idx, :]  # for current features DM
            current_permuted_mis[target_name].append(target_mis)
            all_permuted_mis[target_name].append(np.delete(target_mis, target_col_idx))  # for DM at next steps in future

        if max_runtime_mins and not ran_out_of_time:
            delta = timer() - start_time
            ran_out_of_time = delta > max_runtime_mins * 60
            if ran_out_of_time:
                if verbose:
                    logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                break

    # ----------------------------------------------------------------------------------------------------------------------------
    # Decide what features have no influence & report them.
    # ----------------------------------------------------------------------------------------------------------------------------

    features_usefulness = np.zeros(bins.shape[1], dtype=np.int32)

    for target_idx, target_col_idx in enumerate(target_indices):
        target_name = target_columns[target_idx]

        all_permuted_mis[target_name] = np.hstack(all_permuted_mis[target_name])
        current_permuted_mis[target_name] = np.vstack(current_permuted_mis[target_name])

        if not permuted_max_mi_quantile:
            baseline_mi = all_permuted_mis[target_name].max() * min_mi_prevalence
        else:
            baseline_mi = np.quantile(all_permuted_mis[target_name], permuted_max_mi_quantile) * min_mi_prevalence

        target_features_usefulness = np.zeros(bins.shape[1], dtype=np.int32)
        # test #1: original MI must be above highest permuted MI for this feature
        permuted_prevalence = (current_permuted_mis[target_name] >= original_mi_results[target_idx]).sum(axis=0)
        passed_permutation = ((permuted_prevalence / current_permuted_mis[target_name].shape[0]) <= max_permuted_prevalence_percent).astype(np.int32)
        target_features_usefulness = target_features_usefulness + passed_permutation

        # test #2: original MI must be significanly above highest permuted MI of all features (for this target) seen so far
        passed_baseline = (original_mi_results[target_idx] > baseline_mi).astype(np.int32)
        target_features_usefulness = target_features_usefulness + passed_baseline

        if verbose > 1:
            logger.info(
                f"Target={target_name}, baseline_mi={baseline_mi:.7f}, baseline_n_passed={passed_baseline.sum():_}, permutation_n_passed={passed_permutation.sum():_}"
            )

        features_usefulness += (target_features_usefulness >= 2).astype(np.int32)  # both tests must be passed

    for feature_idx, feature_name in enumerate(bins.columns):
        if features_usefulness[feature_idx] == 0 and feature_idx not in target_indices:
            columns_to_drop.append(feature_name)

    if columns_to_drop:
        if verbose:
            logger.warning(
                f"Found {len(columns_to_drop):_} columns with no direct impact on any target: {textwrap.shorten(', '.join(columns_to_drop), width=max_log_text_width)}"
            )

    return columns_to_drop, original_mi_results, all_permuted_mis, mi_algorithms_ranking


def run_efs(
    df: pl.DataFrame,
    target_columns: list,
    exclude_columns: list,
    permuted_mutual_informations: dict,
    binned_targets: pl.DataFrame,
    mi_algorithms_ranking: list,
    binning_params: dict,
    efs_params: dict,
    use_mis: bool = True,
) -> tuple:

    features_mis = None
    clean_ram()

    bins, binned_targets, public_clips, columns_to_drop, stats = bin_numerical_columns(
        df=df, target_columns=target_columns, binned_targets=binned_targets, exclude_columns=exclude_columns, **binning_params
    )
    if columns_to_drop:
        df = df.drop(columns_to_drop)

    if use_mis:
        clean_ram()
        columns_to_drop, mutual_informations, permuted_mutual_informations, mi_algorithms_ranking = estimate_features_relevancy(
            bins=bins,
            target_columns=target_columns,
            mi_algorithms_ranking=mi_algorithms_ranking,
            permuted_mutual_informations=permuted_mutual_informations,
            **efs_params,
        )

        features_mis = pd.DataFrame({target_columns[col]: mutual_informations[col, :] for col in range(len(target_columns))})
        features_mis["feature"] = bins.columns
        features_mis = features_mis.sort_values(target_columns[0], ascending=False)

        if columns_to_drop:
            df = df.drop(columns_to_drop)

    exclude_columns.update(set(bins.columns))

    clean_ram()

    return (df, exclude_columns, permuted_mutual_informations, binned_targets, mi_algorithms_ranking, features_mis)
