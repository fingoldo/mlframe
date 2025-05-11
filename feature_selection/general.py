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
from pyutilz.benchmarking import benchmark_algos_by_runtime

from mlframe.feature_selection.mi import grok_compute_mutual_information, chatgpt_compute_mutual_information, deepseek_compute_mutual_information

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def find_unrelated_features(
    df: pl.DataFrame,
    target_columns_prefix: str = "target_",
    binned_targets: pl.DataFrame = None,
    clean_targets: bool = False,
    num_bins: int = 10,
    exclude_columns: list = [],
    min_nuniques_to_clip: int = 50,
    tukey_fences_multiplier: float = 3.0,
    max_log_text_width: int = 300,
    verbose: int = 1,
) -> pl.DataFrame:
    """DropFinds features that have no direct relationship to at least one of the targets.
    Mutual Information (MI) is used to estimate presence of a relationship.

    Columns from exclude_columns are exempt of this check, so put here what you arelady have checked is relevant.

    Categorical features support rare categories merging.

    Numerical features:
        often can have outliers, so we provide an option of clipping outliers.

        First stats like min, max, quantiles are computed for each numerical feature.
        Features with no change (min==max) are dropped.
        For the rest, Tukey fences are computed, outliers are reported & clipped (windzorized).

        Next, each num column is binned into num_bins bins using borders=np.linspace(min_val, max_val, num_bins + 1),
            where min_val and max_val account for clipping.

        Marginal frequencies, and then entropies are computed for each column.

        for each of targets:
            Marginal frequencies, and then joint entropies & MIs are computed for each combination of explanatory column and target.
            Columns with zero MI are considered irrelevant to the target.
            for each of n_reps:
                columns bins are randomly permuted
                Marginal frequencies, and then joint entropies & shuffled MIs are computed for each combination of explanatory column and target.
            as long as shuffled MI for any column exceeds original MI, column is considered irrelevant to the target.

        Columns not relevant to any of the targets are reported & dropped.

    """
    return


def estimate_features_relevancy(
    # data
    bins: pl.DataFrame,
    target_columns_prefix: str = "target_",
    entropies: dict = None,
    # precomputed info
    mi_algorithms_rankng: list = None,  # ltr
    benchmark_mi_algorithms: bool = True,
    permuted_mutual_informations: dict = None,
    # working params
    min_mi_prevalence: float = 10,
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
    """Computes relevancy of all features to the targets,
    using the bins computed at previous step.
    Suggest for droppping columns that have no firm impact on any of the targets.

    We think that a feature has impact on target if:
        it's MI with original target is higher than MI with permuted target in (1-max_permuted_prevalence_percent)
            (default ALL) permutations we tried;
        it's MI with original target is at least min_mi_prevalence times higher than permuted_max_mi_quantile
            (default MAXIMUM) MI with that target (permuted) across all features and permutaions we tried.

        MAXIMUM can be replaced with some high quantile, like 0.99.

    Either min_randomized_permutations or max_runtime_mins should be specified.

    Reports:

    """

    start_time = timer()
    ran_out_of_time = False

    columns_to_drop = []

    if entropies is None:
        entropies = {}

    target_cols = cs.expand_selector(bins, cs.starts_with(target_columns_prefix))

    # ----------------------------------------------------------------------------------------------------------------------------
    # What MI implementation is the fastest for current machine?
    # ----------------------------------------------------------------------------------------------------------------------------

    if not mi_algorithms_rankng:
        base_mi_algos = [chatgpt_compute_mutual_information, grok_compute_mutual_information, deepseek_compute_mutual_information]
        if benchmark_mi_algorithms:

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

        else:
            benchmark_mi_algorithms = base_mi_algos

    # ----------------------------------------------------------------------------------------------------------------------------
    # For each of the targets, compute joint freqs and then MI for each of the "normal" columns:
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Computing original MIs...")

    arr = bins.to_numpy(allow_copy=True)
    original_mi_results = base_mi_algos[0](arr, target_indices)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Start randomly shuffling targets, and computing Mis of original features with such shuffled targets.
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Permutation testing...")

    mutual_informations = {}

    # How many times should we evaluate permuted MIs to have a baseline?

    permuted_mutual_informations: dict = None
    min_permuted_mi_evaluations: int = 1000

    for permutation_id in range(min_randomized_permutations):

        for idx in target_indices:
            np.random.shuffle(arr[:, idx])

        permuted_mi_results = grok_compute_mutual_information(arr, target_indices)

    mi_results = chatgpt_compute_mutual_information(bins, target_indices)

    for target_id, target_col in enumerate(tqdmu(target_cols, desc="features MIs with targets", leave=leave_progressbar)):
        for i in tqdmu(range(min_randomized_permutations), desc="bootstrap", leave=leave_progressbar):

            # Shuffle target, if it's time to
            if i > 0:
                bins = bins.with_columns(pl.col(target_col).shuffle())

            mi_results = {}
            cols_to_compute_mis = []

            # Who needs computing, actually?

            for col in tqdmu(cs.expand_selector(bins, cs.all()), desc="var", leave=leave_progressbar):
                if col.startswith(target_columns_prefix):
                    continue
                if (
                    i == 0 or mutual_informations[(col, target_col)] > permuted_mutual_informations[target_col] * min_mi_prevalence
                ):  # only compute when it makes sense
                    cols_to_compute_mis.append(col)

            # Decision making part

            for col, mi in mi_results.items():
                if i == 0:
                    # save original MI of col vs target
                    mutual_informations[(col, target_col)] = mi
                else:
                    # if permuted MI is same good or better than original - zero out original. Feature to be dropped.
                    if mi >= mutual_informations[(col, target_col)]:
                        mutual_informations[(col, target_col)] = 0
                        if verbose > 1:
                            logger.warning(f"After permutation, MI of var={col} with target {target_col} remained high: {mi}")

            if max_runtime_mins and not ran_out_of_time:
                delta = timer() - start_time
                ran_out_of_time = delta > max_runtime_mins * 60
                if ran_out_of_time:
                    if verbose:
                        logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                    break

    # ----------------------------------------------------------------------------------------------------------------------------
    # Sum up MIs per column (over targets), decide what features have no influence, report & drop them.
    # ----------------------------------------------------------------------------------------------------------------------------

    cols_total_mis = defaultdict(int)

    for (col, target_col), mi in mutual_informations.items():
        if mi > permuted_mutual_informations[target_col] * min_mi_prevalence:
            cols_total_mis[col] += 1
        else:
            cols_total_mis[col] += 0

    dead_columns = []
    for col, total_mi in cols_total_mis.items():
        if total_mi == 0:  # not related to any target
            dead_columns.append(col)

    if dead_columns:
        if verbose:
            logger.warning(
                f"Dropping {len(dead_columns):_} columns with no direct impact on any target: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width)}"
            )
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    if verbose > 1:
        logger.info(f"Done. {len(columns_to_drop):_} columns_to_drop: {textwrap.shorten(', '.join(columns_to_drop), width=max_log_text_width)}.")

    del bins
    clean_ram()

    return columns_to_drop, entropies, permuted_mutual_informations, mutual_informations


def run_efs(df, exclude_columns, entropies, permuted_mutual_informations, binned_targets, efs_params) -> tuple:
    binned_targets, public_clips, columns_to_drop, entropies, permuted_mutual_informations, mutual_informations = find_unrelated_features(
        df, entropies=entropies, permuted_mutual_informations=permuted_mutual_informations, binned_targets=binned_targets, **efs_params
    )

    df = df.drop(columns_to_drop)
    exclude_columns.update(set(df.columns))
    features_mis = pd.Series(mutual_informations).sort_values(ascending=False)

    return df, exclude_columns, entropies, permuted_mutual_informations, binned_targets, features_mis
