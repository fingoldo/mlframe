"""General-purpose feature-selection helpers: MI-algorithm benchmarking and exhaustive feature search (EFS)."""
from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


import numpy as np, pandas as pd
import polars as pl, polars.selectors as cs

import textwrap
from timeit import default_timer as timer
from collections import defaultdict

from pyutilz.system import clean_ram, tqdmu
from pyutilz.polarslib import bin_numerical_columns
from pyutilz.benchmarking import benchmark_algos_by_runtime

from mlframe.feature_selection.mi import grok_compute_mutual_information, chatgpt_compute_mutual_information, deepseek_compute_mutual_information

# Statistically-calibrated MI null. ``analytic_mi_null`` returns the Miller-Madow plug-in bias floor
# ``(Bx-1)(By-1)/(2N)`` (the permutation-null MEAN) plus a chi-square G-test p-value; it is the same
# machinery the modern filters/mrmr path uses. Reused here rather than reimplemented so the legacy
# ``estimate_features_relevancy`` selection driver is calibrated to a nominal significance level with a
# genuine multiple-comparison (Benjamini-Hochberg) correction instead of ad-hoc raw-MI exceedances.
from mlframe.feature_selection.filters import analytic_mi_null_batch


def _occupied_bins(codes: np.ndarray) -> int:
    """Number of distinct non-negative integer bin codes actually occupied in ``codes``.

    This is ``Bx`` / ``By`` in the Miller-Madow bias term ``(Bx-1)(By-1)/(2N)`` and the G-test degrees
    of freedom ``(Bx-1)(By-1)``. Counting OCCUPIED (not nominal) bins is what makes the bias floor and
    the chi-square df match the actual contingency table, exactly as ``filters/_analytic_mi_null`` does.
    """
    c = np.asarray(codes)
    if c.size == 0:
        return 0
    c = c[c >= 0]
    if c.size == 0:
        return 0
    return int(np.unique(c).size)


def _benjamini_hochberg_reject(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg (1995) FDR-controlling rejection mask.

    Returns a boolean mask, ``True`` where the null is REJECTED (feature deemed relevant) while
    controlling the false-discovery rate at ``alpha`` across all tested features. This is the
    multiple-comparison control the legacy per-feature ``>=`` exceedance lacked entirely: with hundreds
    of independent noise features the raw exceedance over-selects at ~``max_permuted_prevalence_percent``
    per feature, whereas BH bounds the expected fraction of false positives among the selected set.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passing = ranked <= thresholds
    if not passing.any():
        rejected = np.zeros(m, dtype=bool)
        return rejected
    cutoff_idx = int(np.where(passing)[0].max())
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: cutoff_idx + 1]] = True
    return rejected

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def benchmark_mi_algos(base_mi_algos: list, verbose: int = 0, seed: int = 42) -> list:
    """Time each candidate MI implementation on a fixed synthetic workload and return them ranked fastest-first."""
    target_indices = np.array([0, 10, 20], dtype=np.int64)

    # Deterministic per-call RNG. ``np.random.randint`` on the global legacy RNG made benchmark-based MI-implementation selection non-deterministic across processes, breaking reproducibility tests that expect identical column selection.
    _rng = np.random.default_rng(seed)

    # prewarm
    arr = _rng.integers(0, 15, size=(10, 200), dtype=np.int8)
    for func in base_mi_algos:
        _ = func(data=arr, target_indices=target_indices)

    # main
    arr = _rng.integers(0, 15, size=(1_000_000, 200), dtype=np.int8)
    base_mi_algos, _durations = benchmark_algos_by_runtime(
        implementations=base_mi_algos, algo_name="MI", n_reps=2, verbose=verbose, data=arr, target_indices=target_indices
    )

    return base_mi_algos


def estimate_features_relevancy(
    # data
    bins: pl.DataFrame,
    target_columns: list,
    # precomputed info
    mi_algorithms_ranking: list | None = None,  # ltr
    benchmark_mi_algorithms: bool = True,
    permuted_mutual_informations: dict | None = None,
    # working params
    min_mi_prevalence: float = 2,  # 10 is too high for a weak target
    permuted_max_mi_quantile: float | None = None,
    min_permuted_mi_evaluations: int = 500,
    min_randomized_permutations: int = 1,
    max_permuted_prevalence_percent: float = 0.05,
    # multiple-comparison control (statistical calibration of the relevancy test)
    fdr_alpha: float = 0.05,
    # stopping criteria
    max_runtime_mins: float | None = None,
    # rng
    random_state: int | None = 42,
    # style
    leave_progressbar: bool = False,
    max_log_text_width: int = 300,
    verbose: int = 1,
):
    """Computes relevancy of all features to the targets, using integer bins computed at previous step.
    Suggests for droppping columns that have no firm impact on any of the targets.

    A feature is deemed to impact the target only if it passes ALL THREE of the following. The observed
    plug-in MI is first BIAS-CORRECTED by subtracting the Miller-Madow floor ``(Bx-1)(By-1)/(2N)`` (the
    permutation-null mean), so raw-MI positive bias no longer counts as signal:
        1. its bias-corrected MI exceeds the permuted-target MI in (1-max_permuted_prevalence_percent)
           (default ALL) permutations tried;
        2. its bias-corrected MI is at least ``min_mi_prevalence`` times higher than the
           ``permuted_max_mi_quantile`` (default MAXIMUM) permuted MI across all features/permutations;
        3. it survives a Benjamini-Hochberg FDR correction (at ``fdr_alpha``) over the per-feature
           analytic G-test p-values ``chi2.sf(2*N*MI, (Bx-1)(By-1))`` -- the multiple-comparison control
           that keeps many independent noise features from over-selecting at scale.

    The bias floor and p-values reuse ``feature_selection.filters._analytic_mi_null`` (the same
    calibrated machinery as the modern filters/mrmr path).

    Either min_randomized_permutations or max_runtime_mins should be specified.

    Reports:

    """
    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    start_time = timer()
    ran_out_of_time = False

    columns_to_drop = []

    # Wave 31 (2026-05-20): assert -> ValueError.
    if min_randomized_permutations < 1:
        raise ValueError(f"min_randomized_permutations must be >= 1; got {min_randomized_permutations!r}.")

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

    # ``to_numpy(allow_copy=True)`` may return a buffer that aliases the polars frame's columns, so the permutation loop below must restore every target column it shuffles (see the mutate-and-restore there) rather than leave the caller's data scrambled.
    arr = bins.to_numpy(allow_copy=True)
    target_indices = [bins.columns.index(target_col) for target_col in target_columns]
    # Per-call deterministic RNG so two parallel suite calls (or parent-process pre-seeding) cannot interleave shuffles; ``np.random.shuffle`` on the legacy global RNG was the source of non-deterministic baselines under joblib + fork workers.
    _rng = np.random.default_rng(random_state)

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

    all_permuted_mis: dict = defaultdict(list)
    current_permuted_mis: dict = defaultdict(list)

    if permuted_mutual_informations:
        for target_name, permuted_mis in permuted_mutual_informations.items():
            all_permuted_mis[target_name].append(permuted_mis)

    # Actual permutations

    for _permutation_id in tqdmu(range(num_randomized_permutations), desc="Permutation", leave=leave_progressbar):

        # ``arr`` aliases the caller's ``bins.to_numpy(allow_copy=True)``; shuffling ``arr[:, idx]``
        # (a view under basic indexing) in place would leave the caller's target columns permanently
        # scrambled. Save each target column, permute a copy in place for the MI call, restore after.
        saved_targets = {idx: arr[:, idx].copy() for idx in target_indices}
        try:
            for idx in target_indices:
                shuffled = saved_targets[idx].copy()
                _rng.shuffle(shuffled)
                arr[:, idx] = shuffled

            permuted_mi_results = mi_algorithms_ranking[0](arr, target_indices)
        finally:
            for idx, original in saved_targets.items():
                arr[:, idx] = original

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
                    logger.info("max_runtime_mins=%s reached.", f"{max_runtime_mins:_.1f}")
                break

    # ----------------------------------------------------------------------------------------------------------------------------
    # Decide what features have no influence & report them.
    # ----------------------------------------------------------------------------------------------------------------------------

    features_usefulness = np.zeros(bins.shape[1], dtype=np.int32)

    n_samples = arr.shape[0]
    # Per-column occupied-bin counts drive both the Miller-Madow bias floor and the G-test df. Computed
    # once here (not per target) since the codes are the same across targets.
    occupied_bins_per_col = np.array([_occupied_bins(arr[:, j]) for j in range(bins.shape[1])], dtype=np.int64)

    for target_idx, target_col_idx in enumerate(target_indices):
        target_name = target_columns[target_idx]

        if len(all_permuted_mis[target_name]) == 0 or len(current_permuted_mis[target_name]) == 0:
            if verbose:
                logger.info("Skipping target=%s: no permuted MI samples collected (likely hit max_runtime_mins before the first permutation).", target_name)
            continue

        all_permuted_mis[target_name] = np.hstack(all_permuted_mis[target_name])
        current_permuted_mis[target_name] = np.vstack(current_permuted_mis[target_name])

        # Guard against the edge case where hstack yields a size-0 array
        # (e.g. single-column frames where np.delete strips the only entry).
        if all_permuted_mis[target_name].size == 0 or current_permuted_mis[target_name].size == 0:
            if verbose:
                logger.info("Skipping target=%s: permuted MI array is empty after stacking.", target_name)
            continue

        by = int(occupied_bins_per_col[target_col_idx])  # occupied target bins

        # ----------------------------------------------------------------------------------------------
        # Bias-correct the observed MI before ANY comparison. Raw plug-in MI is positively biased by the
        # Miller-Madow floor ``(Bx-1)(By-1)/(2N)`` even for independent variables; subtracting the same
        # per-feature floor that ``analytic_mi_null`` reports puts the observed statistic on the null's
        # own scale so the raw-MI exceedance tests below are no longer fooled by the bias mean.
        # ----------------------------------------------------------------------------------------------
        raw_mi_row = np.asarray(original_mi_results[target_idx], dtype=np.float64)
        # Vectorized over ALL candidate columns in one gammaincc (was a per-column analytic_mi_null loop -> 138k scalar
        # chi2.sf calls, the #1 MRMR-screen hotspot). Bit-identical to the scalar loop.
        null_mean_row, p_values = analytic_mi_null_batch(raw_mi_row, n_samples, occupied_bins_per_col[: bins.shape[1]], by)
        debiased_mi = np.where(np.isfinite(raw_mi_row), raw_mi_row - null_mean_row, -np.inf)

        if not permuted_max_mi_quantile:
            # Wave 21 P1: nanmax so NaN-MI permutations don't poison the max.
            baseline_mi = np.nanmax(all_permuted_mis[target_name]) * min_mi_prevalence
        else:
            # Wave 21 P1: nanquantile so degenerate target/feature pairs
            # (which the MI estimator emits NaN for) don't make baseline_mi
            # NaN -> downstream `passed_permutation` comparison would
            # evaluate inconsistently (NaN <= x is always False).
            baseline_mi = np.nanquantile(all_permuted_mis[target_name], permuted_max_mi_quantile) * min_mi_prevalence

        target_features_usefulness = np.zeros(bins.shape[1], dtype=np.int32)
        # test #1: BIAS-CORRECTED original MI must be above the highest permuted MI for this feature.
        # The exceedance uses ``>=`` (a permuted MI that ties the observed counts as a failure); on discrete / low-cardinality data ties are frequent, so this is mildly
        # conservative (it can demote a genuinely-weak feature whose null occasionally ties it). Tolerated here because the prevalence is rate-thresholded against
        # ``max_permuted_prevalence_percent`` rather than requiring zero exceedances, and ``nanmax`` / ``nanquantile`` already guard the NaN-MI degenerate-pair case below.
        permuted_prevalence = (current_permuted_mis[target_name] >= debiased_mi).sum(axis=0)
        passed_permutation = ((permuted_prevalence / current_permuted_mis[target_name].shape[0]) <= max_permuted_prevalence_percent).astype(np.int32)
        target_features_usefulness = target_features_usefulness + passed_permutation

        # test #2: bias-corrected original MI must be significantly above the highest permuted MI of all features (for this target) seen so far
        passed_baseline = (debiased_mi > baseline_mi).astype(np.int32)
        target_features_usefulness = target_features_usefulness + passed_baseline

        # test #3: Benjamini-Hochberg FDR control across the per-feature analytic (G-test) p-values. This
        # is the multiple-comparison control the ad-hoc per-feature exceedance lacked: with many noise
        # features the raw ``>=`` test admits ~``max_permuted_prevalence_percent`` false positives PER
        # feature (over-selection at scale), whereas BH bounds the expected false-discovery FRACTION of
        # the selected set at ``fdr_alpha``. Targets/degenerate columns get p=1.0 above and never pass.
        passed_fdr = _benjamini_hochberg_reject(p_values, fdr_alpha).astype(np.int32)
        target_features_usefulness = target_features_usefulness + passed_fdr

        if verbose > 1:
            logger.info(
                "Target=%s, baseline_mi=%.7f, baseline_n_passed=%s, permutation_n_passed=%s, fdr_n_passed=%s",
                target_name, baseline_mi, f"{passed_baseline.sum():_}", f"{passed_permutation.sum():_}", f"{passed_fdr.sum():_}",
            )

        features_usefulness += (target_features_usefulness >= 3).astype(np.int32)  # all three tests must be passed

    for feature_idx, feature_name in enumerate(bins.columns):
        if features_usefulness[feature_idx] == 0 and feature_idx not in target_indices:
            columns_to_drop.append(feature_name)

    if columns_to_drop:
        if verbose:
            logger.warning(
                "Found %s columns with no direct impact on any target: %s",
                f"{len(columns_to_drop):_}",
                textwrap.shorten(", ".join(columns_to_drop), width=max_log_text_width),
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
    """Run exhaustive feature search: drop columns with no permutation-significant MI to any target, then rank the survivors."""
    features_mis = None
    clean_ram()

    bins, binned_targets, _public_clips, columns_to_drop, _stats = bin_numerical_columns(
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

        # Wave 54 (2026-05-20): refuse duplicate target names -- the prior dict-comp
        # silently dropped MI rows when the caller listed the same target twice
        # (e.g. multi-target classification+regression view of the same column).
        if len(set(target_columns)) != len(target_columns):
            from collections import Counter as _Counter
            _dupes = [_t for _t, _n in _Counter(target_columns).items() if _n > 1]
            raise ValueError(f"target_columns has {len(_dupes)} duplicate(s) ({_dupes[:5]}); " "deduplicate to avoid silently dropping MI rows.")
        features_mis = pd.DataFrame({target_columns[col]: mutual_informations[col, :] for col in range(len(target_columns))})
        features_mis["feature"] = bins.columns
        features_mis = features_mis.sort_values(target_columns[0], ascending=False)

        if columns_to_drop:
            df = df.drop(columns_to_drop)

    # Build the augmented exclusion set locally instead of ``.update()``-ing the caller's argument: ``exclude_columns``
    # is annotated ``list`` (no ``.update``) and mutating the caller's collection in place is a surprising side effect.
    # The augmented set is handed back via the return value.
    exclude_columns_out = set(exclude_columns) | set(bins.columns)

    clean_ram()

    return (df, exclude_columns_out, permuted_mutual_informations, binned_targets, mi_algorithms_ranking, features_mis)
