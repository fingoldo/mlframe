"""Feature selection within ML pipelines."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

ensure_installed("numpy pandas")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import pandas as pd, numpy as np


from pyutilz.system import tqdmu
from mlframe.boruta_shap import BorutaShap


from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from numba import njit
import math


# ----------------------------------------------------------------------------------------------------------------------------
# Old code
# ----------------------------------------------------------------------------------------------------------------------------


def find_impactful_features(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    feature_selector: object = None,
    model: object = None,
    importance_measure: str = "shap",
    classification: bool = False,
    n_trials: int = 150,
    normalize: bool = True,
    train_or_test="train",
    verbose: bool = True,
    fit_params: dict = {},
) -> dict:
    """
    Create a dict of inputs impacting every and all target (multitarget supported).
    Wrapped models are not supported (like TransformedTargetRegressor).
    """

    if verbose:
        logger.info("Starting impact analysis for %s row(s), %s feature(s), %s target(s)", X.shape[0], X.shape[1], Y.shape[1])

    if not feature_selector:
        feature_selector = BorutaShap(
            model=model,
            importance_measure=importance_measure,
            classification=classification,
            n_trials=n_trials,
            normalize=normalize,
            verbose=False,
            train_or_test=train_or_test,
            fit_params=fit_params,
        )

    if False:  # when multioutput is not supported
        max_targets = 0
        res = {"accepted": {}, "tentative": {}}
        wholeset_accepted, wholeset_tentative = [], []

        for var in tqdmu(range(Y.shape[1]), desc="target #"):
            if max_targets:
                if var >= max_targets:
                    break
            feature_selector.fit(X=X, y=Y.iloc[:, var], n_trials=n_trials, normalize=normalize, verbose=False, train_or_test=train_or_test)

            res["accepted"][var] = feature_selector.accepted
            res["tentative"][var] = feature_selector.tentative

            if verbose:
                logger.info(
                    "%s feature(s) found impactful on target %s: %s, %s tentative: %s",
                    len(feature_selector.accepted),
                    var,
                    sorted(feature_selector.accepted),
                    len(feature_selector.tentative),
                    sorted(feature_selector.tentative),
                )

            wholeset_accepted.extend(feature_selector.accepted)
            wholeset_tentative.extend(feature_selector.tentative)

        res["wholeset"] = {"accepted": set(wholeset_accepted), "tentative": set(wholeset_tentative)}
        res["mostvoted_accepted"] = Counter(wholeset_accepted)
        res["mostvoted_tentative"] = Counter(wholeset_tentative)
    else:
        feature_selector.fit(X=X, y=Y)
        res = {"accepted": sorted(feature_selector.accepted), "tentative": sorted(feature_selector.tentative)}
        if verbose:
            logger.info(res)
    return res


# ----------------------------------------------------------------------------------------------------------------------------
# Multidimensial vectors
# ----------------------------------------------------------------------------------------------------------------------------


@njit()
def merge_vars(
    data: np.ndarray,
    vars_indices: Sequence,
    var_is_nominal: Sequence,
    var_nbins: Sequence,
    dtype=np.int64,
    min_occupancy: int = None,
    verbose: bool = False,
) -> np.ndarray:
    """Melts multiple vectors (partitioned into bins or categories) into one-dimensional one.
    Data columns are assumed to be ordinal-encoded already (by KBinsDiscretizer or similar).
    Bins, scarcely populated in higher dimension, may be merged with nearest neighbours (in eucl. distance).

    Process truly numerical vars first? To be able to distribute scarcely populated bins more meaningfully, according to the distances.
    What order is best to use in regard of number of categories and variance of their population density? Big-small,big-small, etc? Starting from most (un)evenly distributed?
    """
    current_nclasses = 1
    final_classes = np.zeros(len(data), dtype=dtype)
    for var_number, var_index in enumerate(vars_indices):

        expected_nclasses = current_nclasses * var_nbins[var_index]
        freqs = np.zeros(expected_nclasses, dtype=dtype)
        values = data[:, var_index].astype(dtype)
        if verbose:
            print(f"var={var_index}, classes from {values.min()} to {values.max()}")
        for sample_row, sample_class in enumerate(values):
            newclass = final_classes[sample_row] + sample_class * current_nclasses
            freqs[newclass] += 1
            final_classes[sample_row] = newclass

        # clean zeros
        nzeros = 0
        lookup_table = np.empty(expected_nclasses, dtype=dtype)
        for oldclass, npoints in enumerate(freqs):
            if npoints == 0:
                nzeros += 1
            else:
                pass
                # points from low-populated regions can be
                # 1) distributed into closest N regions proportional to distances to them
                # 2) ALL placed into some separate OUTLIERS region
                # 3) left as is

                # if npoints<min_occupancy: ...

            lookup_table[oldclass] = oldclass - nzeros

        if nzeros:
            if verbose:
                print(
                    f"skipped {nzeros} cells out of {expected_nclasses}, classes from {final_classes.min()} to {final_classes.max()}, lookup_table={lookup_table}"
                )

            for sample_row, old_class in enumerate(final_classes):
                final_classes[sample_row] = lookup_table[old_class]
            if var_number == len(vars_indices) - 1:
                freqs = freqs[freqs > 0]
        current_nclasses = expected_nclasses - nzeros
    return final_classes, freqs / len(data)


# ----------------------------------------------------------------------------------------------------------------------------
# Information-theory measures
# ----------------------------------------------------------------------------------------------------------------------------


@njit()
def entropy(freqs: np.ndarray, min_occupancy: float = 0) -> float:
    if min_occupancy:
        freqs = freqs[freqs >= min_occupancy]

    return -(np.log(freqs) * freqs).sum()


@njit()
def mi(data, x: np.ndarray, y: np.ndarray, var_nbins: np.ndarray, verbose: bool = False) -> float:
    """Computes Mutual Information of X, Y via entropy calculations."""

    classes_x, freqs_x = merge_vars(data=data, vars_indices=x, var_is_nominal=None, var_nbins=var_nbins, verbose=verbose)
    entropy_x = entropy(freqs=freqs_x)
    if verbose:
        print(f"entropy_x={entropy_x}, nclasses_x={len(freqs_x)} ({classes_x.min()} to {classes_x.max()})")

    _, freqs_y = merge_vars(data=data, vars_indices=y, var_is_nominal=None, var_nbins=var_nbins, verbose=verbose)
    entropy_y = entropy(freqs=freqs_y)
    if verbose:
        print(f"entropy_y={entropy_y}, nclasses_y={len(freqs_y)}")

    classes_xy, freqs_xy = merge_vars(data=data, vars_indices=set(x) | set(y), var_is_nominal=None, var_nbins=var_nbins, verbose=verbose)
    entropy_xy = entropy(freqs=freqs_xy)
    if verbose:
        print(f"entropy_xy={entropy_xy}, nclasses_x={len(freqs_xy)} ({classes_xy.min()} to {classes_xy.max()})")

    return entropy_x + entropy_y - entropy_xy


@njit()
def conditional_mi(
    data: np.ndarray,
    x: tuple,
    y: tuple,
    z: tuple,
    var_is_nominal: Sequence,
    var_nbins: Sequence,
    entropy_z: float = -1.0,
    entropy_xz: float = -1.0,
    entropy_yz: float = -1.0,
    entropy_cache: dict = None,
    classes_cache: dict = None,
) -> float:
    """
    Conditional Mutual Information about Y by X given Z = I (X ;Y | Z ) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
    When y is constant, and both X(candidates) and Z (veteranes) repeat a lot, there's room to optimize.
    Also when parts of X repeat a lot (2, 3 way interactions). Z and Y are always 1-dim.
    """

    if entropy_z < 0:
        _, freqs_z = merge_vars(data=data, vars_indices=z, var_is_nominal=None, var_nbins=var_nbins)  # always 1-dim
        entropy_z = entropy(freqs=freqs_z)

    if entropy_xz < 0:
        _, freqs_xz = merge_vars(data=data, vars_indices=[*x, *z], var_is_nominal=None, var_nbins=var_nbins)
        entropy_xz = entropy(freqs=freqs_xz)

    if entropy_yz < 0:
        _, freqs_yz = merge_vars(data=data, vars_indices=[*y, *z], var_is_nominal=None, var_nbins=var_nbins)  # always 2-dim
        entropy_yz = entropy(freqs=freqs_yz)

    _, freqs_xyz = merge_vars(
        data=data, vars_indices=[*y, *z, *x], var_is_nominal=None, var_nbins=var_nbins
    )  # upper classes of [*y, *z] can serve here. (2+x)-dim, ie 3 to 5 dim.

    return entropy_xz + entropy_yz - entropy_z - entropy(freqs=freqs_xyz)


@njit()
def compute_mi_from_classes(
    classes_x,
    freqs_x,
    classes_y,
    freqs_y,
    dtype=np.int64,
) -> float:
    joint_counts = np.zeros((len(freqs_x), len(freqs_y)), dtype=dtype)
    for i, j in zip(classes_x, classes_y):
        joint_counts[i, j] += 1

    joint_freqs = joint_counts / len(classes_x)

    total = 0.0
    nzeros = 0
    for i in range(len(freqs_x)):
        prob_x = freqs_x[i]
        for j in range(len(freqs_y)):
            prob_y = freqs_y[j]
            jf = joint_freqs[i, j]
            if jf:
                total += jf * math.log(jf / (prob_x * prob_y))
            else:
                nzeros += 1
    return total


@njit()
def mi_direct(
    data,
    x: np.ndarray,
    y: np.ndarray,
    var_nbins: np.ndarray,
    min_occupancy: int = None,
    dtype=np.int64,
    verbose: bool = False,
    min_nonzero_confidence: float = 0.95,
    npermutations: int = 10,
) -> tuple:

    classes_x, freqs_x = merge_vars(data=data, vars_indices=np.array(x, dtype=dtype), var_is_nominal=None, var_nbins=var_nbins)
    classes_y, freqs_y = merge_vars(data=data, vars_indices=np.array(y, dtype=dtype), var_is_nominal=None, var_nbins=var_nbins)
    if verbose:
        print(f"nclasses_x={len(freqs_x)} ({classes_x.min()} to {classes_x.max()}), nclasses_y={len(freqs_y)} ({classes_y.min()} to {classes_y.max()})")

    original_mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)

    if npermutations:
        # inits
        nfailed = 0
        max_failed = int(npermutations * (1 - min_nonzero_confidence))

        # copy data for safe shuffling
        if len(x) > 1:
            x_copy = data[:, np.array(x)].copy()
            x_var_nbins = var_nbins[np.array(x)]
        if len(y) > 1:
            y_copy = data[:, np.array(y)].copy()
            y_var_nbins = var_nbins[np.array(y)]

        for i in range(npermutations):
            if len(x) > 1:
                for idx in range(len(x)):
                    np.random.shuffle(x_copy[:, idx])
                classes_x, freqs_x = merge_vars(data=x_copy, vars_indices=np.arange(len(x)), var_is_nominal=None, var_nbins=x_var_nbins)
            else:
                np.random.shuffle(classes_x)

            if len(y) > 1:
                for idx in range(len(y)):
                    np.random.shuffle(y_copy[:, idx])
                classes_y, freqs_y = merge_vars(data=y_copy, vars_indices=np.arange(len(y)), var_is_nominal=None, var_nbins=y_var_nbins)
            else:
                np.random.shuffle(classes_y)

            mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)
            if mi >= original_mi:
                nfailed += 1
                if nfailed >= max_failed:
                    original_mi = 0.0
                    break
        confidence = 1 - nfailed / (i + 1)
        if verbose:
            print("confidence=", confidence, "nfailed=", nfailed)
    else:
        confidence = 0

    return original_mi, confidence


# ----------------------------------------------------------------------------------------------------------------------------
# Factor screening
# ----------------------------------------------------------------------------------------------------------------------------


def get_candidate_name(candidate_indices: list, cols: list) -> str:
    cand_name = "-".join([cols[el] for el in candidate_indices])
    return cand_name


# @njit()
def screen_predictors(
    data: np.ndarray,
    y: np.ndarray,
    cols: list,
    var_nbins: np.ndarray,
    x: np.ndarray = None,
    min_occupancy: int = None,
    dtype=np.int64,
    max_subset_size: int = 1,
    max_joint_subset_size: int = 3,
    min_nonzero_confidence: float = 0.98,
    npermutations: int = 10_000,
    min_gain: float = 0.00001,
    max_cons_confirmation_failures: int = 10,
    prewarm_npermutations: int = 100,
    verbose: bool = False,
) -> float:
    """Finds best predictors for the target.
    x must be n-x-m array of integers (ordinal encoded)
    Parameters:
        npermutations: when computing every MI, repeat calculations with randomly shuffled indices that many times
        min_nonzero_confidence: if in random permutation tests this or higher % of cases had worse current_gain than original, current_gain value is considered valid, otherwise, it's set to zero.
    Returns:
        1) best set of non-redundant single features influencing the target
        2) subsets of size 2..max_subset_size influencing the target. Sucg subsets will eb candidate for predictors and OtherVarsEncoding.
    """
    if x is None:
        x = set(range(data.shape[1])) - set(y)
    selected_vars = []  # stores just indices. can't use set 'cause the order is important for efficient computing
    selected_interactions_vars = []
    predictors = []  # stores more details.
    assert not set(y).issubset(set(x))
    max_failed = int(npermutations * (1 - min_nonzero_confidence))

    cached_MIs = dict()
    cached_cond_MIs = dict()
    cached_confident_MIs = dict()

    # ---------------------------------------------------------------------------------------------------------------
    # prepare data buffer
    # ---------------------------------------------------------------------------------------------------------------

    data_copy = data.copy()

    for subset_size in tqdmu(range(1, max_subset_size + 1), desc="Subset size"):

        # ---------------------------------------------------------------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------------------------------------------------------------

        if subset_size == 1:
            candidates = [(el,) for el in x]
        else:
            candidates = [tuple(el) for el in combinations(x, subset_size)]  # if all([subel not in selected_vars for subel in el])

        min_gain_reached = False
        nunconfirmed = 0
        partial_gains = {}
        failed_candidates = set()
        added_candidates = set()
        ncons_confirmation_failures = 0
        for _ in tqdmu(range(len(candidates)), leave=False, desc="Candidates"):

            # ---------------------------------------------------------------------------------------------------------------
            # Find candidate X with the highest current_gain given already selected factors
            # ---------------------------------------------------------------------------------------------------------------

            best_gain = 0
            expected_gains = np.zeros(len(candidates), dtype=np.float64)

            while True:  # confirmation loop (by random permutations)
                for cand_idx, X in enumerate(tqdmu(candidates, leave=False, desc="Gain evals")):
                    if (cand_idx in failed_candidates) or (cand_idx in added_candidates):
                        continue
                    if expected_gains[cand_idx]:
                        continue
                    if subset_size > 1:  # disabled for single predictors 'cause Fleuret formula won't detect pairs predictors

                        # ---------------------------------------------------------------------------------------------------------------
                        # Check if any of sub-elements is already selected at this stage
                        # ---------------------------------------------------------------------------------------------------------------

                        skip_cand = False
                        for subel in X:
                            if subel in selected_interactions_vars:
                                skip_cand = True
                                break
                        if skip_cand:
                            continue

                        # ---------------------------------------------------------------------------------------------------------------
                        # Or all selected at the lower stages
                        # ---------------------------------------------------------------------------------------------------------------

                        skip_cand = all([(subel in selected_vars) for subel in X])
                        if skip_cand:
                            continue

                    # ---------------------------------------------------------------------------------------------------------------
                    # Is this candidate any good for target 1-vs-1?
                    # ---------------------------------------------------------------------------------------------------------------

                    if X in cached_MIs:
                        direct_gain = cached_MIs[X]
                    else:
                        direct_gain, _ = mi_direct(data, x=X, y=y, var_nbins=var_nbins, min_nonzero_confidence=1.0, npermutations=prewarm_npermutations)
                        cached_MIs[X] = direct_gain

                    if direct_gain > 0:
                        if selected_vars:

                            # ---------------------------------------------------------------------------------------------------------------
                            # some factors already selected.
                            # best gain from including X is the minimum of I (X ;Y | Z ) over every Z in already selected_vars.
                            # but imaging some variable is correlated to every real predictor plus has random noise. It's real value is zero.
                            # only computing I (X ;Y | Z ) will still leave significant impact. but if we sum I(X,Z) over Zs we'll see it shares
                            # all its knowledge with the rest of factors and has no value by itself. But to see that, we must already have all real factors included in S.
                            # otherwise, such 'connected-to-all' trash variables will dominate the scene. So how to handle them?
                            # Solution is to compute sum(X,Z) not only at the step of adding Z, but to repeat this procedure for all Zs once new X is added.
                            # Maybe some Zs will render useless by adding that new X.
                            # ---------------------------------------------------------------------------------------------------------------

                            if cand_idx in partial_gains:
                                current_gain, last_checked_z = partial_gains[cand_idx]
                            else:
                                current_gain = 1e30
                                last_checked_z = -1

                            # max_joint_subset_size

                            for z_idx, Z in enumerate(selected_vars):

                                if z_idx > last_checked_z:

                                    # ---------------------------------------------------------------------------------------------------------------
                                    # additional_knowledge = I (X ;Y | Z ) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
                                    # I (X,Z) would be entropy_x + entropy_z - entropy_xz. we don't have only H(X) which can be computed before checking Zs
                                    # ---------------------------------------------------------------------------------------------------------------

                                    key = (X, (Z,))
                                    if key in cached_cond_MIs:
                                        additional_knowledge = cached_cond_MIs[key]
                                    else:
                                        additional_knowledge = conditional_mi(data=data, x=X, y=y, z=(Z,), var_is_nominal=None, var_nbins=var_nbins)  # TODO
                                        cached_cond_MIs[key] = additional_knowledge

                                    if additional_knowledge < current_gain:

                                        current_gain = additional_knowledge

                                        if current_gain <= best_gain:

                                            # ---------------------------------------------------------------------------------------------------------------
                                            # no point checking other Zs, 'cause current_gain already won't be better than the best_gain (if best_gain was estimated confidently.)
                                            # ---------------------------------------------------------------------------------------------------------------

                                            partial_gains[cand_idx] = current_gain, z_idx
                                            current_gain = 0
                                            break
                            else:  # nobreak
                                partial_gains[cand_idx] = current_gain, z_idx
                                expected_gains[cand_idx] = current_gain

                        else:
                            # no factors selected yet. current_gain is just direct_gain
                            current_gain = direct_gain
                            expected_gains[cand_idx] = current_gain
                    else:
                        current_gain = 0

                    # ---------------------------------------------------------------------------------------------------------------
                    # save best known candidate to be able to use early stopping
                    # ---------------------------------------------------------------------------------------------------------------

                    if current_gain > best_gain:
                        best_candidate = X
                        best_gain = current_gain
                        if verbose:
                            print(f"\t{get_candidate_name(best_candidate,cols)} is so far the best candidate with best_gain={best_gain:.10f}")
                    else:

                        if False and current_gain > min_gain:
                            if verbose:
                                print(f"\t\t{get_candidate_name(X,cols)} current_gain={current_gain:.10f}")

                # ---------------------------------------------------------------------------------------------------------------
                # now need to confirm best expected gain with a permutation test
                # ---------------------------------------------------------------------------------------------------------------

                if npermutations:
                    cand_confirmed = False
                    any_cand_considered = False
                    for n, next_best_candidate in enumerate(np.argsort(expected_gains)[::-1]):
                        next_best_gain = expected_gains[next_best_candidate]
                        if next_best_gain >= min_gain:  # only can consider here candidates fully checked against every Z
                            any_cand_considered = True
                            if verbose:
                                print("confirming candidate", get_candidate_name(candidates[next_best_candidate], cols), "next_best_gain=", next_best_gain)

                            # ---------------------------------------------------------------------------------------------------------------
                            # Compute confidence by bootstrap
                            # ---------------------------------------------------------------------------------------------------------------

                            if candidates[next_best_candidate] in cached_confident_MIs:
                                bootstrapped_gain, confidence = cached_confident_MIs[candidates[next_best_candidate]]
                            else:
                                bootstrapped_gain, confidence = mi_direct(
                                    data,
                                    x=candidates[next_best_candidate],
                                    y=y,
                                    var_nbins=var_nbins,
                                    min_nonzero_confidence=min_nonzero_confidence,
                                    npermutations=npermutations,
                                )
                                cached_confident_MIs[candidates[next_best_candidate]] = bootstrapped_gain, confidence

                            if bootstrapped_gain > 0 and selected_vars:  # additional check of Fleuret criteria

                                if n > 0:

                                    # ---------------------------------------------------------------------------------------------------------------
                                    # for cands other than the top one, if best partial gain <= next_best_gain, we can proceed with confirming next_best_gain. else we have to recompute partial gains
                                    # ---------------------------------------------------------------------------------------------------------------
                                    best_partial_gain, best_key = find_best_partial_gain(
                                        partial_gains=partial_gains,
                                        failed_candidates=failed_candidates,
                                        added_candidates=added_candidates,
                                        candidates=candidates,
                                        selected_vars=selected_vars,
                                    )

                                    if best_partial_gain > next_best_gain:
                                        if verbose:
                                            print(
                                                "Have no best_candidate anymore. Need to recompute partial gains. best_partial_gain of candidate",
                                                get_candidate_name(candidates[best_key], cols),
                                                "was",
                                                best_partial_gain,
                                            )
                                        break

                                # ---------------------------------------------------------------------------------------------------------------
                                # external bootstrapped recheck. is minimal MI of candidate X with Y given all current Zs THAT BIG as next_best_gain?
                                # ---------------------------------------------------------------------------------------------------------------

                                bootstrapped_gain, confidence = get_fleuret_criteria_confidence(
                                    data_copy=data_copy,
                                    var_nbins=var_nbins,
                                    x=candidates[next_best_candidate],
                                    y=y,
                                    selected_vars=selected_vars,
                                    bootstrapped_gain=next_best_gain,
                                    npermutations=npermutations,
                                    max_failed=max_failed,
                                )
                            # ---------------------------------------------------------------------------------------------------------------
                            # Report this particular best candidate
                            # ---------------------------------------------------------------------------------------------------------------

                            if bootstrapped_gain > 0:
                                # bad confidence can make us consider other candidates!
                                next_best_gain = next_best_gain * confidence
                                expected_gains[next_best_candidate] = next_best_gain

                                best_partial_gain, best_key = find_best_partial_gain(
                                    partial_gains=partial_gains,
                                    failed_candidates=failed_candidates,
                                    added_candidates=added_candidates,
                                    candidates=candidates,
                                    selected_vars=selected_vars,
                                    skip_indices=(next_best_candidate,),
                                )
                                if best_partial_gain > next_best_gain:
                                    if verbose:
                                        print(
                                            "Candidate's lowered confidence",
                                            confidence,
                                            "requires re-checking other candidates, as now its expected_gain is only",
                                            next_best_gain,
                                            "vs",
                                            best_partial_gain,
                                            "of",
                                            get_candidate_name(candidates[best_key], cols),
                                        )
                                    break
                                else:
                                    ncons_confirmation_failures = 0
                                    cand_confirmed = True
                                    if verbose:
                                        print("\tconfirmed with confidence", confidence)
                                    break
                            else:
                                expected_gains[next_best_candidate] = 0.0
                                failed_candidates.add(next_best_candidate)
                                if verbose:
                                    print("\tconfirmation failed with confidence", confidence)

                                ncons_confirmation_failures += 1
                                if ncons_confirmation_failures > max_cons_confirmation_failures:
                                    if verbose:
                                        print("max_cons_confirmation_failures=", ncons_confirmation_failures, "reached!")
                                    min_gain_reached = True
                                    break
                                if next_best_gain < min_gain:
                                    if verbose:
                                        print("min_gain=", next_best_gain, "reached!")
                                    min_gain_reached = True
                                    break

                                best_gain = 0
                        else:  # next_best_gain = 0
                            break  # to find more good valid candidates

                    # ---------------------------------------------------------------------------------------------------------------
                    # let's act upon results of the permutation test
                    # ---------------------------------------------------------------------------------------------------------------

                    if cand_confirmed:
                        added_candidates.add(next_best_candidate)
                        expected_gains[next_best_candidate] = 0.0  # so it won't be selected again
                        best_candidate = candidates[next_best_candidate]
                        best_gain = next_best_gain
                        break  # exit confirmation while loop
                    else:
                        if not any_cand_considered:
                            best_gain = 0
                            if verbose:
                                print("No more candidates to confirm.")
                            break  # exit confirmation while loop
                        else:
                            if min_gain_reached:
                                break  # exit while
                else:  # if no npermutations is specified
                    break  # exit confirmation while loop
            if best_gain >= min_gain and ncons_confirmation_failures <= max_cons_confirmation_failures:
                for cand in best_candidate:
                    if cand not in selected_vars:
                        selected_vars.append(cand)
                        if subset_size > 1:
                            selected_interactions_vars.append(cand)
                # if verbose:
                cand_name = get_candidate_name(best_candidate, cols)
                logger.info(f"added new candidate {cand_name} to the list with best_gain={best_gain:.10f}")
                predictors.append({"name": cand_name, "indices": best_candidate, "gain": best_gain, "confidence": confidence})
            else:
                # if verbose:
                print(f"Can't add anything valuable anymore for subset_size={subset_size}")
                break

    # postprocess_candidates(selected_vars)

    return selected_vars, predictors


@njit()
def get_fleuret_criteria_confidence(
    data_copy: np.ndarray,
    var_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    bootstrapped_gain: float,
    npermutations: int,
    max_failed: int,
) -> tuple:

    nfailed = 0
    # permute X,Y,Z npermutations times

    for i in range(npermutations):

        current_gain = 1e30

        for idx in y:
            np.random.shuffle(data_copy[:, idx])
        # for idx in x:
        #     np.random.shuffle(data_copy[:, idx])

        for Z in selected_vars:

            # for idx in [Z]:
            #     np.random.shuffle(data_copy[:, idx])

            additional_knowledge = conditional_mi(data=data_copy, x=x, y=y, z=(Z,), var_is_nominal=None, var_nbins=var_nbins)

            if additional_knowledge < current_gain:

                current_gain = additional_knowledge

        if current_gain >= bootstrapped_gain:
            nfailed += 1
            if nfailed >= max_failed:
                bootstrapped_gain = 0.0
                break
    confidence = 1 - nfailed / (i + 1)

    return bootstrapped_gain, confidence


def find_best_partial_gain(
    partial_gains: dict, failed_candidates: set, added_candidates: set, candidates: list, selected_vars: list, skip_indices: tuple = ()
) -> float:
    best_partial_gain = -1e30
    best_key = None
    for key, value in partial_gains.items():
        if (key not in failed_candidates) and (key not in added_candidates) and (key not in skip_indices):
            skip_cand = False
            for subel in candidates[key]:
                if subel in selected_vars:
                    skip_cand = True  # the subelement or var itself is already selected
                    break
            if skip_cand:
                continue
            partial_gain, _ = value
            if partial_gain > best_partial_gain:
                best_partial_gain = partial_gain
                best_key = key
    return best_partial_gain, best_key


def postprocess_candidates(
    selected_vars: list,
    data: np.ndarray,
    y: np.ndarray,
    var_nbins: np.ndarray,
    min_nonzero_confidence: float = 0.99999,
    npermutations: int = 10_000,
    max_subset_size: int = 1,
    ensure_target_influence: bool = True,
    verbose: bool = True,
):
    """Post-analysis of prescreened candidates.

    1) repeat standard Fleuret screening process. maybe some vars will be removed when taken into account all other candidates.
    2) in the final set, compute for every factor
        a) MI with every remaining predictor (and 2,3 way subsets)

    """

    # ---------------------------------------------------------------------------------------------------------------
    # Make sure with confidence that every candidate is related to target
    # ---------------------------------------------------------------------------------------------------------------
    if ensure_target_influence:
        removed = []
        for X in tqdmu(selected_vars, desc="Ensuring target influence", leave=False):
            bootstrapped_mi, confidence = mi_direct(
                data,
                x=[X],
                y=y,
                var_nbins=var_nbins,
                min_nonzero_confidence=min_nonzero_confidence,
                npermutations=npermutations,
            )
            if bootstrapped_mi == 0:
                if verbose:
                    print("Factor", X, "not related to target with confidence", confidence)
                    removed.append(X)
        selected_vars = [el for el in selected_vars if el not in removed]

    # ---------------------------------------------------------------------------------------------------------------
    # Repeat Fleuret process as many times as there is candidates left.
    # This time account for possible interactions (fix bug in the professor's formula).
    # ---------------------------------------------------------------------------------------------------------------

    """
    Compute redundancy stats for every X

    кто связан с каким количеством других факторов, какое количество информации с ними разделяет, в % к своей собственной энтропии. 
    Можно даже спуститься вниз по уровню и посчитать взвешенные суммы тех же метрик для его партнёров. 
    Тем самым можно косвенно определить, какие фичи скорее всего просто сливные бачки, и попробовать их выбросить.  В итоге мы получим:

    ценные фичи, которые ни с кем другим не связаны, кроме мусорных и таргета. они содержат уникальное знание;
    потенциально мусорные X, которые связаны с множеством других, и шарят очень много общей инфы с другими факторами Z, при том, 
    что эти другие факторы имеют много уникального знания о таргете помимо X: sum(I(Y;Z|X))>e;
    все остальные "середнячки".    
    """
    entropies = {}
    mutualinfos = {}
    for X in tqdmu(selected_vars, desc="Marginal stats", leave=False):
        _, freqs = merge_vars(data=data, vars_indices=[X], var_nbins=var_nbins, var_is_nominal=None)
        factor_entropy = entropy(freqs=freqs)
        entropies[X] = factor_entropy

    for a, b in tqdmu(combinations(selected_vars, 2), desc="Interactions", leave=False):
        bootstrapped_mi, confidence = mi_direct(
            data,
            x=[a],
            y=[b],
            var_nbins=var_nbins,
            min_nonzero_confidence=min_nonzero_confidence,
            npermutations=npermutations,
        )
        if bootstrapped_mi > 0:
            mutualinfos[(a, b)] = bootstrapped_mi
    return entropies, mutualinfos
