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
from timeit import default_timer as timer


from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from numba.core import types
from numba import njit
import numba
import math

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

LARGE_CONST:float=1e30

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
    factors_data: np.ndarray,
    vars_indices: Sequence,
    var_is_nominal: Sequence,
    factors_nbins: Sequence,
    dtype=np.int64,
    min_occupancy: int = None,
    current_nclasses: int = 1,
    final_classes: np.ndarray = np.array([], dtype=np.int64),
    verbose: bool = False,
) -> tuple:
    """Melts multiple vectors (partitioned into bins or categories) into one-dimensional one.
    factors_data columns are assumed to be ordinal-encoded already (by KBinsDiscretizer or similar).
    Bins, scarcely populated in higher dimension, may be merged with nearest neighbours (in eucl. distance).

    Process truly numerical vars first? To be able to distribute scarcely populated bins more meaningfully, according to the distances.
    What order is best to use in regard of number of categories and variance of their population density? Big-small,big-small, etc? Starting from most (un)evenly distributed?
    """

    if len(final_classes) == 0:
        final_classes = np.zeros(len(factors_data), dtype=dtype)
    for var_number, var_index in enumerate(vars_indices):

        expected_nclasses = current_nclasses * factors_nbins[var_index]
        freqs = np.zeros(expected_nclasses, dtype=dtype)
        values = factors_data[:, var_index].astype(dtype)
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
    return final_classes, freqs / len(factors_data), current_nclasses


# ----------------------------------------------------------------------------------------------------------------------------
# Information-theory measures
# ----------------------------------------------------------------------------------------------------------------------------


@njit()
def entropy(freqs: np.ndarray, min_occupancy: float = 0) -> float:
    if min_occupancy:
        freqs = freqs[freqs >= min_occupancy]

    return -(np.log(freqs) * freqs).sum()


@njit()
def mi(factors_data, x: np.ndarray, y: np.ndarray, factors_nbins: np.ndarray, verbose: bool = False) -> float:
    """Computes Mutual Information of X, Y via entropy calculations."""

    classes_x, freqs_x, _ = merge_vars(factors_data=factors_data, vars_indices=x, var_is_nominal=None, factors_nbins=factors_nbins, verbose=verbose)
    entropy_x = entropy(freqs=freqs_x)
    if verbose:
        print(f"entropy_x={entropy_x}, nclasses_x={len(freqs_x)} ({classes_x.min()} to {classes_x.max()})")

    _, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, verbose=verbose)
    entropy_y = entropy(freqs=freqs_y)
    if verbose:
        print(f"entropy_y={entropy_y}, nclasses_y={len(freqs_y)}")

    classes_xy, freqs_xy, _ = merge_vars(factors_data=factors_data, vars_indices=set(x) | set(y), var_is_nominal=None, factors_nbins=factors_nbins, verbose=verbose)
    entropy_xy = entropy(freqs=freqs_xy)
    if verbose:
        print(f"entropy_xy={entropy_xy}, nclasses_x={len(freqs_xy)} ({classes_xy.min()} to {classes_xy.max()})")

    return entropy_x + entropy_y - entropy_xy


@njit()
def arr2str(arr: Sequence) -> str:
    s = ""
    for el in arr:
        s += str(el)
    return s


@njit()
def conditional_mi(
    factors_data: np.ndarray,
    x: tuple,
    y: tuple,
    z: tuple,
    var_is_nominal: Sequence,
    factors_nbins: Sequence,
    entropy_z: float = -1.0,
    entropy_xz: float = -1.0,
    entropy_yz: float = -1.0,
    entropy_xyz: float = -1.0,
    entropy_cache: dict = None,
    can_use_y_cache: bool = False,
) -> float:
    """
    Conditional Mutual Information about Y by X given Z = I (X ;Y | Z ) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
    When y is constant, and both X(candidates) and Z (veteranes) repeat a lot, there's room to optimize.
    Also when parts of X repeat a lot (2, 3 way interactions). Z and Y are always 1-dim.
    """
    key = ""

    if entropy_z < 0:
        if entropy_cache is not None:
            key = arr2str(z)
            entropy_z = entropy_cache.get(key, -1)
        if entropy_z < 0:
            _, freqs_z, _ = merge_vars(factors_data=factors_data, vars_indices=z, var_is_nominal=None, factors_nbins=factors_nbins)  # always 1-dim
            entropy_z = entropy(freqs=freqs_z)
            if entropy_cache is not None:
                entropy_cache[key] = entropy_z

    if entropy_xz < 0:
        indices = sorted([*x, *z])
        if entropy_cache is not None:
            key = arr2str(indices)
            entropy_xz = entropy_cache.get(key, -1)
        if entropy_xz < 0:
            _, freqs_xz, _ = merge_vars(factors_data=factors_data, vars_indices=indices, var_is_nominal=None, factors_nbins=factors_nbins)
            entropy_xz = entropy(freqs=freqs_xz)
            if entropy_cache is not None:
                entropy_cache[key] = entropy_xz

    current_nclasses_yz = 1
    if can_use_y_cache:
        if entropy_yz < 0:
            indices = sorted([*y, *z])
            if entropy_cache is not None:
                key = arr2str(indices)
                entropy_yz = entropy_cache.get(key, -1)
            if entropy_yz < 0:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(factors_data=factors_data, vars_indices=indices, var_is_nominal=None, factors_nbins=factors_nbins)
                entropy_yz = entropy(freqs=freqs_yz)
                if entropy_cache is not None:
                    entropy_cache[key] = entropy_yz
    else:
        classes_yz, freqs_yz, current_nclasses_yz = merge_vars(factors_data=factors_data, vars_indices=[*y, *z], var_is_nominal=None, factors_nbins=factors_nbins)  # always 2-dim
        entropy_yz = entropy(freqs=freqs_yz)

    if entropy_xyz < 0:
        if can_use_y_cache:
            indices = sorted([*x, *y, *z])
            if entropy_cache is not None:
                key = arr2str(indices)
                entropy_xyz = entropy_cache.get(key, -1)
        if entropy_xyz < 0:
            if current_nclasses_yz == 1:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
                    factors_data=factors_data, vars_indices=[*y, *z], var_is_nominal=None, factors_nbins=factors_nbins
                )  # always 2-dim

            _, freqs_xyz, _ = merge_vars(
                # factors_data=factors_data, vars_indices=[*y, *z, *x], var_is_nominal=None, factors_nbins=factors_nbins
                factors_data=factors_data,
                vars_indices=x,
                var_is_nominal=None,
                factors_nbins=factors_nbins,
                current_nclasses=current_nclasses_yz,
                final_classes=classes_yz,
            )  # upper classes of [*y, *z] can serve here. (2+x)-dim, ie 3 to 5 dim.
            entropy_xyz = entropy(freqs=freqs_xyz)
            if entropy_cache is not None and can_use_y_cache:
                entropy_cache[key] = entropy_xz

    return entropy_xz + entropy_yz - entropy_z - entropy_xyz


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
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    min_occupancy: int = None,
    dtype=np.int64,
    verbose: bool = False,
    min_nonzero_confidence: float = 0.95,
    full_npermutations: int = 10,
) -> tuple:

    classes_x, freqs_x, _ = merge_vars(factors_data=factors_data, vars_indices=x, var_is_nominal=None, factors_nbins=factors_nbins)
    classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins)
    if verbose:
        print(f"nclasses_x={len(freqs_x)} ({classes_x.min()} to {classes_x.max()}), nclasses_y={len(freqs_y)} ({classes_y.min()} to {classes_y.max()})")

    original_mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)

    if original_mi>0 and full_npermutations>0:
        # inits
        nfailed = 0
        max_failed = int(full_npermutations * (1 - min_nonzero_confidence))

        # # copy factors_data for safe shuffling
        # if len(x) > 1:
        #     x_copy = factors_data[:, np.array(x)].copy()
        #     x_var_nbins = factors_nbins[np.array(x)]
        # if len(y) > 1:
        #     y_copy = factors_data[:, np.array(y)].copy()
        #     y_var_nbins = factors_nbins[np.array(y)]

        for i in range(full_npermutations):
            # if len(x) > 1:
            #     for idx in range(len(x)):
            #         np.random.shuffle(x_copy[:, idx])
            #     classes_x, freqs_x, _ = merge_vars(factors_data=x_copy, vars_indices=np.arange(len(x)), var_is_nominal=None, factors_nbins=x_var_nbins)
            # else:
            #     np.random.shuffle(classes_x)

            # if len(y) > 1:
            #     for idx in range(len(y)):
            #         np.random.shuffle(y_copy[:, idx])
            #     classes_y, freqs_y, _ = merge_vars(factors_data=y_copy, vars_indices=np.arange(len(y)), var_is_nominal=None, factors_nbins=y_var_nbins)
            # else:
            #     np.random.shuffle(classes_y)

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


def get_candidate_name(candidate_indices: list, factors_names: list) -> str:
    cand_name = "-".join([factors_names[el] for el in candidate_indices])
    return cand_name


def should_skip_candidate(
    cand_idx: int,
    X: tuple,
    interactions_order: int,    
    failed_candidates: set,
    added_candidates: set,
    expected_gains: np.ndarray,
    selected_vars: list,
    selected_interactions_vars: list,
    only_unknown_interactions: bool = True,
    
) -> bool:
    """Decides if current candidate for predictors should be skipped 
    ('cause of being already accepted, failed, computed).
    """
    if (cand_idx in failed_candidates) or (cand_idx in added_candidates) or expected_gains[cand_idx]:
        return True

    if interactions_order > 1:  # disabled for single predictors 'cause Fleuret formula won't detect pairs predictors

        # ---------------------------------------------------------------------------------------------------------------
        # Check if any of sub-elements is already selected at this stage
        # ---------------------------------------------------------------------------------------------------------------

        skip_cand = False
        for subel in X:
            if subel in selected_interactions_vars:
                skip_cand = True
                break
        if skip_cand:
            return True

        # ---------------------------------------------------------------------------------------------------------------
        # Or all selected at the lower stages
        # ---------------------------------------------------------------------------------------------------------------

        skip_cand = [(subel in selected_vars) for subel in X]
        if (only_unknown_interactions and any(skip_cand)) or all(skip_cand):
            return True

def evaluate_candidate(cand_idx:int,X:tuple,
    factors_data:np.ndarray,factors_nbins:np.ndarray,
    expected_gains:np.ndarray,partial_gains:dict,
    selected_vars:list,baseline_npermutations:int,
    mrmr_relevance_algo:str="fleuret",
    mrmr_redundancy_algo:str="fleuret",    
    cached_MIs:dict,cached_confident_MIs:dict,cached_cond_MIs:dict)->None:

    # ---------------------------------------------------------------------------------------------------------------
    # Is this candidate any good for target 1-vs-1?
    # ---------------------------------------------------------------------------------------------------------------

    if X in cached_confident_MIs: #use cached_confident_MIs first here as they are more reliable. (but not fill them)
        direct_gain = cached_confident_MIs[X]
    else:
        if X in cached_MIs: 
            direct_gain = cached_MIs[X]
        else:
            direct_gain, _ = mi_direct(factors_data, x=X, y=y, factors_nbins=factors_nbins, min_nonzero_confidence=1.0, full_npermutations=baseline_npermutations)
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
                current_gain = LARGE_CONST
                last_checked_z = -1
            
            positive_mode=False

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
                        additional_knowledge = conditional_mi(
                            factors_data=factors_data,
                            x=X,
                            y=y,
                            z=(Z,),
                            var_is_nominal=None,
                            factors_nbins=factors_nbins,
                            entropy_cache=entropy_cache,
                            can_use_y_cache=True,
                        )
                        cached_cond_MIs[key] = additional_knowledge
                    
                    # ---------------------------------------------------------------------------------------------------------------
                    # account for possible extra knowledge from conditioning on Z
                    # that must update best_gain globally. log such cases. Note that we do not guarantee finding them in order,
                    # but they are too precious to ignore. Adding this will also allow to skip higher order interactions
                    # containing all of already approved candidates.
                    # ---------------------------------------------------------------------------------------------------------------

                    if additional_knowledge > direct_gain:
                        if verbose:
                            logger.info(f"Candidate {get_candidate_name(X)} together with factor {get_candidate_name(Z)} has synergetic influence {additional_knowledge:5f} on target {target} (direct MI={direct_gain:5f})")                                        
                        if not positive_mode:
                            current_gain=additional_knowledge
                            positive_mode=True
                        else:
                            # rare chance that a candidate has many excellent relationships
                            if additional_knowledge > current_gain:
                                current_gain=additional_knowledge

                    if not positive_mode and (additional_knowledge < current_gain):

                        current_gain = additional_knowledge

                        if current_gain <= best_gain:

                            # ---------------------------------------------------------------------------------------------------------------
                            # no point checking other Zs, 'cause current_gain already won't be better than the best_gain 
                            # (if best_gain was estimated confidently, which we'll check at the end.)
                            # ---------------------------------------------------------------------------------------------------------------

                            partial_gains[cand_idx] = current_gain, z_idx
                            current_gain = 0
                            break

            else:  # there was no break. current_gain computed fully.
                partial_gains[cand_idx] = current_gain, z_idx
                expected_gains[cand_idx] = current_gain
        else:
            # no factors selected yet. current_gain is just direct_gain
            current_gain = direct_gain
            expected_gains[cand_idx] = current_gain
    else:
        current_gain = 0


# @njit()
def screen_predictors(
    # factors
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str]=None, 
    factors_to_use: Sequence[int]=None,
    # targets
    targets_data: np.ndarray,
    targets_nbins: Sequence[int],
    y: Sequence[int],
    targets:Union[dict,Sequence[Sequence]]=None,
    # algorithm
    mrmr_relevance_algo:str="fleuret",
    mrmr_redundancy_algo:str="fleuret",
    # performance
    dtype=np.int64,
    # confidence
    min_occupancy: int = None,
    min_nonzero_confidence: float = 0.99,
    full_npermutations: int = 10_000,
    baseline_npermutations: int = 100,
    # stopping conditions
    min_relevance_gain: float = 0.00001,
    max_consec_unconfirmed: int = 10,
    max_run_time_mins:int=None,
    interactions_min_order: int = 1,
    interactions_max_order: int = 1,
    interactions_order_reversed:bool=False,
    max_joint_interactions_order: int = 3,
    only_unknown_interactions: bool = True,
    # verbosity and formatting
    verbose: int = 1,
    ndigits:int=5,
) -> float:
    """Finds best predictors for the target.
    x must be n-x-m array of integers (ordinal encoded)
    Parameters:
        full_npermutations: when computing every MI, repeat calculations with randomly shuffled indices that many times
        min_nonzero_confidence: if in random permutation tests this or higher % of cases had worse current_gain than original, current_gain value is considered valid, otherwise, it's set to zero.
    Returns:
        1) best set of non-redundant single features influencing the target
        2) subsets of size 2..interactions_max_order influencing the target. Such subsets will be candidates for predictors and OtherVarsEncoding.
        3) all 1-vs-1 influencers (not necessarily in mRMR)
    Parameters:
        only_unknown_interactions: True for speed, False for completeness of higher order interactions discovery.
        verbose: int  1=log only important info,>1=also log additional details
        mrmr_relevance_algo:str 
                        "fleuret": max(min(I(X,Y|Z)),max(I(X,Y|Z)-I(X,Y))) Possible to use n-way interactions here.
                        "pld": I(X,Y)
        mrmr_redundancy_algo:str 
                        "fleuret": 0 ('cause redundancy already accounted for)                        
                        "pld_max": max(I(veterane,cand)) Possible to use n-way interactions here.
                        "pld_mean": mean(I(veterane,cand)) Possible to use n-way interactions here.
    """
    # ---------------------------------------------------------------------------------------------------------------
    # Input checks
    # ---------------------------------------------------------------------------------------------------------------

    assert mrmr_relevance_algo in ("fleuret","pld")
    assert mrmr_redundancy_algo in ("fleuret","pld_max","pld_mean")

    assert len(factors_data)>=10
    if targets_data is None:
        targets_data=factors_data
    else:
        assert len(factors_data)==len(targets_data)
    
    if targets_nbins is None:
        targets_nbins=factors_nbins    
    
    assert targets_data.shape[1]==len(targets_nbins)
    assert factors_data.shape[1]==len(factors_nbins)

    if len(factors_names)==0:
        factors_names=["F"+str(i) for i in range(len(factors_data))]
    else:
        assert factors_data.shape[1]==len(factors_names)
    
    # warn if inputs are identical to targets
    if factors_data.shape==targets_data.shape:
        if np.shares_memory(factors_data,targets_data):
        if factors_to_use is None:
         
            if verbose>1:
                logger.info("factors_data and targets_data share the same memory. factors_to_use will be determinted automatically to not contain any target columns.")
        x = set(range(factors_data.shape[1])) - set(y)
        else:

            assert not set(y).issubset(set(x))

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    start_time=timer()
    run_out_of_time=False

    max_failed = int(full_npermutations * (1 - min_nonzero_confidence))

    selected_interactions_vars = []
    selected_vars = []  # stores just indices. can't use set 'cause the order is important for efficient computing
    predictors = []  # stores more details.

    cached_MIs = dict()
    cached_cond_MIs = dict()
    cached_confident_MIs = dict()
    entropy_cache = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    data_copy = factors_data.copy()

    subsets=range(interactions_min_order, interactions_max_order + 1)
    if interactions_order_reversed:        
        subsets=subsets[::-1]
    
    for interactions_order in (subsets_pbar:=tqdmu(subsets, desc="Interactions order")):

        if run_out_of_time: break
        subsets_pbar.set_description(f"{interactions_order}-way interactions")

        # ---------------------------------------------------------------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------------------------------------------------------------

        candidates = [tuple(el) for el in combinations(x, interactions_order)]

        # ---------------------------------------------------------------------------------------------------------------
        # Subset level inits
        # ---------------------------------------------------------------------------------------------------------------

        partial_gains = {}
        added_candidates = set()
        failed_candidates = set()
        min_gain_reached = False
        nconsec_unconfirmed = 0

        for _ in tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors"):

            if run_out_of_time: break

            # ---------------------------------------------------------------------------------------------------------------
            # Find candidate X with the highest current_gain given already selected factors
            # ---------------------------------------------------------------------------------------------------------------

            best_gain = min_relevance_gain-1
            expected_gains = np.zeros(len(candidates), dtype=np.float64)

            while True:  # confirmation loop (by random permutations)
                for cand_idx, X in enumerate(tqdmu(candidates, leave=False, desc="Candidates")):

                    if should_skip_candidate(cand_idx=cand_idx,X=X,interactions_order=interactions_order,only_unknown_interactions=only_unknown_interactions,
                        failed_candidates=failed_candidates,added_candidates=added_candidates,expected_gains=expected_gains,selected_vars=selected_vars,
                        selected_interactions_vars=selected_interactions_vars):
                        continue
                    
                    current_gain,_=evaluate_candidate(cand_idx=cand_idx,X=X,
                        mrmr_relevance_algo=mrmr_relevance_algo,
                        mrmr_redundancy_algo=mrmr_redundancy_algo,
                        expected_gains=expected_gains=expected_gains,selected_vars=selected_vars,cached_MIs=cached_MIs,cached_confident_MIs=cached_confident_MIs,cached_cond_MIs=cached_cond_MIs)
                

                    # ---------------------------------------------------------------------------------------------------------------
                    # Save best known candidate, to be able to use early stopping
                    # ---------------------------------------------------------------------------------------------------------------

                    if current_gain > best_gain:
                        best_candidate = X
                        best_gain = current_gain
                        if verbose>1:
                            print(f"\t{get_candidate_name(best_candidate,factors_names)} is so far the best candidate with best_gain={best_gain:.10f}")
                    else:
                        if verbose>1 and  current_gain > min_relevance_gain:
                            print(f"\t\t{get_candidate_name(X,factors_names)} current_gain={current_gain:.{ndigits}f}")

                    run_out_of_time=timer()-start_time>max_run_time_minutes
                    if run_out_of_time:
                        logging.info(f"Time limit exhausted. Finalizing the search.")
                        break                

                if best_gain < min_relevance_gain:
                    if verbose:
                        logger.info("Minimum expected gain reached.")   
                    break  # exit confirmation while loop

                # ---------------------------------------------------------------------------------------------------------------
                # Now need to confirm best expected gain with a permutation test
                # ---------------------------------------------------------------------------------------------------------------

                if full_npermutations:
                    cand_confirmed = False
                    any_cand_considered = False
                    for n, next_best_candidate_idx in enumerate(np.argsort(expected_gains)[::-1]):
                        next_best_gain = expected_gains[next_best_candidate_idx]
                        if next_best_gain >= min_relevance_gain:  # only can consider here candidates fully checked against every Z
                            any_cand_considered = True
                            if verbose>1:
                                print("confirming candidate", get_candidate_name(candidates[next_best_candidate_idx], factors_names), "next_best_gain=", next_best_gain)

                            bootstrapped_gain,confidence=evaluate_candidate(cand_idx=cand_idx,X=X,
                                mrmr_relevance_algo=mrmr_relevance_algo,
                                mrmr_redundancy_algo=mrmr_redundancy_algo,
                                expected_gains=expected_gains=expected_gains,selected_vars=selected_vars,cached_MIs=cached_MIs,cached_confident_MIs=cached_confident_MIs,cached_cond_MIs=cached_cond_MIs)                            
                            
                            # ---------------------------------------------------------------------------------------------------------------
                            # Report this particular best candidate
                            # ---------------------------------------------------------------------------------------------------------------

                            if bootstrapped_gain > 0:
                                
                                nconsec_unconfirmed = 0

                                # ---------------------------------------------------------------------------------------------------------------
                                # Bad confidence can make us consider other candidates!
                                # ---------------------------------------------------------------------------------------------------------------

                                next_best_gain = next_best_gain * confidence
                                expected_gains[next_best_candidate_idx] = next_best_gain

                                best_partial_gain, best_key = find_best_partial_gain(
                                    partial_gains=partial_gains,
                                    failed_candidates=failed_candidates,
                                    added_candidates=added_candidates,
                                    candidates=candidates,
                                    selected_vars=selected_vars,
                                    skip_indices=(next_best_candidate_idx,),
                                )
                                if best_partial_gain > next_best_gain:
                                    if verbose>1:
                                        print(
                                            "Candidate's lowered confidence",
                                            confidence,
                                            "requires re-checking other candidates, as now its expected gain is only",
                                            next_best_gain,
                                            "vs",
                                            best_partial_gain,
                                            "of",
                                            get_candidate_name(candidates[best_key], factors_names),
                                        )
                                    break # out of best candidates confirmation, to retry all cands evaluation
                                else:                                    
                                    cand_confirmed = True
                                    if verbose>1:
                                        print("\tconfirmed with confidence", confidence)
                                    break # out of best candidates confirmation, to add candidate to the list, and go to more candidates
                            else:
                                expected_gains[next_best_candidate_idx] = 0.0
                                failed_candidates.add(next_best_candidate_idx)
                                if verbose>1:
                                    print("\tconfirmation failed with confidence", confidence)

                                nconsec_unconfirmed += 1
                                if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                                    if verbose:
                                        logger.info(f"Maximum consecutive confirmation failures reached.")
                                    break # out of best candidates confirmation, to finish the level
                        else:  # next_best_gain = 0                                                     
                            break # nothing wrong, just retry all cands evaluation
                    
                    # ---------------------------------------------------------------------------------------------------------------
                    # Let's act upon results of the permutation test
                    # ---------------------------------------------------------------------------------------------------------------

                    if cand_confirmed:
                        added_candidates.add(next_best_candidate_idx) # so it won't be selected again
                        best_candidate = candidates[next_best_candidate_idx]
                        best_gain = next_best_gain
                        break  # exit confirmation while loop
                    else:
                        if not any_cand_considered:
                            best_gain = min_relevance_gain-1
                            if verbose:
                                logger.info("No more candidates to confirm.")
                            break  # exit confirmation while loop
                        else:
                            if max_consec_unconfirmed and nconsec_unconfirmed > max_consec_unconfirmed:
                                break  # exit confirmation while loop
                            else:
                                pass # retry all cands evaluation
                else:  # if no full_npermutations is specified
                    break  # exit confirmation while loop
                
            # ---------------------------------------------------------------------------------------------------------------
            # Add best candidate to the list, if criteria are met, or proceed to the next interactions_order
            # ---------------------------------------------------------------------------------------------------------------

            if best_gain >= min_relevance_gain:
                for var in best_candidate:
                    if var not in selected_vars:
                        selected_vars.append(var)
                        if interactions_order > 1:
                            selected_interactions_vars.append(var)                
                cand_name = get_candidate_name(best_candidate, factors_names)
                if verbose: 
                    logger.info(f"Added new predictor {cand_name} to the list with expected gain={best_gain:.{ndigits}f}")
                predictors.append({"name": cand_name, "indices": best_candidate, "gain": best_gain, "confidence": confidence})
            else:
                if verbose:
                    logger.info(f"Can't add anything valuable anymore for interactions_order={interactions_order}")
                break

    # postprocess_candidates(selected_vars)

    return selected_vars, predictors


@njit()
def get_fleuret_criteria_confidence(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    bootstrapped_gain: float,
    full_npermutations: int,
    max_failed: int,
    entropy_cache: dict = None,
) -> tuple:

    nfailed = 0
    # permute X,Y,Z full_npermutations times

    for i in range(full_npermutations):

        current_gain = 1e30

        for idx in y:
            np.random.shuffle(data_copy[:, idx])
        # for idx in x:
        #     np.random.shuffle(data_copy[:, idx])

        for Z in selected_vars:

            # for idx in [Z]:
            #     np.random.shuffle(data_copy[:, idx])

            additional_knowledge = conditional_mi(factors_data=data_copy, x=x, y=y, z=(Z,), var_is_nominal=None, factors_nbins=factors_nbins, entropy_cache=entropy_cache)

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
    factors_data: np.ndarray,
    y: np.ndarray,
    factors_nbins: np.ndarray,
    min_nonzero_confidence: float = 0.99999,
    full_npermutations: int = 10_000,
    interactions_max_order: int = 1,
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
                factors_data,
                x=[X],
                y=y,
                factors_nbins=factors_nbins,
                min_nonzero_confidence=min_nonzero_confidence,
                full_npermutations=full_npermutations,
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
        _, freqs, _ = merge_vars(factors_data=factors_data, vars_indices=[X], factors_nbins=factors_nbins, var_is_nominal=None)
        factor_entropy = entropy(freqs=freqs)
        entropies[X] = factor_entropy

    for a, b in tqdmu(combinations(selected_vars, 2), desc="Interactions", leave=False):
        bootstrapped_mi, confidence = mi_direct(
            factors_data,
            x=[a],
            y=[b],
            factors_nbins=factors_nbins,
            min_nonzero_confidence=min_nonzero_confidence,
            full_npermutations=full_npermutations,
        )
        if bootstrapped_mi > 0:
            mutualinfos[(a, b)] = bootstrapped_mi
    return entropies, mutualinfos
