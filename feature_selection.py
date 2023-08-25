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
import cupy as cp


from pyutilz.system import tqdmu
from pyutilz.parallel import mem_map_array, split_list_into_chunks
from pyutilz.numbalib import set_random_seed, arr2str, python_dict_2_numba_dict, generate_combinations_recursive_njit

# from mlframe.boruta_shap import BorutaShap
from timeit import default_timer as timer


from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from numba.core import types
from numba import njit
import numba
import math

from joblib import Parallel, delayed

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

MAX_JOBLIB_NBYTES = 1e3
NMAX_NONPARALLEL_ITERS = 2
MAX_ITERATIONS_TO_TRACK = 5

LARGE_CONST: float = 1e30
GPU_MAX_BLOCK_SIZE = 1024

caching_hits_xyz = 0
caching_hits_z = 0
caching_hits_xz = 0
caching_hits_yz = 0

compute_joint_hist_cuda = cp.RawKernel(
    r"""
extern "C" __global__
void compute_joint_hist_cuda(const int *classes_x, const int *classes_y, int *joint_counts, int n, int nbins_y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid<n){
        atomicAdd(&joint_counts[classes_x[tid]*nbins_y+classes_y[tid]],1);
    }
}
""",
    "compute_joint_hist_cuda",
)
compute_mi_from_classes_cuda = cp.RawKernel(
    r"""
extern "C" __global__
void compute_mi_from_classes_cuda(const int *classes_x, const double *freqs_x,const int *classes_y, const double *freqs_y, int *joint_counts, double *totals, int n,int nbins_x,int nbins_y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;    
    if (tid==0){
        double total = 0.0;
        for (int i=0;i<nbins_x;++i){
            float prob_x = freqs_x[i];
            for (int j=0;j<nbins_y;++j){
                int jc = joint_counts[i*2+j];
                if (jc>0){
                    float prob_y = freqs_y[j];
                    double jf=(float)jc/ (float)n;
                    total += jf* log(jf / (prob_x * prob_y));
                }
            }
        }    
        totals[0]=total;
    }
}
""",
    "compute_mi_from_classes_cuda",
)

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
    dtype=np.int32,
    min_occupancy: int = None,
    current_nclasses: int = 1,
    final_classes: np.ndarray = None,  # , dtype=np.int32
    verbose: bool = False,
) -> tuple:
    """Melts multiple vectors (partitioned into bins or categories) into one-dimensional one.
    factors_data columns are assumed to be ordinal-encoded already (by KBinsDiscretizer or similar).
    Bins, scarcely populated in higher dimension, may be merged with nearest neighbours (in eucl. distance).

    Process truly numerical vars first? To be able to distribute scarcely populated bins more meaningfully, according to the distances.
    What order is best to use in regard of number of categories and variance of their population density? Big-small,big-small, etc? Starting from most (un)evenly distributed?
    """

    if final_classes is None:
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
def mi(factors_data, x: np.ndarray, y: np.ndarray, factors_nbins: np.ndarray, verbose: bool = False, dtype=np.int32) -> float:
    """Computes Mutual Information of X, Y via entropy calculations."""

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x, var_is_nominal=None, factors_nbins=factors_nbins, verbose=verbose, dtype=dtype
    )
    entropy_x = entropy(freqs=freqs_x)
    if verbose:
        print(f"entropy_x={entropy_x}, nclasses_x={len(freqs_x)} ({classes_x.min()} to {classes_x.max()})")

    _, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, verbose=verbose, dtype=dtype)
    entropy_y = entropy(freqs=freqs_y)
    if verbose:
        print(f"entropy_y={entropy_y}, nclasses_y={len(freqs_y)}")

    classes_xy, freqs_xy, _ = merge_vars(
        factors_data=factors_data, vars_indices=set(x) | set(y), var_is_nominal=None, factors_nbins=factors_nbins, verbose=verbose, dtype=dtype
    )
    entropy_xy = entropy(freqs=freqs_xy)
    if verbose:
        print(f"entropy_xy={entropy_xy}, nclasses_x={len(freqs_xy)} ({classes_xy.min()} to {classes_xy.max()})")

    return entropy_x + entropy_y - entropy_xy


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
    can_use_x_cache: bool = False,
    can_use_y_cache: bool = False,
    dtype=np.int32,
) -> float:
    """
    Conditional Mutual Information about Y by X given Z = I (X ;Y | Z ) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
    When y is constant, and both X(candidates) and Z (veteranes) repeat a lot, there's room to optimize.
    Also when parts of X repeat a lot (2, 3 way interactions). Z and Y are always 1-dim.
    """
    # global caching_hits_xyz, caching_hits_z, caching_hits_xz, caching_hits_yz

    key = ""

    if entropy_z < 0:
        if entropy_cache is not None:
            key = arr2str(sorted(z))
            entropy_z = entropy_cache.get(key, -1)
        if entropy_z < 0:
            _, freqs_z, _ = merge_vars(factors_data=factors_data, vars_indices=z, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)  # always 1-dim
            entropy_z = entropy(freqs=freqs_z)
            if entropy_cache is not None:
                entropy_cache[key] = entropy_z
        # else:
        #    caching_hits_z += 1

    if entropy_xz < 0:
        indices = sorted([*x, *z])
        if can_use_x_cache and entropy_cache is not None:
            key = arr2str(indices)
            entropy_xz = entropy_cache.get(key, -1)
        if entropy_xz < 0:
            _, freqs_xz, _ = merge_vars(factors_data=factors_data, vars_indices=indices, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
            entropy_xz = entropy(freqs=freqs_xz)
            if can_use_x_cache and entropy_cache is not None:
                entropy_cache[key] = entropy_xz
        # else:
        #    caching_hits_xz += 1

    current_nclasses_yz = 1
    if can_use_y_cache:
        if entropy_yz < 0:
            indices = sorted([*y, *z])
            if entropy_cache is not None:
                key = arr2str(indices)
                entropy_yz = entropy_cache.get(key, -1)
            if entropy_yz < 0:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
                    factors_data=factors_data, vars_indices=indices, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype
                )
                entropy_yz = entropy(freqs=freqs_yz)
                if entropy_cache is not None:
                    entropy_cache[key] = entropy_yz
            # else:
            #    caching_hits_yz += 1
    else:
        classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
            factors_data=factors_data, vars_indices=[*y, *z], var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype
        )  # always 2-dim
        entropy_yz = entropy(freqs=freqs_yz)

    if entropy_xyz < 0:
        if can_use_y_cache and can_use_x_cache:
            indices = sorted([*x, *y, *z])
            if entropy_cache is not None:
                key = arr2str(indices)
                entropy_xyz = entropy_cache.get(key, -1)
        if entropy_xyz < 0:
            if current_nclasses_yz == 1:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
                    factors_data=factors_data, vars_indices=[*y, *z], var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype
                )  # always 2-dim

            _, freqs_xyz, _ = merge_vars(
                # factors_data=factors_data, vars_indices=[*y, *z, *x], var_is_nominal=None, factors_nbins=factors_nbins
                factors_data=factors_data,
                vars_indices=x,
                var_is_nominal=None,
                factors_nbins=factors_nbins,
                current_nclasses=current_nclasses_yz,
                final_classes=classes_yz,
                dtype=dtype,
            )  # upper classes of [*y, *z] can serve here. (2+x)-dim, ie 3 to 5 dim.
            entropy_xyz = entropy(freqs=freqs_xyz)
            if entropy_cache is not None and can_use_y_cache and can_use_x_cache:
                entropy_cache[key] = entropy_xyz
        # else:
        #    caching_hits_xyz += 1
    res = entropy_xz + entropy_yz - entropy_z - entropy_xyz
    if res < 0.0:
        # print(x, y, z, res, can_use_y_cache)
        res = 0.0
    return res


@njit()
def compute_mi_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:

    joint_counts = np.zeros((len(freqs_x), len(freqs_y)), dtype=dtype)
    for i, j in zip(classes_x, classes_y):
        joint_counts[i, j] += 1
    joint_freqs = joint_counts / len(classes_x)

    total = 0.0
    for i in range(len(freqs_x)):
        prob_x = freqs_x[i]
        for j in range(len(freqs_y)):
            jf = joint_freqs[i, j]
            if jf:
                prob_y = freqs_y[j]
                total += jf * math.log(jf / (prob_x * prob_y))
    return total


@njit()
def distribute_permutations(npermutations: int, nworkers: int) -> list:

    avg_perms_per_worker = npermutations // nworkers
    diff = npermutations - avg_perms_per_worker * nworkers
    workload = [avg_perms_per_worker] * nworkers
    if diff > 0:
        workload[-1] = workload[-1] + diff
    return workload


# @njit()
def mi_direct(
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    min_occupancy: int = None,
    dtype=np.int32,
    npermutations: int = 10,
    max_failed: int = None,
    min_nonzero_confidence: float = 0.95,
    classes_y: np.ndarray = None,
    classes_y_safe: cp.ndarray = None,
    freqs_y: np.ndarray = None,
    nworkers: int = 1,
    workers_pool: object = None,
) -> tuple:

    classes_x, freqs_x, _ = merge_vars(factors_data=factors_data, vars_indices=x, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)

    original_mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)

    if original_mi > 0 and npermutations > 0:

        # Inits

        nfailed = 0
        if not max_failed:
            max_failed = int(npermutations * (1 - min_nonzero_confidence))
            if max_failed <= 1:
                max_failed = 1

        if nworkers and nworkers > 1 and npermutations > NMAX_NONPARALLEL_ITERS:

            # logger.info("Memmapping classes_x...")
            # classes_x_memmap = mem_map_array(obj=classes_x, file_name="classes_x", mmap_mode="r")

            if workers_pool is None:
                workers_pool = Parallel(n_jobs=nworkers, max_nbytes=MAX_JOBLIB_NBYTES)

            res = workers_pool(
                delayed(parallel_mi)(
                    classes_x=classes_x,
                    freqs_x=freqs_x,
                    classes_y=classes_y_safe,
                    freqs_y=freqs_y,
                    dtype=dtype,
                    npermutations=worker_npermutations,
                    original_mi=original_mi,
                    max_failed=max_failed,
                )
                for worker_npermutations in distribute_permutations(npermutations=npermutations, nworkers=nworkers)
            )

            i = 0
            for worker_nfailed, worker_i in res:
                nfailed += worker_nfailed
                i += worker_i

            if nfailed >= max_failed:
                original_mi = 0.0

        else:

            if classes_y_safe is None:
                classes_y_safe = classes_y.copy()

            for i in range(npermutations):

                np.random.shuffle(classes_y_safe)
                mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype)

                if mi >= original_mi:
                    nfailed += 1
                    if nfailed >= max_failed:
                        original_mi = 0.0
                        break

        confidence = 1 - nfailed / (i + 1)

    else:
        confidence = 0

    return original_mi, confidence


@njit()
def parallel_mi(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    max_failed: int,
    dtype=np.int32,
) -> int:

    # print("In parallel_mi. Copying classes_y...")
    nfailed = 0
    classes_y_safe = np.asarray(classes_y).copy()
    # print("In parallel_mi. Copied.")
    for i in range(npermutations):

        np.random.shuffle(classes_y_safe)
        mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype)

        if mi >= original_mi:
            nfailed += 1
            if nfailed >= max_failed:
                break
    return nfailed, i + 1


def mi_direct_gpu(
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    min_occupancy: int = None,
    dtype=np.int32,
    npermutations: int = 10,
    max_failed: int = None,
    min_nonzero_confidence: float = 0.95,
    classes_y: np.ndarray = None,
    classes_y_safe: cp.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: cp.ndarray = None,
    use_gpu: bool = True,
) -> tuple:

    classes_x, freqs_x, _ = merge_vars(factors_data=factors_data, vars_indices=x, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)

    original_mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)

    if original_mi > 0 and npermutations > 0:

        # Inits

        if not max_failed:
            max_failed = int(npermutations * (1 - min_nonzero_confidence))
            if max_failed <= 1:
                max_failed = 1

        if classes_y_safe is None:
            classes_y_safe = cp.asarray(classes_y).astype(cp.int32)
        if freqs_y_safe is None:
            freqs_y_safe = cp.asarray(freqs_y)

        totals = cp.zeros(1, dtype=np.float64)
        joint_counts = cp.zeros((len(freqs_x), len(freqs_y)), dtype=cp.int32)

        block_size = GPU_MAX_BLOCK_SIZE
        grid_size = math.ceil(len(classes_x) / block_size)

        classes_x = cp.asarray(classes_x.astype(np.int32))
        freqs_x = cp.asarray(freqs_x)

        for i in range(npermutations):

            cp.random.shuffle(classes_y_safe)
            joint_counts.fill(0)
            compute_joint_hist_cuda(
                (grid_size,),
                (block_size,),
                (classes_x, classes_y_safe, joint_counts, len(classes_x), len(freqs_y)),
            )
            compute_mi_from_classes_cuda(
                (1,),
                (1,),
                (classes_x, freqs_x, classes_y_safe, freqs_y_safe, joint_counts, totals, len(classes_x), len(freqs_x), len(freqs_y)),
            )

            mi = totals.get()[0]

            if mi >= original_mi:
                nfailed += 1
                if nfailed >= max_failed:
                    original_mi = 0.0
                    break
        confidence = 1 - nfailed / (i + 1)
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
) -> tuple:
    """Decides if current candidate for predictors should be skipped
    ('cause of being already accepted, failed, computed).
    """

    nexisting = 0

    if (cand_idx in failed_candidates) or (cand_idx in added_candidates) or expected_gains[cand_idx]:
        return True, nexisting

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
            return True, nexisting

        # ---------------------------------------------------------------------------------------------------------------
        # Or all selected at the lower stages
        # ---------------------------------------------------------------------------------------------------------------

        skip_cand = [(subel in selected_vars) for subel in X]
        nexisting = sum(skip_cand)
        if (only_unknown_interactions and any(skip_cand)) or all(skip_cand):
            return True, nexisting

    return False, nexisting


def get_fleuret_criteria_confidence_parallel(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    bootstrapped_gain: float,
    npermutations: int,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    nworkers: int = 1,
    workers_pool: object = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
) -> tuple:

    nfailed = 0

    if workers_pool is None:
        workers_pool = Parallel(n_jobs=nworkers, max_nbytes=MAX_JOBLIB_NBYTES)
    res = workers_pool(
        delayed(parallel_fleuret)(
            data=data_copy,
            factors_nbins=factors_nbins,
            x=x,
            y=y,
            selected_vars=selected_vars,
            npermutations=worker_npermutations,
            bootstrapped_gain=bootstrapped_gain,
            max_failed=max_failed,
            nexisting=nexisting,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=dict(cached_cond_MIs),  # cached_cond_MIs needs to be unjitted before sending to joblib.
            entropy_cache=dict(entropy_cache),  # entropy_cache needs to be unjitted before sending to joblib.
            extra_x_shuffling=extra_x_shuffling,
            dtype=dtype,
        )
        for worker_npermutations in distribute_permutations(npermutations=npermutations, nworkers=nworkers)
    )

    nchecked = 0
    for worker_nfailed, worker_i, entropy_cache_dict in res:
        nfailed += worker_nfailed
        nchecked += worker_i
        for key, value in entropy_cache_dict.items():
            entropy_cache[key] = value

    if nfailed >= max_failed:
        bootstrapped_gain = 0.0

    confidence = 1 - nfailed / nchecked

    return bootstrapped_gain, confidence, entropy_cache


def parallel_fleuret(
    data: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
):
    """Stub that gets called by joblib directly. Njits entropy_cache, allocates data copy. Calls fast njitted core sub."""
    data_copy = data.copy()

    entropy_cache_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=entropy_cache, numba_dict=entropy_cache_dict)

    cached_cond_MIs_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=cached_cond_MIs, numba_dict=cached_cond_MIs_dict)

    nfailed, i = get_fleuret_criteria_confidence(
        data_copy=data_copy,
        factors_nbins=factors_nbins,
        x=x,
        y=y,
        selected_vars=selected_vars,
        npermutations=npermutations,
        bootstrapped_gain=bootstrapped_gain,
        max_failed=max_failed,
        nexisting=nexisting,
        mrmr_relevance_algo=mrmr_relevance_algo,
        mrmr_redundancy_algo=mrmr_redundancy_algo,
        max_veteranes_interactions_order=max_veteranes_interactions_order,
        cached_cond_MIs=cached_cond_MIs_dict,
        entropy_cache=entropy_cache_dict,
        extra_x_shuffling=extra_x_shuffling,
        dtype=dtype,
    )

    return nfailed, i, dict(entropy_cache_dict)


@njit()
def get_fleuret_criteria_confidence(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
) -> tuple:
    """Sub to njit work with random shuffling as well."""

    nfailed = 0

    for i in range(npermutations):

        for idx in y:
            np.random.shuffle(data_copy[:, idx])

        if extra_x_shuffling:
            for idx in x:
                np.random.shuffle(data_copy[:, idx])

        stopped_early, current_gain, k, sink_reasons = evaluate_gain(
            current_gain=LARGE_CONST,
            last_checked_k=-1,
            X=x,
            y=y,
            best_gain=None,
            factors_data=data_copy,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars,
            nexisting=nexisting,
            direct_gain=bootstrapped_gain,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=cached_cond_MIs,
            entropy_cache=entropy_cache,
            can_use_x_cache=not extra_x_shuffling,
            can_use_y_cache=False,
            confidence_mode=True,
        )

        if current_gain >= bootstrapped_gain:
            nfailed += 1
            if nfailed >= max_failed:
                break

    return nfailed, i + 1


def evaluate_candidates(
    workload: list,
    y: Sequence[int],
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    factors_names: Sequence[str],
    partial_gains: dict,
    selected_vars: list,
    baseline_npermutations: int,
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
    cached_MIs: dict = None,
    cached_confident_MIs: dict = None,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    dtype=np.int32,
    max_runtime_mins: int = None,
    start_time: float = None,
    min_relevance_gain: float = None,
    verbose: int = 1,
    ndigits: int = 5,
) -> None:

    best_gain = -LARGE_CONST
    best_candidate = None
    expected_gains = {}

    global logger
    logger = logging.getLogger(__name__)

    entropy_cache_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=entropy_cache, numba_dict=entropy_cache_dict)

    cached_cond_MIs_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=cached_cond_MIs, numba_dict=cached_cond_MIs_dict)

    classes_y_safe = classes_y.copy()

    for cand_idx, X, nexisting in (candidates_pbar := tqdmu(workload, leave=False, desc="Thread Candidates")):

        current_gain, sink_reasons = evaluate_candidate(
            cand_idx=cand_idx,
            X=X,
            y=y,
            nexisting=nexisting,
            best_gain=best_gain,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=factors_names,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            use_gpu=use_gpu,
            freqs_y_safe=freqs_y_safe,
            partial_gains=partial_gains,
            baseline_npermutations=baseline_npermutations,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            expected_gains=expected_gains,
            selected_vars=selected_vars,
            cached_MIs=cached_MIs,
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs_dict,
            entropy_cache=entropy_cache_dict,
            verbose=verbose,
            ndigits=ndigits,
            dtype=dtype,
        )

        best_gain, best_candidate, run_out_of_time = handle_best_candidate(
            current_gain=current_gain,
            best_gain=best_gain,
            X=X,
            best_candidate=best_candidate,
            factors_names=factors_names,
            verbose=verbose,
            ndigits=ndigits,
            max_runtime_mins=max_runtime_mins,
            start_time=start_time,
            min_relevance_gain=min_relevance_gain,
        )

        if run_out_of_time:
            break

    for key, value in entropy_cache_dict.items():
        entropy_cache[key] = value
    for key, value in cached_cond_MIs_dict.items():
        cached_cond_MIs[key] = value

    return best_gain, best_candidate, partial_gains, expected_gains, cached_MIs, cached_cond_MIs, entropy_cache


def handle_best_candidate(
    current_gain: float,
    best_gain: float,
    X: Sequence,
    best_candidate: Sequence,
    factors_names: list,
    verbose: int = 1,
    ndigits: int = 5,
    max_runtime_mins: int = None,
    start_time: float = None,
    min_relevance_gain: float = None,
):
    # ---------------------------------------------------------------------------------------------------------------
    # Save best known candidate, to be able to use early stopping
    # ---------------------------------------------------------------------------------------------------------------

    run_out_of_time = False

    if current_gain > best_gain:
        best_candidate = X
        best_gain = current_gain
        if verbose > 1:
            logger.info(
                f"\t{get_candidate_name(best_candidate,factors_names=factors_names)} is so far the best candidate with best_gain={best_gain:.{ndigits}f}"
            )
    else:
        if min_relevance_gain and verbose > 1 and current_gain > min_relevance_gain:
            logger.info(f"\t\t{get_candidate_name(X,factors_names=factors_names)} current_gain={current_gain:.{ndigits}f}")

    if max_runtime_mins and not run_out_of_time:
        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60

    return best_gain, best_candidate, run_out_of_time


@njit()
def evaluate_gain(
    current_gain: float,
    last_checked_k: int,
    direct_gain: float,
    X: Sequence[int],
    y: Sequence[int],
    nexisting: int,
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    selected_vars: list,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 2,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    entropy_cache: dict = None,
    cached_cond_MIs: dict = None,
    can_use_x_cache=False,
    can_use_y_cache=False,
    dtype=np.int32,
    confidence_mode: bool = False,
) -> tuple:

    positive_mode = False
    stopped_early = False
    sink_reasons = None

    k = 0
    for interactions_order in range(max_veteranes_interactions_order):

        for Z in generate_combinations_recursive_njit(np.array(selected_vars, dtype=np.int32), interactions_order + 1):

            if k > last_checked_k:

                if mrmr_relevance_algo == "fleuret":

                    # ---------------------------------------------------------------------------------------------------------------
                    # additional_knowledge = I (X ;Y | Z ) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
                    # I (X,Z) would be entropy_x + entropy_z - entropy_xz.
                    # ---------------------------------------------------------------------------------------------------------------
                    key_found = False
                    if not confidence_mode:
                        key = arr2str(X) + "_" + arr2str(Z)
                        if key in cached_cond_MIs:
                            additional_knowledge = cached_cond_MIs[key]
                            key_found = True

                    if not key_found:

                        additional_knowledge = conditional_mi(
                            factors_data=factors_data,
                            x=X,
                            y=y,
                            z=Z,
                            var_is_nominal=None,
                            factors_nbins=factors_nbins,
                            entropy_cache=entropy_cache,
                            can_use_x_cache=can_use_x_cache,
                            can_use_y_cache=can_use_y_cache,
                            dtype=dtype,
                        )

                        if nexisting > 0:
                            additional_knowledge = additional_knowledge ** (nexisting + 1)

                        if not confidence_mode:
                            cached_cond_MIs[key] = additional_knowledge

                # ---------------------------------------------------------------------------------------------------------------
                # Account for possible extra knowledge from conditioning on Z?
                # that must update best_gain globally. log such cases. Note that we do not guarantee finding them in order,
                # but they are too precious to ignore. Adding this will also allow to skip higher order interactions
                # containing all of already approved candidates.
                # ---------------------------------------------------------------------------------------------------------------

                if extra_knowledge_multipler > 0 and additional_knowledge > direct_gain * extra_knowledge_multipler:
                    bwarn = False
                    if not positive_mode:
                        current_gain = additional_knowledge
                        positive_mode = True
                        bwarn = True
                    else:
                        # rare chance that a candidate has many excellent relationships
                        if additional_knowledge > current_gain:
                            current_gain = additional_knowledge
                            bwarn = True

                    # if bwarn:
                    #    if verbose:
                    #        if current_gain > best_gain:
                    #            logger.info(
                    #                f"\tCandidate {get_candidate_name(X,factors_names=factors_names)} together with factor {get_candidate_name(Z,factors_names=factors_names)} has synergetic influence {additional_knowledge:{ndigits}f} (direct MI={direct_gain:{ndigits}f})"
                    #            )

                if not positive_mode and (additional_knowledge < current_gain):

                    current_gain = additional_knowledge

                    if best_gain is not None and current_gain <= best_gain:

                        # ---------------------------------------------------------------------------------------------------------------
                        # No point checking other Zs, 'cause current_gain already won't be better than the best_gain
                        # (if best_gain was estimated confidently, which we'll check at the end.)
                        # ---------------------------------------------------------------------------------------------------------------

                        # let's also fix what Z caused X (the most) to sink

                        if sink_threshold > -1 and current_gain < sink_threshold:
                            sink_reasons = Z

                        stopped_early = True
                        return stopped_early, current_gain, k, sink_reasons
            k += 1

    return stopped_early, current_gain, k, sink_reasons


def evaluate_candidate(
    cand_idx: int,
    X: Sequence[int],
    y: Sequence[int],
    nexisting: int,
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    factors_names: Sequence[str],
    expected_gains: np.ndarray,
    partial_gains: dict,
    selected_vars: list,
    baseline_npermutations: int,
    classes_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
    cached_MIs: dict = None,
    cached_confident_MIs: dict = None,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 2,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    dtype=np.int32,
    verbose: int = 1,
    ndigits: int = 5,
) -> None:

    sink_reasons = set()

    # ---------------------------------------------------------------------------------------------------------------
    # Is this candidate any good for target 1-vs-1?
    # ---------------------------------------------------------------------------------------------------------------

    if X in cached_confident_MIs:  # use cached_confident_MIs first here as they are more reliable. (but not fill them)
        direct_gain = cached_confident_MIs[X]
    else:
        if X in cached_MIs:
            direct_gain = cached_MIs[X]
        else:
            if use_gpu:
                direct_gain, _ = mi_direct_gpu(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    freqs_y_safe=freqs_y_safe,
                    min_nonzero_confidence=1.0,
                    npermutations=baseline_npermutations,
                    dtype=dtype,
                )
            else:
                direct_gain, _ = mi_direct(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=1.0,
                    npermutations=baseline_npermutations,
                    dtype=dtype,
                )
            cached_MIs[X] = direct_gain

    if direct_gain > 0:
        if selected_vars:

            # ---------------------------------------------------------------------------------------------------------------
            # Some factors already selected.
            # best gain from including X is the minimum of I (X ;Y | Z ) over every Z in already selected_vars.
            # but imaging some variable is correlated to every real predictor plus has random noise. It's real value is zero.
            # only computing I (X ;Y | Z ) will still leave significant impact. but if we sum I(X,Z) over Zs we'll see it shares
            # all its knowledge with the rest of factors and has no value by itself. But to see that, we must already have all real factors included in S.
            # otherwise, such 'connected-to-all' trash variables will dominate the scene. So how to handle them?
            # Solution is to compute sum(X,Z) not only at the step of adding Z, but to repeat this procedure for all Zs once new X is added.
            # Maybe some Zs will render useless by adding that new X.
            # ---------------------------------------------------------------------------------------------------------------

            if cand_idx in partial_gains:
                current_gain, last_checked_k = partial_gains[cand_idx]
            else:
                current_gain = LARGE_CONST
                last_checked_k = -1

            stopped_early, current_gain, k, sink_reasons = evaluate_gain(
                current_gain=current_gain,
                last_checked_k=last_checked_k,
                direct_gain=direct_gain,
                X=X,
                y=y,
                nexisting=nexisting,
                best_gain=best_gain,
                factors_data=factors_data,
                factors_nbins=factors_nbins,
                selected_vars=selected_vars,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                extra_knowledge_multipler=extra_knowledge_multipler,
                sink_threshold=sink_threshold,
                entropy_cache=entropy_cache,
                cached_cond_MIs=cached_cond_MIs,
                can_use_x_cache=True,
                can_use_y_cache=True,
            )

            partial_gains[cand_idx] = current_gain, k
            if not stopped_early:  # there was no break. current_gain computed fully.
                expected_gains[cand_idx] = current_gain
        else:
            # no factors selected yet. current_gain is just direct_gain
            current_gain = direct_gain
            expected_gains[cand_idx] = current_gain
    else:
        current_gain = 0

    return current_gain, sink_reasons


def test(a):
    return 0


# @njit()
def screen_predictors(
    # factors
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str] = None,
    factors_to_use: Sequence[int] = None,
    # targets
    targets_data: np.ndarray = None,
    targets_nbins: Sequence[int] = None,
    y: Sequence[int] = None,
    targets: Union[dict, Sequence[Sequence]] = None,
    # algorithm
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    reduce_gain_on_subelement_chosen: bool = True,
    # performance
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    random_seed: int = None,
    use_gpu: bool = False,
    nworkers: int = 1,
    # confidence
    min_occupancy: int = None,
    min_nonzero_confidence: float = 0.99,
    full_npermutations: int = 1_000,
    baseline_npermutations: int = 100,
    # stopping conditions
    min_relevance_gain: float = 0.00001,
    max_consec_unconfirmed: int = 10,
    max_runtime_mins: int = None,
    interactions_min_order: int = 1,
    interactions_max_order: int = 1,
    interactions_order_reversed: bool = False,
    max_veteranes_interactions_order: int = 1,
    only_unknown_interactions: bool = False,
    # verbosity and formatting
    verbose: int = 1,
    ndigits: int = 5,
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

    assert mrmr_relevance_algo in ("fleuret", "pld")
    assert mrmr_redundancy_algo in ("fleuret", "pld_max", "pld_mean")

    assert len(factors_data) >= 10
    if targets_data is None:
        targets_data = factors_data
    else:
        assert len(factors_data) == len(targets_data)

    if targets_nbins is None:
        targets_nbins = factors_nbins

    assert targets_data.shape[1] == len(targets_nbins)
    assert factors_data.shape[1] == len(factors_nbins)

    if len(factors_names) == 0:
        factors_names = ["F" + str(i) for i in range(len(factors_data))]
    else:
        assert factors_data.shape[1] == len(factors_names)

    # warn if inputs are identical to targets
    if factors_data.shape == targets_data.shape:
        if np.shares_memory(factors_data, targets_data):
            if factors_to_use is None:

                if verbose > 1:
                    logger.info(
                        "factors_data and targets_data share the same memory. factors_to_use will be determined automatically to not contain any target columns."
                    )
                x = set(range(factors_data.shape[1])) - set(y)
        else:

            assert not set(y).issubset(set(x))

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    start_time = timer()
    run_out_of_time = False

    if random_seed is not None:
        np.random.seed(random_seed)
        cp.random.seed(random_seed)
        set_random_seed(random_seed)

    max_failed = int(full_npermutations * (1 - min_nonzero_confidence))
    if max_failed <= 1:
        max_failed = 1

    selected_interactions_vars = []
    selected_vars = []  # stores just indices. can't use set 'cause the order is important for efficient computing
    predictors = []  # stores more details.

    cached_MIs = dict()
    # cached_cond_MIs = dict()
    cached_confident_MIs = dict()
    entropy_cache = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    cached_cond_MIs = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    data_copy = factors_data.copy()

    classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
    classes_y_safe = classes_y.copy()

    if use_gpu:
        classes_y_safe = cp.asarray(classes_y.astype(np.int32))
        freqs_y_safe = cp.asarray(freqs_y)
    else:
        freqs_y_safe = None

    if nworkers and nworkers > 1:
        #    global classes_y_memmap
        #    classes_y_memmap = mem_map_array(obj=classes_y, file_name="classes_y", mmap_mode="r")
        if verbose:
            logger.info("Starting parallel pool...")
        workers_pool = Parallel(n_jobs=nworkers)  # , max_nbytes=MAX_JOBLIB_NBYTES
        workers_pool(delayed(test)(i) for i in range(nworkers))
    else:
        workers_pool = None

    subsets = range(interactions_min_order, interactions_max_order + 1)
    if interactions_order_reversed:
        subsets = subsets[::-1]

    if verbose:
        logger.info(
            f"Starting work with full_npermutations={full_npermutations:_}, min_nonzero_confidence={min_nonzero_confidence:.{ndigits}f}, max_failed={max_failed:_}"
        )

    for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Interactions order", leave=False)):

        if run_out_of_time:
            break
        subsets_pbar.set_description(f"{interactions_order}-way interactions")

        # ---------------------------------------------------------------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------------------------------------------------------------

        candidates = [tuple(el) for el in combinations(x, interactions_order)]

        # ---------------------------------------------------------------------------------------------------------------
        # Subset level inits
        # ---------------------------------------------------------------------------------------------------------------

        total_disproved = 0
        total_checked = 0
        partial_gains = {}
        added_candidates = set()
        failed_candidates = set()
        min_gain_reached = False
        nconsec_unconfirmed = 0

        for _ in (predictors_pbar := tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors")):

            if run_out_of_time:
                break

            # ---------------------------------------------------------------------------------------------------------------
            # Find candidate X with the highest current_gain given already selected factors
            # ---------------------------------------------------------------------------------------------------------------

            best_candidate = None
            best_gain = min_relevance_gain - 1
            expected_gains = np.zeros(len(candidates), dtype=np.float64)

            while True:  # confirmation loop (by random permutations)

                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    eval_start = timer()

                feasible_candidates = []
                for cand_idx, X in enumerate(candidates):
                    skip, nexisting = should_skip_candidate(
                        cand_idx=cand_idx,
                        X=X,
                        interactions_order=interactions_order,
                        only_unknown_interactions=only_unknown_interactions,
                        failed_candidates=failed_candidates,
                        added_candidates=added_candidates,
                        expected_gains=expected_gains,
                        selected_vars=selected_vars,
                        selected_interactions_vars=selected_interactions_vars,
                    )
                    if skip:
                        continue

                    feasible_candidates.append((cand_idx, X, nexisting if reduce_gain_on_subelement_chosen else 0))

                if nworkers and nworkers > 1 and len(feasible_candidates) > NMAX_NONPARALLEL_ITERS:

                    res = workers_pool(
                        delayed(evaluate_candidates)(
                            workload=workload,  # cand_idx=cand_idx,X=X,
                            y=y,
                            best_gain=best_gain,
                            factors_data=factors_data,
                            factors_nbins=factors_nbins,
                            factors_names=factors_names,
                            classes_y=classes_y,
                            freqs_y=freqs_y,
                            use_gpu=use_gpu,
                            freqs_y_safe=freqs_y_safe,
                            partial_gains=partial_gains,
                            baseline_npermutations=baseline_npermutations,
                            mrmr_relevance_algo=mrmr_relevance_algo,
                            mrmr_redundancy_algo=mrmr_redundancy_algo,
                            max_veteranes_interactions_order=max_veteranes_interactions_order,
                            selected_vars=selected_vars,
                            cached_MIs=cached_MIs,
                            cached_confident_MIs=cached_confident_MIs,
                            cached_cond_MIs=dict(cached_cond_MIs),
                            entropy_cache=dict(entropy_cache),
                            max_runtime_mins=max_runtime_mins,
                            start_time=start_time,
                            min_relevance_gain=min_relevance_gain,
                            verbose=verbose,
                            ndigits=ndigits,
                        )
                        for workload in split_list_into_chunks(feasible_candidates, len(feasible_candidates) // nworkers)
                    )

                    for (
                        worker_best_gain,
                        worker_best_candidate,
                        worker_partial_gains,
                        worker_expected_gains,
                        worker_cached_MIs,
                        worker_cached_cond_MIs,
                        worker_entropy_cache,
                    ) in res:

                        if worker_best_gain > best_gain:
                            best_candidate = worker_best_candidate
                            best_gain = worker_best_gain

                        # sync caches
                        for local_storage, global_storage in [
                            (worker_expected_gains, expected_gains),
                            (worker_cached_MIs, cached_MIs),
                            (worker_cached_cond_MIs, cached_cond_MIs),
                            (worker_entropy_cache, entropy_cache),
                        ]:
                            for key, value in local_storage.items():
                                global_storage[key] = value

                        for cand_idx, (worker_current_gain, worker_z_idx) in worker_partial_gains.items():
                            if cand_idx in partial_gains:
                                current_gain, z_idx = partial_gains[cand_idx]
                            else:
                                z_idx = -2
                            if worker_z_idx > z_idx:
                                partial_gains[cand_idx] = (worker_current_gain, worker_z_idx)

                    if max_runtime_mins and not run_out_of_time:
                        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60
                        if run_out_of_time:
                            logging.info(f"Time limit exhausted. Finalizing the search...")
                            break

                else:
                    for cand_idx, X, nexisting in (candidates_pbar := tqdmu(feasible_candidates, leave=False, desc="Candidates")):

                        current_gain, sink_reasons = evaluate_candidate(
                            cand_idx=cand_idx,
                            X=X,
                            y=y,
                            nexisting=nexisting,
                            best_gain=best_gain,
                            factors_data=factors_data,
                            factors_nbins=factors_nbins,
                            factors_names=factors_names,
                            classes_y=classes_y,
                            classes_y_safe=classes_y_safe,
                            freqs_y=freqs_y,
                            use_gpu=use_gpu,
                            freqs_y_safe=freqs_y_safe,
                            partial_gains=partial_gains,
                            baseline_npermutations=baseline_npermutations,
                            mrmr_relevance_algo=mrmr_relevance_algo,
                            mrmr_redundancy_algo=mrmr_redundancy_algo,
                            max_veteranes_interactions_order=max_veteranes_interactions_order,
                            expected_gains=expected_gains,
                            selected_vars=selected_vars,
                            cached_MIs=cached_MIs,
                            cached_confident_MIs=cached_confident_MIs,
                            cached_cond_MIs=cached_cond_MIs,
                            entropy_cache=entropy_cache,
                            verbose=verbose,
                            ndigits=ndigits,
                        )

                        best_gain, best_candidate, run_out_of_time = handle_best_candidate(
                            current_gain=current_gain,
                            best_gain=best_gain,
                            X=X,
                            best_candidate=best_candidate,
                            factors_names=factors_names,
                            max_runtime_mins=max_runtime_mins,
                            start_time=start_time,
                            min_relevance_gain=min_relevance_gain,
                            verbose=verbose,
                            ndigits=ndigits,
                        )

                        if run_out_of_time:
                            if verbose:
                                logging.info(f"Time limit exhausted. Finalizing the search...")
                            break

                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    logger.info(f"evaluate_candidates took {timer() - eval_start:.1f} sec.")

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

                            X = candidates[next_best_candidate_idx]

                            # ---------------------------------------------------------------------------------------------------------------
                            # For cands other than the top one, if best partial gain <= next_best_gain, we can proceed with confirming next_best_gain. else we have to recompute partial gains
                            # ---------------------------------------------------------------------------------------------------------------

                            if n > 0:
                                best_partial_gain, best_key = find_best_partial_gain(
                                    partial_gains=partial_gains,
                                    failed_candidates=failed_candidates,
                                    added_candidates=added_candidates,
                                    candidates=candidates,
                                    selected_vars=selected_vars,
                                )

                                if best_partial_gain > next_best_gain:
                                    if verbose > 1:
                                        print(
                                            "Have no best_candidate anymore. Need to recompute partial gains. best_partial_gain of candidate",
                                            get_candidate_name(candidates[best_key], factors_names=factors_names),
                                            "was",
                                            best_partial_gain,
                                        )
                                    break  # out of best candidates confirmation, to retry all cands evaluation

                            any_cand_considered = True
                            if verbose > 1:
                                logger.info(
                                    f"confirming candidate {get_candidate_name(X, factors_names=factors_names)}, next_best_gain={next_best_gain:.{ndigits}f}"
                                )

                            # ---------------------------------------------------------------------------------------------------------------
                            # Compute confidence by bootstrap
                            # ---------------------------------------------------------------------------------------------------------------

                            total_checked += 1
                            if X in cached_confident_MIs:
                                bootstrapped_gain, confidence = cached_confident_MIs[X]
                            else:
                                if use_gpu:
                                    bootstrapped_gain, confidence = mi_direct_gpu(
                                        factors_data,
                                        x=X,
                                        y=y,
                                        factors_nbins=factors_nbins,
                                        classes_y=classes_y,
                                        freqs_y=freqs_y,
                                        freqs_y_safe=freqs_y_safe,
                                        classes_y_safe=classes_y_safe,
                                        min_nonzero_confidence=min_nonzero_confidence,
                                        npermutations=full_npermutations,
                                    )
                                else:
                                    if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                        eval_start = timer()
                                    bootstrapped_gain, confidence = mi_direct(
                                        factors_data,
                                        x=X,
                                        y=y,
                                        factors_nbins=factors_nbins,
                                        classes_y=classes_y,
                                        freqs_y=freqs_y,
                                        classes_y_safe=classes_y_safe,
                                        min_nonzero_confidence=min_nonzero_confidence,
                                        npermutations=full_npermutations,
                                        nworkers=nworkers,
                                        workers_pool=workers_pool,
                                    )
                                    if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                        logger.info(f"mi_direct bootstrapped eval took {timer() - eval_start:.1f} sec.")
                                cached_confident_MIs[X] = bootstrapped_gain, confidence

                            if bootstrapped_gain > 0 and selected_vars:  # additional check of Fleuret criteria
                                skip_cand = [(subel in selected_vars) for subel in X]
                                nexisting = sum(skip_cand)
                                # ---------------------------------------------------------------------------------------------------------------
                                # external bootstrapped recheck. is minimal MI of candidate X with Y given all current Zs THAT BIG as next_best_gain?
                                # ---------------------------------------------------------------------------------------------------------------
                                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                    eval_start = timer()

                                if nworkers and nworkers > 1 and full_npermutations > NMAX_NONPARALLEL_ITERS:
                                    bootstrapped_gain, confidence, parallel_entropy_cache = get_fleuret_criteria_confidence_parallel(
                                        data_copy=data_copy,
                                        factors_nbins=factors_nbins,
                                        x=X,
                                        y=y,
                                        selected_vars=selected_vars,
                                        bootstrapped_gain=next_best_gain,
                                        npermutations=full_npermutations,
                                        max_failed=max_failed,
                                        nexisting=nexisting,
                                        mrmr_relevance_algo=mrmr_relevance_algo,
                                        mrmr_redundancy_algo=mrmr_redundancy_algo,
                                        max_veteranes_interactions_order=max_veteranes_interactions_order,
                                        cached_cond_MIs=cached_cond_MIs,
                                        entropy_cache=entropy_cache,
                                        extra_x_shuffling=extra_x_shuffling,
                                        nworkers=nworkers,
                                        workers_pool=workers_pool,
                                    )
                                    for key, value in parallel_entropy_cache.items():
                                        entropy_cache[key] = value
                                else:
                                    nfailed, nchecked = get_fleuret_criteria_confidence(
                                        data_copy=data_copy,
                                        factors_nbins=factors_nbins,
                                        x=X,
                                        y=y,
                                        selected_vars=selected_vars,
                                        bootstrapped_gain=next_best_gain,
                                        npermutations=full_npermutations,
                                        max_failed=max_failed,
                                        nexisting=nexisting,
                                        mrmr_relevance_algo=mrmr_relevance_algo,
                                        mrmr_redundancy_algo=mrmr_redundancy_algo,
                                        max_veteranes_interactions_order=max_veteranes_interactions_order,
                                        cached_cond_MIs=cached_cond_MIs,
                                        entropy_cache=entropy_cache,
                                        extra_x_shuffling=extra_x_shuffling,
                                    )

                                    confidence = 1 - nfailed / nchecked
                                    if nfailed >= max_failed:
                                        bootstrapped_gain = 0.0

                                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                    logger.info(f"get_fleuret_criteria_confidence bootstrapped eval took {timer() - eval_start:.1f} sec.")
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
                                    if verbose > 1:
                                        logger.info(
                                            f"\t\tCandidate's lowered confidence {confidence} requires re-checking other candidates, as now its expected gain is only {next_best_gain:.{ndigits}f}, vs {best_partial_gain:.{ndigits}f}, of {get_candidate_name(candidates[best_key], factors_names=factors_names)}"
                                        )
                                    break  # out of best candidates confirmation, to retry all cands evaluation
                                else:
                                    cand_confirmed = True
                                    if verbose > 1:
                                        logger.info(f"\t\tconfirmed with confidence {confidence:.{ndigits}f}")
                                    break  # out of best candidates confirmation, to add candidate to the list, and go to more candidates
                            else:
                                expected_gains[next_best_candidate_idx] = 0.0
                                failed_candidates.add(next_best_candidate_idx)
                                if verbose > 1:
                                    logger.info(f"\t\tconfirmation failed with confidence {confidence:.{ndigits}f}")

                                nconsec_unconfirmed += 1
                                total_disproved += 1
                                if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                                    if verbose:
                                        logger.info(f"Maximum consecutive confirmation failures reached.")
                                    break  # out of best candidates confirmation, to finish the level

                        else:  # next_best_gain = 0
                            break  # nothing wrong, just retry all cands evaluation

                    # ---------------------------------------------------------------------------------------------------------------
                    # Let's act upon results of the permutation test
                    # ---------------------------------------------------------------------------------------------------------------

                    if cand_confirmed:
                        added_candidates.add(next_best_candidate_idx)  # so it won't be selected again
                        best_candidate = X
                        best_gain = next_best_gain
                        break  # exit confirmation while loop
                    else:
                        if not any_cand_considered:
                            best_gain = min_relevance_gain - 1
                            if verbose:
                                logger.info("No more candidates to confirm.")
                            break  # exit confirmation while loop
                        else:
                            if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                                best_gain = min_relevance_gain - 1
                                break  # exit confirmation while loop
                            else:
                                pass  # retry all cands evaluation
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
                cand_name = get_candidate_name(best_candidate, factors_names=factors_names)
                if verbose:
                    logger.info(f"Added new predictor {cand_name} to the list with expected gain={best_gain:.{ndigits}f}")
                predictors.append({"name": cand_name, "indices": best_candidate, "gain": best_gain, "confidence": confidence})
            else:
                if verbose:
                    if total_checked > 0:
                        details = f" Total candidates disproved: {total_disproved:_}/{total_checked:_} ({total_disproved*100/total_checked:.2f}%)"
                    else:
                        details = ""
                    logger.info(f"Can't add anything valuable anymore for interactions_order={interactions_order}.{details}")
                predictors_pbar.total = len(candidates)
                predictors_pbar.close()
                break

    # postprocess_candidates(selected_vars)
    # print(caching_hits_xyz, caching_hits_z, caching_hits_xz, caching_hits_yz)
    if verbose:
        logger.info(f"Finished.")

    return selected_vars, predictors


def find_best_partial_gain(
    partial_gains: dict, failed_candidates: set, added_candidates: set, candidates: list, selected_vars: list, skip_indices: tuple = ()
) -> float:
    best_partial_gain = -LARGE_CONST
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
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,
    min_nonzero_confidence: float = 0.99999,
    npermutations: int = 10_000,
    interactions_max_order: int = 1,
    ensure_target_influence: bool = True,
    dtype=np.int32,
    verbose: bool = True,
):
    """Post-analysis of prescreened candidates.

    1) repeat standard Fleuret screening process. maybe some vars will be removed when taken into account all other candidates.
    2)
    3) in the final set, compute for every factor
        a) MI with every remaining predictor (and 2,3 way subsets)

    """
    # ---------------------------------------------------------------------------------------------------------------
    # Repeat standard Fleuret screening process. maybe some vars will be removed when taken into account all other candidates.
    # ---------------------------------------------------------------------------------------------------------------
    for cand_idx, X, nexisting in (candidates_pbar := tqdmu(selected_vars, leave=False, desc="Finalizing Candidates")):
        current_gain, sink_reasons = evaluate_candidate(
            cand_idx=cand_idx,
            X=X,
            y=y,
            nexisting=nexisting,
            best_gain=best_gain,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=factors_names,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            use_gpu=use_gpu,
            freqs_y_safe=freqs_y_safe,
            partial_gains=partial_gains,
            baseline_npermutations=baseline_npermutations,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            expected_gains=expected_gains,
            selected_vars=selected_vars,
            cached_MIs=cached_MIs,
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs,
            entropy_cache=entropy_cache,
            verbose=verbose,
            ndigits=ndigits,
        )
    # ---------------------------------------------------------------------------------------------------------------
    # Make sure with confidence that every candidate is related to the target
    # ---------------------------------------------------------------------------------------------------------------

    if ensure_target_influence:
        removed = []
        for X in tqdmu(selected_vars, desc="Ensuring target influence", leave=False):
            bootstrapped_mi, confidence = mi_direct(
                factors_data,
                x=[X],
                y=y,
                factors_nbins=factors_nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                classes_y_safe=classes_y_safe,
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
        _, freqs, _ = merge_vars(factors_data=factors_data, vars_indices=[X], factors_nbins=factors_nbins, var_is_nominal=None, dtype=dtype)
        factor_entropy = entropy(freqs=freqs)
        entropies[X] = factor_entropy

    for a, b in tqdmu(combinations(selected_vars, 2), desc="1-way interactions", leave=False):
        bootstrapped_mi, confidence = mi_direct(
            factors_data,
            x=[a],
            y=[b],
            factors_nbins=factors_nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            classes_y_safe=classes_y_safe,
            min_nonzero_confidence=min_nonzero_confidence,
            npermutations=npermutations,
        )
        if bootstrapped_mi > 0:
            mutualinfos[(a, b)] = bootstrapped_mi

    for y in tqdmu(selected_vars, desc="2-way interactions", leave=False):
        for pair in combinations(set(selected_vars) - set([y]), 2):
            bootstrapped_mi, confidence = mi_direct(
                factors_data,
                x=pair,
                y=[y],
                factors_nbins=factors_nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                classes_y_safe=classes_y_safe,
                min_nonzero_confidence=min_nonzero_confidence,
                npermutations=npermutations,
            )
            if bootstrapped_mi > 0:
                mutualinfos[(y, pair)] = bootstrapped_mi

    return entropies, mutualinfos
