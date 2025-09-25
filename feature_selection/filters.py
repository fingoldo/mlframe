"""Feature selection within ML pipelines. Based on filters. Currently includes mRMR."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

while True:
    try:
        # ----------------------------------------------------------------------------------------------------------------------------
        # Normal Imports
        # ----------------------------------------------------------------------------------------------------------------------------

        from typing import *

        import copy
        import psutil
        import textwrap
        from collections import defaultdict
        from timeit import default_timer as timer

        import pandas as pd, numpy as np
        from itertools import combinations
        from os.path import exists
        import math
        import os
        import gc

        from pyutilz.system import tqdmu
        from pyutilz.pythonlib import sort_dict_by_value
        from pyutilz.pythonlib import store_params_in_object, get_parent_func_args
        from pyutilz.parallel import mem_map_array, split_list_into_chunks, parallel_run
        from pyutilz.numbalib import set_numba_random_seed, arr2str, python_dict_2_numba_dict, generate_combinations_recursive_njit

        # from mlframe.boruta_shap import BorutaShap

        from scipy.stats import mode
        from scipy import special as sp
        from astropy.stats import histogram
        from numpy.polynomial.hermite import hermval

        from catboost import CatBoostClassifier
        from sklearn.metrics import make_scorer

        from mlframe.arrays import arrayMinMax
        from mlframe.utils import set_random_seed
        from mlframe.feature_selection.wrappers import RFECV
        from mlframe.metrics import compute_probabilistic_multiclass_error

        from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
        from sklearn.model_selection import KFold
        from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
        from sklearn.impute import SimpleImputer
        from itertools import combinations

        from numba.core import types
        from numba import njit, jit
        import numba

        from joblib import Parallel, delayed

        from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
        import warnings

    except ModuleNotFoundError as e:

        logger.warning(e)

        if "cannot import name" in str(e):
            raise (e)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Packages auto-install
        # ----------------------------------------------------------------------------------------------------------------------------

        from pyutilz.pythonlib import ensure_installed

        ensure_installed("numba numpy pandas scipy astropy scikit-learn joblib catboost psutil")  # cupy-cuda11x

    else:
        break

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings("ignore", module=".*_discretization")
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

MAX_JOBLIB_NBYTES = 1e3
NMAX_NONPARALLEL_ITERS = 2
MAX_ITERATIONS_TO_TRACK = 5

LARGE_CONST: float = 1e30
GPU_MAX_BLOCK_SIZE: int = 1024
MAX_CONFIRMATION_CAND_NBINS: int = 50

caching_hits_xyz = 0
caching_hits_z = 0
caching_hits_xz = 0
caching_hits_yz = 0


def init_kernels():

    global compute_joint_hist_cuda, compute_mi_from_classes_cuda

    import cupy as cp  # pip install cupy-cuda11x; python -m cupyx.tools.install_library --cuda 11.x --library cutensor

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
def unpack_and_sort(x, z):
    res = []
    for el in x:
        res.append(el)
    for el in z:
        res.append(el)
    return sorted(res)


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

        # indices = sorted([*x, *z])
        indices = unpack_and_sort(x, z)

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

            # indices = sorted([*y, *z])
            indices = unpack_and_sort(y, z)

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
            factors_data=factors_data, vars_indices=unpack_and_sort(y, z), var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype  # [*y, *z]
        )  # always 2-dim
        entropy_yz = entropy(freqs=freqs_yz)

    if entropy_xyz < 0:
        if can_use_y_cache and can_use_x_cache:

            # indices = sorted([*x, *y, *z])
            indices = unpack_and_sort(x, y)
            indices = unpack_and_sort(indices, z)

            if entropy_cache is not None:
                key = arr2str(indices)
                entropy_xyz = entropy_cache.get(key, -1)
        if entropy_xyz < 0:
            if current_nclasses_yz == 1:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
                    factors_data=factors_data, vars_indices=unpack_and_sort(y, z), var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype  # [*y, *z]
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
def distribute_permutations(npermutations: int, n_workers: int) -> list:

    avg_perms_per_worker = npermutations // n_workers
    diff = npermutations - avg_perms_per_worker * n_workers
    workload = [avg_perms_per_worker] * n_workers
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
    classes_y_safe: np.ndarray = None,
    freqs_y: np.ndarray = None,
    n_workers: int = 1,
    workers_pool: object = None,
    parallel_kwargs: dict = {},
) -> tuple:

    classes_x, freqs_x, _ = merge_vars(factors_data=factors_data, vars_indices=x, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)

    original_mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)

    # logger.info(f"original_mi={original_mi}")

    if original_mi > 0 and npermutations > 0:

        # Inits

        nfailed = 0
        if not max_failed:
            max_failed = int(npermutations * (1 - min_nonzero_confidence))
            if max_failed <= 1:
                max_failed = 1

        if n_workers and n_workers > 1 and npermutations > NMAX_NONPARALLEL_ITERS:

            # logger.info("Memmapping classes_x...")
            # classes_x_memmap = mem_map_array(obj=classes_x, file_name="classes_x", mmap_mode="r")

            if workers_pool is None:
                workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)

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
                for worker_npermutations in distribute_permutations(npermutations=npermutations, n_workers=n_workers)
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

            # Create a Generator instance with a specific RNG state
            # seed=int.from_bytes(os.urandom(4), byteorder='little')
            # rng = np.random.default_rng(seed)

            for i in range(npermutations):
                # logger.info(f"x={x} perm {i}")
                # np.random.shuffle(classes_y_safe)
                shuffle_arr(classes_y_safe)
                # Shuffle the array using the local RNG
                # classes_y_shuffled=np.random.choice(classes_y, len(classes_y), replace=False)
                # rng.shuffle(classes_y_safe)
                # logger.info(f"x={x} shuffled")
                # mi = compute_mi_from_classes(classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y_shuffled, freqs_y=freqs_y, dtype=dtype)
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
def shuffle_arr(arr: np.ndarray) -> None:
    np.random.shuffle(arr)


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
    classes_y_safe: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
) -> tuple:

    import cupy as cp

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
        nfailed = 0
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
    n_workers: int = 1,
    workers_pool: object = None,
    parallel_kwargs: dict = {},
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
) -> tuple:

    nfailed = 0

    if workers_pool is None:
        workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)

    gc.collect()
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
        for worker_npermutations in distribute_permutations(npermutations=npermutations, n_workers=n_workers)
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
    max_runtime_mins: float = None,
    start_time: float = None,
    min_relevance_gain: float = None,
    verbose: int = 1,
    ndigits: int = 5,
    use_simple_mode: bool = True,
) -> None:

    best_gain = -LARGE_CONST
    best_candidate = None
    expected_gains = {}

    from pyutilz.logginglib import init_logging

    global logger
    logger = init_logging(default_caller_name="scalping.py", format="%(asctime)s - %(levelname)s - %(funcName)s-line:%(lineno)d - %(message)s")

    # if verbose: logger.info("In evaluate_candidates")

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
    # np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    for cand_idx, X, nexisting in (candidates_pbar := tqdmu(workload, leave=False, desc="Thread Candidates")):

        # if verbose: logger.info(f"Evaluating cand {X}")

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
            use_simple_mode=use_simple_mode,
        )

        # if verbose: logger.info(f"X={X}, gain={current_gain}")

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

        """
        if use_simple_mode:
            if best_gain > 0:
                break
        """

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
    max_runtime_mins: float = None,
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
        if verbose > 2:
            logger.info(
                f"\t{get_candidate_name(best_candidate,factors_names=factors_names)} is so far the best candidate with best_gain={best_gain:.{ndigits}f}"
            )
    else:
        if min_relevance_gain and verbose > 2 and current_gain > min_relevance_gain:
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
        combs = generate_combinations_recursive_njit(np.array(selected_vars, dtype=np.int32), interactions_order + 1)[::-1]
        # if X==(425,): logger.info(f"\t combs={combs}")

        for Z in combs:

            if k > last_checked_k:
                if confidence_mode and count_cand_nbins(Z, factors_nbins) > MAX_CONFIRMATION_CAND_NBINS:
                    additional_knowledge = 0.0  # this is needed to skip checking agains hi cardinality approved factors
                else:
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
                                # if X==(425,): logger.info(f"\t additional_knowledge from {Z} found to be {additional_knowledge}, k={k}, last_checked_k={last_checked_k}")
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
                            # else:
                            #    if len(X) > 1:
                            #        additional_knowledge = additional_knowledge ** (len(X) + 1)

                            if not confidence_mode:
                                cached_cond_MIs[key] = additional_knowledge

                            # if X==(425,): logger.info(f"\t additional_knowledge from {Z}={additional_knowledge}, k={k}, last_checked_k={last_checked_k}")

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

    return stopped_early, current_gain, k - 1, sink_reasons


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
    use_simple_mode: bool = True,
) -> None:
    # logger.info("In evaluate_candidate")
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
                # logger.info("Computing mi_direct")
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
                # logger.info("Computed mi_direct")
            cached_MIs[X] = direct_gain

    if direct_gain > 0:
        if selected_vars and not use_simple_mode:

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
                # if X==(425,): logger.info(f"\t cand_idx in partial_gains: {current_gain, last_checked_k}")
                if best_gain is not None and (current_gain <= best_gain):
                    return current_gain, sink_reasons
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
            # if X==(425,): logger.info(f"\t stopped_early, current_gain, k, sink_reasons={stopped_early, current_gain, k, sink_reasons}")

            partial_gains[cand_idx] = current_gain, k
            if not stopped_early:  # there was no break. current_gain computed fully. this line was (and most likely should be) commented out.
                expected_gains[cand_idx] = current_gain
        else:
            # no factors selected yet. current_gain is just direct_gain
            current_gain = direct_gain
            expected_gains[cand_idx] = current_gain
    else:
        current_gain = 0

    return current_gain, sink_reasons


def test(a):
    logger.info("test")
    return 0


# @njit()
def screen_predictors(
    # factors
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str] = None,
    factors_names_to_use: Sequence[str] = None,
    factors_to_use: Sequence[int] = None,
    # targets
    targets_data: np.ndarray = None,
    targets_nbins: Sequence[int] = None,
    y: Sequence[int] = None,
    # algorithm
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    reduce_gain_on_subelement_chosen: bool = True,
    # performance
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    random_seed: int = None,
    use_gpu: bool = False,
    n_workers: int = 1,
    # confidence
    min_occupancy: int = None,
    min_nonzero_confidence: float = 0.99,
    full_npermutations: int = 1_000,
    baseline_npermutations: int = 100,
    # stopping conditions
    min_relevance_gain: float = 0.00001,
    max_consec_unconfirmed: int = 30,
    max_runtime_mins: float = None,
    interactions_min_order: int = 1,
    interactions_max_order: int = 1,
    interactions_order_reversed: bool = False,
    max_veteranes_interactions_order: int = 1,
    only_unknown_interactions: bool = False,
    # verbosity and formatting
    verbose: int = 1,
    ndigits: int = 5,
    parallel_kwargs=dict(max_nbytes=MAX_JOBLIB_NBYTES),
    stop_file: str = None,
    use_simple_mode: bool = True,
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
            if factors_to_use is None and factors_names_to_use is None:
                if verbose > 2:
                    logger.info(
                        "factors_data and targets_data share the same memory. factors_to_use will be determined automatically to not contain any target columns."
                    )
                x = set(range(factors_data.shape[1])) - set(y)
            else:
                if factors_to_use is not None:
                    x = set(factors_to_use) - set(y)
                    if verbose > 2:
                        logger.info(f"Using only {len(factors_to_use):_} predefined factors: {factors_to_use}")
                else:
                    x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use and i != y]
                    if verbose > 2:
                        logger.info(f"Using only {len(factors_names_to_use):_} predefined factors: {factors_names_to_use}")
        else:

            assert not set(y).issubset(set(x))

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    start_time = timer()
    run_out_of_time = False

    if random_seed is not None:
        np.random.seed(random_seed)
        set_numba_random_seed(random_seed)
        try:
            cp.random.seed(random_seed)
        except Exception as e:
            pass

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
        import cupy as cp

        classes_y_safe = cp.asarray(classes_y.astype(np.int32))
        freqs_y_safe = cp.asarray(freqs_y)
    else:
        freqs_y_safe = None

    if n_workers and n_workers > 1:
        #    global classes_y_memmap
        #    classes_y_memmap = mem_map_array(obj=classes_y, file_name="classes_y", mmap_mode="r")
        if verbose >= 2:
            logger.info("Starting parallel pool...")
        workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)
        workers_pool(delayed(test)(i) for i in range(n_workers))
    else:
        workers_pool = None

    subsets = range(interactions_min_order, interactions_max_order + 1)
    if interactions_order_reversed:
        subsets = subsets[::-1]

    if verbose >= 2:
        logger.info(
            f"Starting work with full_npermutations={full_npermutations:_}, min_nonzero_confidence={min_nonzero_confidence:.{ndigits}f}, max_failed={max_failed:_}"
        )

    num_possible_candidates = 0  # needed to refrain from multiprocessing when all direct MIs are in cache already

    for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Interactions order", leave=False)):

        if run_out_of_time:
            break
        subsets_pbar.set_description(f"{interactions_order}-way interactions")

        # ---------------------------------------------------------------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------------------------------------------------------------

        candidates = [tuple(el) for el in combinations(x, interactions_order)]

        num_possible_candidates += len(candidates)

        # ---------------------------------------------------------------------------------------------------------------
        # Subset level inits
        # ---------------------------------------------------------------------------------------------------------------

        total_disproved = 0
        total_checked = 0
        partial_gains = {}
        added_candidates = set()
        failed_candidates = set()
        nconsec_unconfirmed = 0

        for n_confirmed_predictors in (predictors_pbar := tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors")):
            # if n_confirmed_predictors>4: n_jobs=1
            if run_out_of_time:
                break
            if stop_file and exists(stop_file):
                logger.warning(f"Stop file {stop_file} detected, quitting.")
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

                if (
                    n_workers > 1
                    and (use_simple_mode is False or len(cached_MIs) < num_possible_candidates)
                    and len(feasible_candidates) > NMAX_NONPARALLEL_ITERS
                ):

                    res = workers_pool(
                        delayed(evaluate_candidates)(
                            workload=workload,
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
                            use_simple_mode=use_simple_mode,
                        )
                        for workload in split_list_into_chunks(feasible_candidates, max(1, len(feasible_candidates) // n_workers))
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

                    if use_simple_mode:
                        # need to sort all cands by perfs
                        pass
                    if max_runtime_mins and not run_out_of_time:
                        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60
                        if run_out_of_time:
                            logging.info(f"Time limit exhausted. Finalizing the search...")
                            break

                else:
                    if use_simple_mode and False:
                        # No need to check every can out of order: let's just return next best known candidate
                        best_gain, best_candidate, run_out_of_time = 1, 1, False
                    else:
                        for cand_idx, X, nexisting in feasible_candidates:  # (candidates_pbar := tqdmu(, leave=False, desc="Candidates"))

                            # tmp_idx=X[0]
                            # print(X,factors_nbins[tmp_idx],factors_names[tmp_idx])
                            # from time import sleep
                            # sleep(5)

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
                                use_simple_mode=use_simple_mode,
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

                            """
                            if use_simple_mode:
                                if best_gain > 0:
                                    break
                            """

                            if run_out_of_time:
                                if verbose:
                                    logging.info(f"Time limit exhausted. Finalizing the search...")
                                break

                if verbose > 2 and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    logger.info(f"evaluate_candidates took {timer() - eval_start:.1f} sec.")

                if best_gain < min_relevance_gain:
                    if verbose >= 2:
                        logger.info("Minimum expected gain reached or no candidates to check anymore.")
                    break  # exit confirmation while loop

                # ---------------------------------------------------------------------------------------------------------------
                # Now need to confirm best expected gain with a permutation test
                # ---------------------------------------------------------------------------------------------------------------

                cand_confirmed = False
                any_cand_considered = False
                for n, next_best_candidate_idx in enumerate(np.argsort(expected_gains)[::-1]):
                    next_best_gain = expected_gains[next_best_candidate_idx]
                    # logger.info(f"{n}, {next_best_gain}, {min_relevance_gain}")
                    if next_best_gain >= min_relevance_gain:  # only can consider here candidates fully checked against every Z

                        X = candidates[next_best_candidate_idx]

                        # ---------------------------------------------------------------------------------------------------------------
                        # For cands other than the top one, if best partial gain <= next_best_gain, we can proceed with confirming next_best_gain.
                        # else we have to recompute partial gains
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
                                best_gain = next_best_gain
                                if verbose > 2:
                                    print(
                                        "Have no best_candidate anymore. Need to recompute partial gains. best_partial_gain of candidate",
                                        get_candidate_name(candidates[best_key], factors_names=factors_names),
                                        "was",
                                        best_partial_gain,
                                    )
                                break  # out of best candidates confirmation, to retry all cands evaluation

                        any_cand_considered = True

                        if full_npermutations:

                            if verbose > 2:
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
                                        n_workers=n_workers,
                                        workers_pool=workers_pool,
                                        parallel_kwargs=parallel_kwargs,
                                    )
                                    if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                        logger.info(f"mi_direct bootstrapped eval took {timer() - eval_start:.1f} sec.")
                                cached_confident_MIs[X] = bootstrapped_gain, confidence
                        else:
                            if X in cached_confident_MIs:
                                bootstrapped_gain, confidence = cached_confident_MIs[X]
                            else:
                                bootstrapped_gain, confidence = next_best_gain, 1.0

                        if full_npermutations and bootstrapped_gain > 0 and selected_vars and not use_simple_mode:  # additional check of Fleuret criteria

                            if count_cand_nbins(X, factors_nbins) <= MAX_CONFIRMATION_CAND_NBINS:

                                skip_cand = [(subel in selected_vars) for subel in X]
                                nexisting = sum(skip_cand)

                                # ---------------------------------------------------------------------------------------------------------------
                                # external bootstrapped recheck. is minimal MI of candidate X with Y given all current Zs THAT BIG as next_best_gain?
                                # ---------------------------------------------------------------------------------------------------------------

                                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                    eval_start = timer()

                                if n_workers and n_workers > 1 and full_npermutations > NMAX_NONPARALLEL_ITERS:
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
                                        n_workers=n_workers,
                                        workers_pool=workers_pool,
                                        parallel_kwargs=parallel_kwargs,
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
                                    # logger.info(f"nfailed={nfailed}, nchecked={nchecked}")
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
                                if verbose > 2:
                                    logger.info(
                                        f"\t\tCandidate's lowered confidence {confidence} requires re-checking other candidates, as now its expected gain is only {next_best_gain:.{ndigits}f}, vs {best_partial_gain:.{ndigits}f}, of {get_candidate_name(candidates[best_key], factors_names=factors_names)}"
                                    )
                                break  # out of best candidates confirmation, to retry all cands evaluation
                            else:
                                cand_confirmed = True
                                if full_npermutations:
                                    if verbose > 2:
                                        logger.info(f"\t\tconfirmed with confidence {confidence:.{ndigits}f}")
                                break  # out of best candidates confirmation, to add candidate to the list, and go to more candidates
                        else:
                            expected_gains[next_best_candidate_idx] = 0.0
                            failed_candidates.add(next_best_candidate_idx)
                            if verbose > 2:
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
                        best_gain = min_relevance_gain - 1
                        if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                            break  # exit confirmation while loop
                        else:
                            pass  # retry all cands evaluation

            # ---------------------------------------------------------------------------------------------------------------
            # Add best candidate to the list, if criteria are met, or proceed to the next interactions_order
            # ---------------------------------------------------------------------------------------------------------------

            if best_gain >= (min_relevance_gain if interactions_order == 1 else min_relevance_gain ** (1 / (interactions_order + 1))):
                for var in best_candidate:
                    if var not in selected_vars:
                        selected_vars.append(var)
                        if interactions_order > 1:
                            selected_interactions_vars.append(var)
                cand_name = get_candidate_name(best_candidate, factors_names=factors_names)

                res = {"name": cand_name, "indices": best_candidate, "gain": best_gain}
                if full_npermutations:
                    res["confidence"] = confidence
                predictors.append(res)

                if verbose >= 2:
                    mes = f"Added new predictor {cand_name} to the list with expected gain={best_gain:.{ndigits}f}"
                    if full_npermutations:
                        mes += f" and confidence={confidence:.3f}"
                    logger.info(mes)

            else:
                if verbose >= 2:
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
    if verbose >= 2:
        logger.info(f"Finished.")

    any_influencing = set()
    for vars_combination, (bootstrapped_gain, confidence) in cached_confident_MIs.items():
        if bootstrapped_gain > 0:
            any_influencing.update(set(vars_combination))

    """Выбрать группы/кластера скоррелированных факторов. Вместо использования 1 самого крутого, рассмотреть средние от всех
        отброшенных факторов, имеющих высокое прямое direct_MI с таргетом, но близкое к 0 additional_knowledge с каждым 
        "победившим" фактором, проверить, не могут ли они усилить свой победивший фактор через ансамблирование. Усиление в смысле
        среднего и вариативности MI с таргетом на бутстрепе подвыборок?

        key = arr2str(X) + "_" + arr2str(Z)
        if key in cached_cond_MIs:
            additional_knowledge = cached_cond_MIs[key]                        
    """
    return selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe, freqs_y


@njit()
def count_cand_nbins(X, factors_nbins) -> int:
    sum_cand_nbins = 0
    for factor in X:
        sum_cand_nbins += factors_nbins[factor]
    return sum_cand_nbins


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
    ndigits: int = 4,
):
    """Post-analysis of prescreened candidates.

    1) repeat standard Fleuret screening process. maybe some vars will be removed when taken into account all other candidates.
    2)
    3) in the final set, compute for every factor
        a) MI with every remaining predictor (and 2,3 way subsets)

    """

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
    """


# ----------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------


def create_redundant_continuous_factor(
    df: pd.DataFrame,
    factors: Sequence[str],
    agg_func: object = np.sum,
    noise_percent: float = 5.0,
    dist: object = None,
    dist_args: tuple = (),
    name: str = None,
    sep: str = "_",
) -> None:
    """In a pandas dataframe, out of a few continuous factors, craft a new factor with known relationship and amount of redundancy."""
    if dist:
        rvs = getattr(dist, "rvs")
        assert rvs
        noise = rvs(*dist_args, size=len(df))
    else:
        noise = np.random.random(len(df))

    # now the entire range of generated noise is scaled to the noise_percent of factors interaction's range
    val_min, val_max = noise.min(), noise.max()
    if np.isclose(val_max, val_min):
        noise = np.zeros(len(noise), dtype=np.float32)
    else:
        noise = (noise - val_min) / (val_max - val_min)

    if not name:
        name = sep.join(factors) + sep + f"{noise_percent:.0f}%{dist.name if dist else ''}noise"

    df[name] = agg_func(df[factors].values, axis=1) * (1 + (noise - 0.5) * noise_percent / 100)


def categorize_1d_array(vals: np.ndarray, min_ncats: int, method: str, astropy_sample_size: int, method_kwargs: dict, dtype=np.int16, nan_filler: float = 0.0):

    ordinal_encoder = OrdinalEncoder()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Booleans bust become int8
    # ----------------------------------------------------------------------------------------------------------------------------

    if vals.dtype.name != "category" and np.issubdtype(vals.dtype, np.bool_):
        vals = vals.astype(np.int8)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Missings are imputed using rolling median (for ts safety)
    # ----------------------------------------------------------------------------------------------------------------------------

    if pd.isna(vals).any():
        # imputer = SimpleImputer(strategy="most_frequent", add_indicator=False)
        # vals = imputer.fit_transform(vals.reshape(-1, 1))
        vals = pd.Series(vals)
        # vals=vals.fillna(vals.rolling(window=nan_rolling_window,min_periods=nan_rolling_min_periods).apply(lambda x: mode(x)[0])).fillna(nan_filler).values
        vals = vals.fillna(nan_filler).values

    vals = vals.reshape(-1, 1)

    if vals.dtype.name != "category":
        nuniques = len(np.unique(vals[: min_ncats * 10]))
        if nuniques <= min_ncats:
            nuniques = len(np.unique(vals))
    else:
        nuniques = min_ncats

    if method == "discretizer":
        bins = method_kwargs.get("n_bins")
    else:
        bins = method_kwargs.get("bins")

    if vals.dtype.name != "category" and nuniques > min_ncats:
        if method == "discretizer":
            if nuniques > bins:
                discretizer = KBinsDiscretizer(**method_kwargs, encode="ordinal")
                new_vals = discretizer.fit_transform(vals)
            else:
                new_vals = ordinal_encoder.fit_transform(vals)
        else:
            if method == "numpy":

                bin_edges = np.histogram_bin_edges(
                    vals,
                    bins=bins,
                )
            elif method == "astropy":
                if bins == "blocks" and len(vals) >= astropy_sample_size:
                    _, bin_edges = histogram(np.random.choice(vals.ravel(), size=astropy_sample_size, replace=False), bins=bins)
                elif bins == "knuth" and len(vals) >= astropy_sample_size:
                    _, bin_edges = histogram(np.random.choice(vals.ravel(), size=astropy_sample_size, replace=False), bins=bins)
                else:
                    _, bin_edges = histogram(vals, bins=bins)

            if bin_edges[0] <= vals.min():
                bin_edges = bin_edges[1:]

            new_vals = ordinal_encoder.fit_transform(np.digitize(vals, bins=bin_edges, right=True))

    else:
        new_vals = ordinal_encoder.fit_transform(vals)

    return new_vals.ravel().astype(dtype)


def categorize_dataset_old(
    df: pd.DataFrame,
    method: str = "discretizer",
    method_kwargs: dict = dict(strategy="quantile", n_bins=4),
    min_ncats: int = 50,
    astropy_sample_size: int = 10_000,
    dtype=np.int16,
    n_jobs: int = -1,
    parallel_kwargs: dict = {},
):
    """
    Convert dataframe into ordinal-encoded one.
    """

    data = None
    numerical_cols = []
    categorical_factors = []

    numerical_cols = df.head(5).select_dtypes(exclude=("category", "object", "bool")).columns.values.tolist()

    data = []
    if n_jobs == -1 or n_jobs > 1:
        fnc = delayed(categorize_1d_array)
    else:
        fnc = categorize_1d_array

    for col in tqdmu(numerical_cols, leave=False, desc="Binning of numericals"):
        data.append(
            fnc(vals=df[col].values, min_ncats=min_ncats, method=method, astropy_sample_size=astropy_sample_size, method_kwargs=method_kwargs, dtype=dtype)
        )

    if n_jobs == -1 or n_jobs > 1:
        data = parallel_run(data, n_jobs=n_jobs, **parallel_kwargs)
    data = np.vstack(data).T

    categorical_factors = df.select_dtypes(include=("category", "object", "bool"))
    categorical_cols = []
    if categorical_factors.shape[1] > 0:
        categorical_cols = categorical_factors.columns.values.tolist()
        ordinal_encoder = OrdinalEncoder()
        new_vals = ordinal_encoder.fit_transform(categorical_factors)

        max_cats = new_vals.max(axis=0)
        exc_idx = max_cats > np.iinfo(dtype).max
        n_max_cats = exc_idx.sum()
        if n_max_cats:
            logger.warning(f"{n_max_cats:_} factors exceeded dtype {dtype} and were truncated: {np.asarray(categorical_cols)[exc_idx]}")
        new_vals = new_vals.astype(dtype)

        if data is None:
            data = new_vals
        else:
            data = np.append(data, new_vals, axis=1)

    nbins = data.max(axis=0).astype(np.int32) + 1 - data.min(axis=0).astype(np.int32)

    return data, numerical_cols + categorical_cols, nbins


@njit
def digitize(arr: np.ndarray, bins: np.ndarray, dtype=np.int32) -> np.ndarray:
    res = np.empty(len(arr), dtype=dtype)
    for i, val in enumerate(arr):
        for j, bin_edge in enumerate(bins):
            if val <= bin_edge:
                res[i] = j
                break
    return res


from numba import prange


# @njit()
def edges(arr, quantiles):
    bin_edges = np.asarray(np.percentile(arr, quantiles))
    return bin_edges


@njit()
def quantize_dig(arr, bins):
    return np.digitize(arr, bins[1:-1], right=True)


@njit()
def quantize_search(arr, bins):
    return np.searchsorted(bins[1:-1], arr, side="right")


@njit()
def discretize_uniform(arr: np.ndarray, n_bins: int, min_value: float = None, max_value: float = None, dtype: object = np.int8) -> np.ndarray:
    if min_value is None or max_value is None:
        min_value, max_value = arrayMinMax(arr)
    rev_bin_width = n_bins / (max_value - min_value + min_value / 2)
    return ((arr - min_value) * rev_bin_width).astype(dtype)


@njit()
def discretize_array(
    arr: np.ndarray, n_bins: int = 10, method: str = "quantile", min_value: float = None, max_value: float = None, dtype: object = np.int8
) -> np.ndarray:
    """Discretize cont variable into bins.

    Optimized version with mix of pure numpy and njitting.


    %timeit quantize_search(df['a'].values,bins) #njitted
    24.6 ms ± 191 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    time: 2 s (started: 2024-02-09 19:58:31 +03:00)

    %timeit quantize_search(df['a'].values,bins) #just numpy
    27.2 ms ± 219 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    time: 2.2 s (started: 2024-02-09 19:52:59 +03:00)

    %timeit quantize_dig(df['a'].values, bins) #njitted
    23.7 ms ± 222 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    time: 1.92 s (started: 2024-02-09 19:58:24 +03:00)

    %timeit quantize_dig(df['a'].values, bins) #just numpy
    31.1 ms ± 292 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    time: 2.52 s (started: 2024-02-09 19:53:01 +03:00)


    """
    if method == "uniform":
        return discretize_uniform(arr=arr, n_bins=n_bins, min_value=min_value, max_value=max_value, dtype=dtype)
    elif method == "quantile":
        bins_edges = get_binning_edges(arr=arr, n_bins=n_bins, method=method, min_value=min_value, max_value=max_value)  # pure numpy
    # return quantize_dig(arr,bins_edges).astype(dtype) #njitted
    return quantize_search(arr, bins_edges).astype(dtype)  # njitted


@njit(parallel=True)
def discretize_2d_array(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    min_ncats: int = 50,
    min_values: float = None,
    max_values: float = None,
    dtype: object = np.int8,
) -> np.ndarray:
    """ """

    res = np.empty_like(arr, dtype=dtype)

    # for col in tqdmu(range(arr.shape[1]), desc="col", leave=False):
    for col in prange(arr.shape[1]):
        res[:, col] = discretize_array(
            arr=arr[:, col],
            n_bins=n_bins,
            method=method,
            min_value=min_values[col] if min_values is not None else None,
            max_value=max_values[col] if max_values is not None else None,
            dtype=dtype,
        )
    return res


@jit(nopython=False)
def get_binning_edges(arr: np.ndarray, n_bins: int = 10, method: str = "uniform", min_value: float = None, max_value: float = None):
    """
    np.quantiles works faster when unjitted

    %timeit edges(df['a'].values,quantiles) #njitted
    83.9 ms ± 274 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    time: 6.81 s (started: 2024-02-09 17:36:50 +03:00)

    %timeit edges(df['a'].values,quantiles) #just numpy
    30.9 ms ± 541 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    time: 2.52 s (started: 2024-02-09 17:35:58 +03:00)
    """
    if method == "uniform":
        if min_value is None or max_value is None:
            min_value, max_value = arrayMinMax(arr)
        bin_edges = np.linspace(min_value, max_value, n_bins + 1)

    elif method == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.asarray(np.percentile(arr, quantiles))

    return bin_edges


def discretize_sklearn(
    arr: np.ndarray, n_bins: int = 10, method: str = "uniform", min_value: float = None, max_value: float = None, dtype: object = np.int8
) -> np.ndarray:
    """Simplified vesrion taken from Sklearn's KBinsdiscretizer.
    np.searchsorted runs twice faster when unjitted (as of Feb 2024 at least), so the func is not njitted.
    """

    bins_edges = get_binning_edges(arr=arr, n_bins=n_bins, method=method, min_value=min_value, max_value=max_value)
    return np.searchsorted(bins_edges[1:-1], arr, side="right").astype(dtype)


def categorize_dataset(
    df: pd.DataFrame,
    method: str = "quantile",
    n_bins: int = 4,
    min_ncats: int = 50,
    dtype=np.int16,
):
    """
    Convert dataframe into ordinal-encoded one.
    For cat columns uses OrdinalEncoder.
    For the rest uses new discretize_2d_array.
    Does not care for min_cats yet.
    """

    data = None
    numerical_cols = []
    categorical_factors = []

    numerical_cols = df.head(5).select_dtypes(exclude=("category", "object", "bool")).columns.values.tolist()

    data = discretize_2d_array(arr=df[numerical_cols].values, n_bins=n_bins, method=method, min_ncats=min_ncats, min_values=None, max_values=None, dtype=dtype)

    categorical_factors = df.select_dtypes(include=("category", "object", "bool"))
    categorical_cols = []
    if categorical_factors.shape[1] > 0:
        categorical_cols = categorical_factors.columns.values.tolist()
        ordinal_encoder = OrdinalEncoder()
        new_vals = ordinal_encoder.fit_transform(categorical_factors)

        max_cats = new_vals.max(axis=0)
        exc_idx = max_cats > np.iinfo(dtype).max
        n_max_cats = exc_idx.sum()
        if n_max_cats:
            logger.warning(f"{n_max_cats:_} factors exceeded dtype {dtype} and were truncated: {np.asarray(categorical_cols)[exc_idx]}")
        new_vals = new_vals.astype(dtype)

        if data is None:
            data = new_vals
        else:
            data = np.append(data, new_vals, axis=1)

    nbins = data.max(axis=0).astype(np.int32) + 1  # -data.min(axis=0).astype(np.int32)

    return data, numerical_cols + categorical_cols, nbins.tolist()


class MRMR(BaseEstimator, TransformerMixin):
    """Finds subset of features having highest impact on target and least redundancy.

    Parameters
    ----------
        cv : int, cross-validation generator or an iterable, default=None

    Attributes
    ----------


    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ranking_ ?: narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    """

    def __init__(
        self,
        # quantization
        quantization_method: str = "quantile",
        quantization_nbins: int = 10,
        quantization_dtype: object = np.int16,
        # factors
        factors_names_to_use: Sequence[str] = None,
        factors_to_use: Sequence[int] = None,
        # algorithm
        mrmr_relevance_algo: str = "fleuret",
        mrmr_redundancy_algo: str = "fleuret",
        reduce_gain_on_subelement_chosen: bool = True,
        use_simple_mode: bool = True,  # when true, works very fast but leaves redundant features
        run_additional_rfecv_minutes: bool = False,
        # performance
        extra_x_shuffling: bool = True,
        dtype=np.int32,
        random_seed: int = None,
        use_gpu: bool = False,
        n_workers: int = 1,
        # confidence
        min_occupancy: int = None,
        min_nonzero_confidence: float = 0.99,
        full_npermutations: int = 1_000,
        baseline_npermutations: int = 100,
        # stopping conditions
        min_relevance_gain: float = 0.0001,
        max_consec_unconfirmed: int = 10,
        max_runtime_mins: float = None,
        interactions_min_order: int = 1,
        interactions_max_order: int = 1,
        interactions_order_reversed: bool = False,
        max_veteranes_interactions_order: int = 1,
        only_unknown_interactions: bool = False,
        # feature engineering settings
        fe_max_steps=1,
        fe_npermutations=0,
        fe_ntop_features=0,
        fe_unary_preset="minimal",
        fe_binary_preset="minimal",
        fe_max_pair_features: int = 1,
        fe_min_nonzero_confidence: float = 1.0,
        fe_min_pair_mi: float = 0.001,
        fe_min_pair_mi_prevalence: float = 1.05,  # transformations of what exactly pairs of factors we consider, at all. mi of entire pair must be at least that higher than the mi of its individual factors.
        fe_min_engineered_mi_prevalence: float = 0.98,  # mi of transformed pair must be at least that higher than the mi of the entire pair
        fe_good_to_best_feature_mi_threshold: float = 0.98,  # when multiple good transformations exist for the same factors pair.
        fe_max_external_validation_factors: int = 0,  # how many other factors to validate against
        fe_max_polynoms: int = 0,
        fe_print_best_mis_only: bool = True,
        fe_smart_polynom_iters: int = 0,
        fe_smart_polynom_optimization_steps: int = 1000,
        fe_min_polynom_degree: int = 3,
        fe_max_polynom_degree: int = 8,
        fe_min_polynom_coeff: float = -10.0,
        fe_max_polynom_coeff: float = 10.0,
        # verbosity and formatting
        verbose: Union[bool, int] = 0,
        ndigits: int = 5,
        parallel_kwargs=dict(max_nbytes=MAX_JOBLIB_NBYTES),
        # CV
        cv: Union[object, int, None] = 3,
        cv_shuffle: bool = False,
        # service
        random_state: int = None,
        n_jobs: int = -1,
        skip_retraining_on_same_shape: bool = False,
        # hidden
        n_features_in_: int = 0,
        feature_names_in_: Sequence = None,
        support_: np.ndarray = None,
        stop_file: str = "stop",
    ):

        # checks
        if n_jobs == -1:
            n_jobs = psutil.cpu_count(logical=False)

        # assert isinstance(estimator, (BaseEstimator,))

        # save params
        store_params_in_object(obj=self, params=get_parent_func_args())
        self.signature = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, **fit_params):
        """We run N selections on data subsets, and pick only features that appear in all selections"""

        # ----------------------------------------------------------------------------------------------------------------------------
        # Compute inputs/outputs signature
        # ----------------------------------------------------------------------------------------------------------------------------

        signature = (X.shape, y.shape)
        if self.skip_retraining_on_same_shape:
            if signature == self.signature:
                if self.verbose:
                    logger.info(f"Skipping retraining on the same inputs signature {signature}")
                return self

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        start_time = timer()
        ran_out_of_time = False

        quantization_method = self.quantization_method
        quantization_nbins = self.quantization_nbins
        dtype = self.dtype

        max_runtime_mins = self.max_runtime_mins
        random_state = self.random_state
        parallel_kwargs = self.parallel_kwargs
        n_jobs = self.n_jobs
        verbose = self.verbose
        cv_shuffle = self.cv_shuffle
        cv = self.cv

        prefetch_factor = 4

        fe_max_steps = self.fe_max_steps
        fe_npermutations = self.fe_npermutations
        fe_unary_preset = self.fe_unary_preset
        fe_binary_preset = self.fe_binary_preset
        fe_max_pair_features = self.fe_max_pair_features
        fe_min_nonzero_confidence = self.fe_min_nonzero_confidence
        fe_min_pair_mi = self.fe_min_pair_mi
        fe_min_pair_mi_prevalence = self.fe_min_pair_mi_prevalence
        fe_min_engineered_mi_prevalence = self.fe_min_engineered_mi_prevalence
        fe_good_to_best_feature_mi_threshold = self.fe_good_to_best_feature_mi_threshold
        fe_max_external_validation_factors = self.fe_max_external_validation_factors
        fe_max_polynoms = self.fe_max_polynoms
        fe_print_best_mis_only = self.fe_print_best_mis_only
        fe_smart_polynom_iters = self.fe_smart_polynom_iters
        fe_smart_polynom_optimization_steps = self.fe_smart_polynom_optimization_steps
        fe_min_polynom_degree = self.fe_min_polynom_degree
        fe_max_polynom_degree = self.fe_max_polynom_degree
        fe_min_polynom_coeff = self.fe_min_polynom_coeff
        fe_max_polynom_coeff = self.fe_max_polynom_coeff

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init cv
        # ----------------------------------------------------------------------------------------------------------------------------

        """
        if cv is None or str(cv).isnumeric():
            if cv is None:
                cv = 3

            if groups is not None:
                cv = GroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
            else:
                cv = KFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
            
            if verbose:
                logger.info(f"Using cv={cv}")
        """

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        # ---------------------------------------------------------------------------------------------------------------
        # Temporarily inject targets
        # ---------------------------------------------------------------------------------------------------------------

        target_prefix = "targ_" + str(np.random.random())[3:9]
        y_shape = y.shape
        if len(y_shape) == 2:
            y_shape = y_shape[1]
        else:
            y_shape = 1
        target_names = [target_prefix + str(i) for i in range(y_shape)]

        if isinstance(y, np.ndarray):
            vals = y
        else:
            vals = y.values

        if vals.dtype == np.int64:
            print("Converted targets from int64 to int16.")
            vals = vals.astype(np.int16)

        X[target_names] = vals.reshape(-1, 1)

        # ---------------------------------------------------------------------------------------------------------------
        # Discretize continuous data
        # ---------------------------------------------------------------------------------------------------------------

        logger.info("categorizing dataset...")
        data, cols, nbins = categorize_dataset(
            df=X.ffill().bfill(),
            method=self.quantization_method,
            n_bins=self.quantization_nbins,
            dtype=self.quantization_dtype,
        )
        logger.info("categorized.")

        target_indices = [cols.index(col) for col in target_names]

        # ---------------------------------------------------------------------------------------------------------------
        # Core
        # ---------------------------------------------------------------------------------------------------------------

        """
        if random_state is not None:            
            set_random_seed(random_state) 

        splitter = cv.split(X=X, y=y, groups=groups)

        subsets_selections=[]
        if n_jobs==-1 or n_jobs>1:
            fnc=delayed(screen_predictors)
        else:
            fnc=screen_predictors
            
            if verbose:
                splitter = tqdmu(splitter, desc="CV folds", leave=False, total=cv.n_splits)            

        for nfold, (train_index, test_index) in enumerate(splitter):
            subsets_selections.append(fnc(
                            factors_data=data,
                            y=[target_idx],
                            factors_nbins=n_bins,
                            factors_names=cols,
                            interactions_max_order=1,
                            full_npermutations=0,
                            baseline_npermutations=10,
                            min_nonzero_confidence=1.0,
                            max_consec_unconfirmed=20,
                            min_relevance_gain=0.025,
                            max_runtime_mins=None,
                            max_veteranes_interactions_order=1,
                            reduce_gain_on_subelement_chosen=True,
                            random_seed=None,
                            use_gpu=False,
                            n_workers=n_jobs,
                            verbose=2,
                            ndigits=5,
                            parallel_kwargs=parallel_kwargs,

                    ))

        if n_jobs==-1 or n_jobs>1:
            subsets_selections = parallel_run(jobs,n_jobs=n_jobs,**parallel_kwargs)
        
        for selected_vars, predictors, any_influencing in subsets_selections:
            pass
            
        """

        """
        #service
        random_state: int = None,
        n_jobs:int=-1,  
        """

        categorical_vars_names = X.head().select_dtypes(include=("category", "object", "bool")).columns.values.tolist()
        categorical_vars = [cols.index(col) for col in categorical_vars_names]

        if fe_max_steps > 0:
            unary_transformations = create_unary_transformations(preset=fe_unary_preset)
            binary_transformations = create_binary_transformations(preset=fe_binary_preset)
            if fe_max_polynoms:
                polynomial_transformations = {}  # 'identity':lambda x: x,
                for _ in range(fe_max_polynoms):
                    length = np.random.randint(3, 9)
                    # coef=(np.random.random(length)-0.5)*1
                    coef = np.empty(shape=length, dtype=np.float32)
                    for i in range(length):
                        coef[i] = np.random.normal((1.0 if i == 1 else 0.0), scale=0.05)

                    unary_transformations["poly_" + str(coef)] = coef

            if verbose > 2:
                print(f"nunary_transformations: {len(unary_transformations):_}")
                print(f"nbinary_transformations: {len(binary_transformations):_}")

            engineered_features = set()
            checked_pairs = set()

        num_fs_steps = 0
        while True:
            n_recommended_features = 0
            times_spent = defaultdict(float)
            selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe, freqs_y = (
                screen_predictors(
                    factors_data=data,
                    y=target_indices,
                    factors_nbins=nbins,
                    factors_names=cols,
                    factors_names_to_use=self.factors_names_to_use,
                    factors_to_use=self.factors_to_use,
                    # algorithm
                    mrmr_relevance_algo=self.mrmr_relevance_algo,
                    mrmr_redundancy_algo=self.mrmr_redundancy_algo,
                    reduce_gain_on_subelement_chosen=self.reduce_gain_on_subelement_chosen,
                    use_simple_mode=self.use_simple_mode,
                    # performance
                    extra_x_shuffling=self.extra_x_shuffling,
                    dtype=self.dtype,
                    random_seed=self.random_seed,
                    use_gpu=self.use_gpu,
                    n_workers=self.n_workers,
                    # confidence
                    min_occupancy=self.min_occupancy,
                    min_nonzero_confidence=self.min_nonzero_confidence,
                    full_npermutations=self.full_npermutations,
                    baseline_npermutations=self.baseline_npermutations,
                    # stopping conditions
                    min_relevance_gain=self.min_relevance_gain,
                    max_consec_unconfirmed=self.max_consec_unconfirmed,
                    max_runtime_mins=self.max_runtime_mins,
                    interactions_min_order=self.interactions_min_order,
                    interactions_max_order=self.interactions_max_order,
                    interactions_order_reversed=self.interactions_order_reversed,
                    max_veteranes_interactions_order=self.max_veteranes_interactions_order,
                    only_unknown_interactions=self.only_unknown_interactions,
                    # verbosity and formatting
                    verbose=self.verbose,
                    ndigits=self.ndigits,
                    parallel_kwargs=self.parallel_kwargs,
                    stop_file=self.stop_file,
                )
            )

            if fe_max_steps == 0 or num_fs_steps >= fe_max_steps:
                break

            # Feature engineering part here

            if verbose:
                logger.info(f"MRMR+ selected {len(selected_vars):_} out of {self.n_features_in_:_} features before the Feature Engineering step.")

            if verbose >= 2:
                logger.info(f"Computing prospective FE pairs...")

            if self.fe_ntop_features:
                numeric_vars_to_consider = selected_vars[: self.fe_ntop_features]
            else:
                numeric_vars_to_consider = selected_vars

            numeric_vars_to_consider = set(numeric_vars_to_consider) - set(categorical_vars)

            all_pairs = list(combinations(numeric_vars_to_consider, 2))

            if verbose:
                logger.info(f"Feature Engineering: Computing MIs of {len(all_pairs):_} most prospective feature pairs...")

            if len(numeric_vars_to_consider) < 50:
                compute_pairs_mis(
                    all_pairs=tqdmu(all_pairs, desc="getting pairs MIs", leave=False, mininterval=5),
                    data=data,
                    target_indices=target_indices,
                    nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                    fe_npermutations=fe_npermutations,
                    cached_confident_MIs=cached_confident_MIs,
                    cached_MIs=cached_MIs,
                    fe_min_pair_mi=fe_min_pair_mi,
                    fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
                )
            else:

                dicts = parallel_run(
                    [
                        delayed(compute_pairs_mis)(
                            all_pairs=chunk,
                            data=data,
                            target_indices=target_indices,
                            nbins=nbins,
                            classes_y=classes_y,
                            classes_y_safe=classes_y_safe,
                            freqs_y=freqs_y,
                            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                            fe_npermutations=fe_npermutations,
                            cached_confident_MIs=cached_confident_MIs,
                            cached_MIs=cached_MIs,
                            fe_min_pair_mi=fe_min_pair_mi,
                            fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
                        )
                        for chunk in split_list_into_chunks(all_pairs, len(all_pairs) // (n_jobs * prefetch_factor))
                    ],
                    n_jobs=n_jobs,
                    **parallel_kwargs,
                )
                for next_dict in dicts:
                    cached_MIs.update(next_dict)

            # ---------------------------------------------------------------------------------------------------------------
            # For every pair of factors (A,B), select ones having MI((A,B),Y)>MI(A,Y)+MI(B,Y). Such ones must posess more special connection!
            # ---------------------------------------------------------------------------------------------------------------

            vars_usage_counter = defaultdict(int)
            prospective_pairs = {}
            for raw_vars_pair, pair_mi in sort_dict_by_value(cached_MIs).items():
                if len(raw_vars_pair) == 2:
                    if raw_vars_pair in checked_pairs:
                        continue
                    if raw_vars_pair[0] in numeric_vars_to_consider and raw_vars_pair[1] in numeric_vars_to_consider:
                        ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
                        if pair_mi > ind_elems_mi_sum * fe_min_pair_mi_prevalence:
                            uplift = pair_mi / ind_elems_mi_sum
                            if verbose >= 2:
                                logger.info(
                                    f"Factors pair {raw_vars_pair} will be considered for Feature Engineering, {ind_elems_mi_sum:.4f}->{pair_mi:.4f}, rat={uplift:.2f}"
                                )
                            prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                            for var in raw_vars_pair:
                                vars_usage_counter[var] += 1

            # Now need to sort prospective_pairs by the uplift, to check most promising pairs within the time budget.
            # Also need to sort them by their members usage frequency+members ids sum. this way, their splitting will benefit more from caching.
            prospective_pairs = sort_dict_by_value(prospective_pairs, reverse=True)

            if fe_smart_polynom_iters:

                # ---------------------------------------------------------------------------------------------------------------
                # We search for best unary & binary transforms using Hermit polinomials & Optuna!
                # Degrees kep reasonable small as a form of regularization.
                # In theory (esp if inputs are normalized), Hermit polynomials can approximate any functional form, therefore replacing our
                # random experimenting with arbitrary functions (that was pretty limited anyways).
                # ---------------------------------------------------------------------------------------------------------------

                import optuna
                from optuna.samplers import TPESampler

                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def get_best_polynom_mi(coef_a, coef_b, vals_a, vals_b) -> float:

                    transformed_var_a = hermval(vals_a, c=coef_a)
                    transformed_var_b = hermval(vals_b, c=coef_b)

                    best_mi = -1
                    best_config = None

                    for bin_func_name, bin_func in binary_transformations.items():

                        final_transformed_vals = bin_func(transformed_var_a, transformed_var_b)

                        discretized_transformed_values = discretize_array(
                            arr=final_transformed_vals, n_bins=self.quantization_nbins, method=self.quantization_method, dtype=self.quantization_dtype
                        )
                        fe_mi, fe_conf = mi_direct(
                            discretized_transformed_values.reshape(-1, 1),
                            x=[0],
                            y=None,
                            factors_nbins=[self.quantization_nbins],
                            classes_y=classes_y,
                            classes_y_safe=classes_y_safe,
                            freqs_y=freqs_y,
                            min_nonzero_confidence=fe_min_nonzero_confidence,
                            npermutations=fe_npermutations,
                        )

                        if fe_mi > best_mi:
                            best_mi = fe_mi
                            best_config = bin_func_name

                    return best_mi

                for (raw_vars_pair, pair_mi), uplift in prospective_pairs.items():
                    vals_a = X.iloc[:, raw_vars_pair[0]].values
                    vals_b = X.iloc[:, raw_vars_pair[1]].values

                    for _ in range(fe_smart_polynom_iters):

                        length_a = np.random.randint(fe_min_polynom_degree, fe_max_polynom_degree)
                        length_b = np.random.randint(fe_min_polynom_degree, fe_max_polynom_degree)

                        # Define an objective function to be minimized.
                        def objective(trial):

                            coef_a = np.empty(shape=length_a, dtype=np.float32)
                            for i in range(length_a):
                                coef_a[i] = trial.suggest_float(f"a_{i}", fe_min_polynom_coeff, fe_max_polynom_coeff)

                            coef_b = np.empty(shape=length_b, dtype=np.float32)
                            for i in range(length_b):
                                coef_b[i] = trial.suggest_float(f"b_{i}", fe_min_polynom_coeff, fe_max_polynom_coeff)

                            res = get_best_polynom_mi(coef_a=coef_a, coef_b=coef_b, vals_a=vals_a, vals_b=vals_b)

                            return res

                        study = optuna.create_study(direction="maximize", sampler=TPESampler(multivariate=True))  # Create a new study.
                        study.optimize(objective, n_trials=fe_smart_polynom_optimization_steps)  # Invoke optimization of the objective function.

                        print(f"Best MI: {study.best_trial.value:.4f}, pair_mi={pair_mi:.4f}")
                        print(f"Best hyperparameters: {study.best_params}")
            else:
                original_cols = {i: self.feature_names_in_.index(col) for i, col in enumerate(cols) if col in self.feature_names_in_}
                if verbose >= 1:
                    logging.info(f"Checking {len(prospective_pairs):_} most prospective_pairs for feature engineering...")
                if len(X) < 50_000 or len(prospective_pairs) < 2:
                    prospective_additions = check_prospective_fe_pairs(
                        prospective_pairs,
                        X,
                        unary_transformations,
                        binary_transformations,
                        classes_y,
                        classes_y_safe,
                        freqs_y,
                        num_fs_steps,
                        cols,
                        original_cols,
                        fe_max_steps,
                        fe_npermutations,
                        fe_max_pair_features,
                        fe_print_best_mis_only,
                        fe_min_nonzero_confidence,
                        fe_min_engineered_mi_prevalence,
                        fe_good_to_best_feature_mi_threshold,
                        fe_max_external_validation_factors,
                        numeric_vars_to_consider,
                        self.quantization_nbins,
                        self.quantization_method,
                        self.quantization_dtype,
                        times_spent,
                        verbose,
                    )
                else:

                    prospective_additions = {}
                    desired_nitems = max(1, len(prospective_pairs) // (n_jobs * prefetch_factor))

                    jobs_list = []

                    nitems = 0
                    cur_dict = {}
                    for key, value in prospective_pairs.items():
                        nitems += 1
                        cur_dict[key] = value
                        if nitems >= desired_nitems:
                            jobs_list.append(cur_dict)
                            nitems = 0
                            cur_dict = {}
                    if cur_dict:
                        jobs_list.append(cur_dict)

                    if verbose:
                        logger.info(
                            f"Using {desired_nitems:_} items per thread for checking {len(prospective_pairs):_} prospective_pairs with gain>{fe_min_pair_mi_prevalence:.2f}."
                        )

                    dicts = parallel_run(
                        [
                            delayed(check_prospective_fe_pairs)(
                                chunk,
                                X,
                                unary_transformations,
                                binary_transformations,
                                classes_y,
                                classes_y_safe,
                                freqs_y,
                                num_fs_steps,
                                cols,
                                original_cols,
                                fe_max_steps,
                                fe_npermutations,
                                fe_max_pair_features,
                                fe_print_best_mis_only,
                                fe_min_nonzero_confidence,
                                fe_min_engineered_mi_prevalence,
                                fe_good_to_best_feature_mi_threshold,
                                fe_max_external_validation_factors,
                                numeric_vars_to_consider,
                                self.quantization_nbins,
                                self.quantization_method,
                                self.quantization_dtype,
                                times_spent,
                                verbose,
                            )
                            for chunk in jobs_list
                        ],
                        # max_nbytes=0,
                        n_jobs=n_jobs,
                        **parallel_kwargs,
                    )
                    for next_dict in dicts:
                        prospective_additions.update(next_dict)

                for raw_vars_pair, (this_pair_features, transformed_vals, new_cols, new_nbins, messages) in prospective_additions.items():
                    if this_pair_features:
                        engineered_features.update(this_pair_features)
                        if verbose:
                            for mes in messages:
                                logger.info(mes)
                            # logger.info(f"Features {new_cols} are recommended to use as new features!")
                        if fe_max_steps > 1:
                            new_vals = np.empty(shape=(len(X), len(this_pair_features)), dtype=self.quantization_dtype)
                            for j in range(len(this_pair_features)):
                                new_vals[:, j] = discretize_array(
                                    arr=transformed_vals[:, j],
                                    n_bins=self.quantization_nbins,
                                    method=self.quantization_method,
                                    dtype=self.quantization_dtype,
                                )
                            data = np.append(data, new_vals, axis=1)
                            nbins = nbins + new_nbins
                            cols = cols + new_cols
                            for col in new_cols:
                                X[col] = transformed_vals[:, j]

                        n_recommended_features += len(this_pair_features)

                    # !TODO!  handle factors_to_use etc
                    """
                    factors_names_to_use=self.factors_names_to_use,
                    factors_to_use=self.factors_to_use,                    
                        """
                    checked_pairs.add(raw_vars_pair)

            if n_recommended_features == 0:
                break

            num_fs_steps += 1
            if num_fs_steps >= fe_max_steps:
                break  # uncomment to avoid recheck of single-rounded FE

        if verbose > 2:
            print("time spent by binary func:", sort_dict_by_value(times_spent))
        # Possibly decide on eliminating original features? (if constructed ones cover 90%+ of MI)

        # ---------------------------------------------------------------------------------------------------------------
        # Drop Temporarily targets
        # ---------------------------------------------------------------------------------------------------------------

        X.drop(columns=target_names, inplace=True)

        # ---------------------------------------------------------------------------------------------------------------
        # selected_vars needs to be transformed to names using the cols variable and then back to indices using original Df columns names.
        # It's needed 'casue categorize_data can rearrange cat columns.
        # ---------------------------------------------------------------------------------------------------------------

        selected_vars_names = np.array(cols)[np.array(selected_vars)]
        selected_vars = [self.feature_names_in_.index(col) for col in selected_vars_names]  # !TODO! failing when fe_max_steps>1. need other source.

        # ---------------------------------------------------------------------------------------------------------------
        # additional_rfecv run
        # ---------------------------------------------------------------------------------------------------------------

        if self.run_additional_rfecv_minutes:
            """On the factors discarded by MRMR, let's run RFECV to see if any of them participate in interactions"""
            n_unexplored = X.shape[1] - len(selected_vars)
            if n_unexplored > 0:
                if verbose:
                    logger.info(
                        f"Running RFECV for {self.run_additional_rfecv_minutes} minute(s) over {n_unexplored:_} feature(s) discarded by MRMR to extract interactions..."
                    )

                from mlframe.training import get_training_configs

                configs = get_training_configs(has_time=True)

                params = configs.COMMON_RFECV_PARAMS.copy()
                params["max_runtime_mins"] = self.run_additional_rfecv_minutes

                if len(y) / len(np.unique(y)) > 100:  # classification

                    cb_num_rfecv = RFECV(
                        estimator=CatBoostClassifier(**configs.CB_CLASSIF),
                        fit_params=dict(plot=False),
                        cat_features=categorical_vars_names,
                        scoring=make_scorer(
                            score_func=compute_probabilistic_multiclass_error, needs_proba=True, needs_threshold=False, greater_is_better=False
                        ),
                        **params,
                    )
                    temp_columns = list(set(X.columns) - set(X.columns[selected_vars]))
                    cb_num_rfecv.fit(X[temp_columns], y)

                    if cb_num_rfecv.n_features_ > 0:
                        new_features = np.array(temp_columns)[cb_num_rfecv.support_]
                        if verbose:
                            logger.info(f"RFECV selected {cb_num_rfecv.n_features_:_} additional feature(s): {new_features}")
                        for feature in new_features:
                            selected_vars.append(self.feature_names_in_.index(feature))
                    else:
                        if verbose:
                            logger.info(f"RFECV selected no additional features.")

        # ---------------------------------------------------------------------------------------------------------------
        # Assign support
        # ---------------------------------------------------------------------------------------------------------------

        self.support_ = np.array(selected_vars)
        if selected_vars:
            self.n_features_ = len(selected_vars)
        else:
            self.n_features_ = 0

        # ---------------------------------------------------------------------------------------------------------------
        # assign extra vars for upcoming vars improving
        # ---------------------------------------------------------------------------------------------------------------

        # self.cached_MIs_ = cached_MIs
        # self.cached_cond_MIs_ = cached_cond_MIs
        # self.cached_confident_MIs_ = cached_confident_MIs

        # ---------------------------------------------------------------------------------------------------------------
        # Report FS results
        # ---------------------------------------------------------------------------------------------------------------

        if verbose:
            predictors_str = ", ".join([f"{el['name']}: {el['gain']:.4f}" for el in predictors[:50]])
            predictors_str = textwrap.shorten(predictors_str, width=300)
            logger.info(f"MRMR+ selected {self.n_features_:_} out of {self.n_features_in_:_} features: {predictors_str}")

        self.signature = signature
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_]
        else:
            return X[:, self.support_]


def check_prospective_fe_pairs(
    prospective_pairs,
    X,
    unary_transformations,
    binary_transformations,
    classes_y,
    classes_y_safe,
    freqs_y,
    num_fs_steps,
    cols,
    original_cols,
    fe_max_steps,
    fe_npermutations,
    fe_max_pair_features,
    fe_print_best_mis_only,
    fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    numeric_vars_to_consider,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    times_spent,
    verbose,
):
    # ---------------------------------------------------------------------------------------------------------------
    # Starting from the most heavily connected pairs, create a big pool of original features+their unary transforms.
    # ---------------------------------------------------------------------------------------------------------------

    # individual vars referenced more than once go to the global pool, rest to the local (not stored)?

    res = {}

    if verbose >= 2:
        logging.info(f"Creating a pool of {len(prospective_pairs) * len(unary_transformations) * 2:_} unary transformations for feature engineering.")

    transformed_vars = np.empty(shape=(len(X), len(prospective_pairs) * len(unary_transformations) * 2), dtype=np.float32)

    vars_transformations = {}
    i = 0
    for (raw_vars_pair, pair_mi), uplift in prospective_pairs.items():
        for var in raw_vars_pair:
            vals = X.iloc[:, original_cols[var]].values
            for tr_name, tr_func in unary_transformations.items():
                key = (var, tr_name)
                if key not in vars_transformations:
                    try:
                        if "poly_" in tr_name:
                            transformed_vars[:, i] = hermval(vals, c=tr_func)
                        else:
                            transformed_vars[:, i] = tr_func(vals)
                    except Exception as e:
                        logger.error(
                            f"Error when performing {tr_name} on array {vals[:5]}, var={cols[var]}: {str(e)}, isnan={np.isnan(vals).sum()}, isinf={np.isinf(vals).sum()}, nanmin={np.nanmin(vals)}"
                        )
                    else:
                        vars_transformations[key] = i
                        i += 1

    if verbose >= 2:
        logging.info(f"Created. For every pair from the pool, trying all known functions...")

    # ---------------------------------------------------------------------------------------------------------------
    # Then, for every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM).
    # Record best pairs.
    # ---------------------------------------------------------------------------------------------------------------

    for (
        raw_vars_pair,
        pair_mi,
    ), uplift in tqdmu(
        prospective_pairs.items(), desc="pair", leave=False
    ):  # better to start considering form the most prospective pairs with highest mis ratio!

        messages = []

        combs = list(
            combinations(
                [(raw_vars_pair[0], key) for key in unary_transformations.keys()] + [(raw_vars_pair[1], key) for key in unary_transformations.keys()],
                2,
            )
        )

        combs = [
            transformations_pair for transformations_pair in combs if transformations_pair[0][0] != transformations_pair[1][0]
        ]  # let's skip trying to transform the same factor for now
        # print(f"trying {len(combs):_} combs")

        best_config, best_mi = None, -1
        this_pair_features = set()
        var_pairs_perf = {}

        final_transformed_vals = np.empty(
            shape=(len(X), len(combs) * len(binary_transformations)), dtype=np.float32
        )  # !TODO! optimize allocation of this array before the main loop!

        i = 0
        for transformations_pair in combs:
            if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                continue
            param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
            param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]

            for bin_func_name, bin_func in binary_transformations.items():

                start = timer()
                try:
                    # with np.errstate(invalid='ignore'):
                    final_transformed_vals[:, i] = bin_func(param_a, param_b)
                except Exception as e:
                    logger.error(f"Error when performing {bin_func}")
                else:
                    # if np.isnan(final_transformed_vals[:, i]).sum()>0:
                    #    final_transformed_vals[:, i] =pd.Series(final_transformed_vals[:, i] ).ffill().bfill().values
                    np.nan_to_num(final_transformed_vals[:, i], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                    times_spent[bin_func_name] += timer() - start

                    discretized_transformed_values = discretize_array(
                        arr=final_transformed_vals[:, i], n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                    )
                    fe_mi, fe_conf = mi_direct(
                        discretized_transformed_values.reshape(-1, 1),
                        x=[0],
                        y=None,
                        factors_nbins=[quantization_nbins],
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        min_nonzero_confidence=fe_min_nonzero_confidence,
                        npermutations=fe_npermutations,
                    )

                    config = (transformations_pair, bin_func_name, i)
                    var_pairs_perf[config] = fe_mi

                    if fe_mi > best_mi:
                        best_mi = fe_mi
                        best_config = config
                    if fe_mi > best_mi * 0.85:
                        if not fe_print_best_mis_only or (fe_mi == best_mi):
                            if verbose > 2:
                                print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
                    i += 1

        if verbose > 2:
            print(f"For pair {raw_vars_pair}, best config is {best_config} with best mi= {best_mi}")

        if best_mi / pair_mi > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025):  # Best transformation is good enough

            # ---------------------------------------------------------------------------------------------------------------
            # Now, if there is a group of leaders with almost same performance, we need to approve them through some of the orther variables.
            # если будут возникать такие группы примерно одинаковых по силе лидеров, их придётся разрешать с помощью одного из других влияющих факторов
            # ---------------------------------------------------------------------------------------------------------------

            leading_features = []
            for next_config, next_mi in sort_dict_by_value(var_pairs_perf).items():
                if next_mi > best_mi * fe_good_to_best_feature_mi_threshold:
                    leading_features.append(next_config)

            if len(leading_features) > 1:
                if len(numeric_vars_to_consider) > 2:

                    if verbose > 2:
                        print(f"Taking {len(leading_features)} new features for a separate validation step!")

                    # ---------------------------------------------------------------------------------------------------------------
                    # Now let's test all of the candidates as is against the rest of the approved factors (also as is).
                    # Caindidates significantly outstanding (in terms of MI with target) with any of other approved factors are kept.
                    # ---------------------------------------------------------------------------------------------------------------

                    valid_pairs_perf = {}

                    for transformations_pair, bin_func_name, i in leading_features:
                        param_a = final_transformed_vals[:, i]

                        best_valid_mi = -1
                        config = (transformations_pair, bin_func_name, i)

                        external_factors = list(set(numeric_vars_to_consider) - set(raw_vars_pair))
                        if fe_max_external_validation_factors and len(external_factors) > fe_max_external_validation_factors:
                            external_factors = np.random.choice(external_factors, fe_max_external_validation_factors)

                        for external_factor in tqdmu(external_factors, desc="external validation factor", leave=False):
                            param_b = X.iloc[:, original_cols[external_factor]].values

                            for valid_bin_func_name, valid_bin_func in binary_transformations.items():

                                valid_vals = valid_bin_func(param_a, param_b)

                                discretized_transformed_values = discretize_array(
                                    arr=valid_vals, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                                )
                                fe_mi, fe_conf = mi_direct(
                                    discretized_transformed_values.reshape(-1, 1),
                                    x=[0],
                                    y=None,
                                    factors_nbins=[quantization_nbins],
                                    classes_y=classes_y,
                                    classes_y_safe=classes_y_safe,
                                    freqs_y=freqs_y,
                                    min_nonzero_confidence=fe_min_nonzero_confidence,
                                    npermutations=fe_npermutations,
                                )

                                if fe_mi > best_valid_mi:
                                    best_valid_mi = fe_mi
                                    if verbose > 2:
                                        print(
                                            f"MI of transformed pair {valid_bin_func_name}({(transformations_pair,bin_func_name)} with ext factor {external_factor})={fe_mi:.4f}"
                                        )

                        valid_pairs_perf[config] = best_valid_mi

                    # ---------------------------------------------------------------------------------------------------------------
                    # Now we recommend proceeding with top N best transformations!
                    # ---------------------------------------------------------------------------------------------------------------

                    for j, (config, valid_mi) in enumerate(sort_dict_by_value(valid_pairs_perf, reverse=True).items()):
                        if j < fe_max_pair_features:
                            new_feature_name = get_new_feature_name(fe_tuple=config, cols_names=cols)
                            if verbose:
                                messages.append(
                                    f"{new_feature_name} is recommended to use as a new feature! (won in validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                                )
                            this_pair_features.add((config, j))
                        else:
                            break
                else:
                    if verbose:
                        messages.append(
                            f"{len(leading_features)} are recommended to use as new features! (can't narrow down the list by validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                        )
                    for j, config in enumerate(leading_features):
                        if j < fe_max_pair_features:
                            this_pair_features.add((config, j))
            else:
                new_feature_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
                if verbose:
                    messages.append(
                        f"{new_feature_name} is recommended to use as a new feature! (clear winner) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                    )
                j = 0
                this_pair_features.add((best_config, j))

            transformed_vals, new_cols, new_nbins = None, None, None

            if this_pair_features:

                # ---------------------------------------------------------------------------------------------------------------
                # Bulk adding of found & checked best features.
                # ---------------------------------------------------------------------------------------------------------------

                if fe_max_steps > 1:
                    transformed_vals = np.empty(shape=(len(X), fe_max_pair_features), dtype=quantization_dtype)
                new_nbins = []
                new_cols = []

                for config, j in this_pair_features:
                    new_feature_name = get_new_feature_name(fe_tuple=config, cols_names=cols)
                    transformations_pair, bin_func_name, i = config

                    if fe_max_steps > 1:
                        transformed_vals[:, j] = final_transformed_vals[:, i]
                        new_nbins += [quantization_nbins]
                    new_cols += [new_feature_name]

                if fe_max_steps > 1:
                    transformed_vals = transformed_vals[:, : min(fe_max_pair_features, j + 1)]

            res[raw_vars_pair] = (this_pair_features, transformed_vals, new_cols, new_nbins, messages)

    return res


def compute_pairs_mis(
    all_pairs: Sequence,
    data,
    target_indices,
    nbins,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_min_nonzero_confidence,
    fe_npermutations,
    cached_confident_MIs: dict,
    cached_MIs: dict,
    fe_min_pair_mi: float,
    fe_min_pair_mi_prevalence: float,
):
    for raw_vars_pair in all_pairs:
        # check that every element of a pair has computed its MI with target
        for var in raw_vars_pair:
            if (var,) not in cached_confident_MIs and (var,) not in cached_MIs:
                mi, conf = mi_direct(
                    data,
                    x=(var,),
                    y=target_indices,
                    factors_nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )
                cached_MIs[(var,)] = mi

        # ensure that pair as a whole has computed its MI with target
        if raw_vars_pair not in cached_confident_MIs and raw_vars_pair not in cached_MIs:
            ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
            if ind_elems_mi_sum > fe_min_pair_mi:
                mi, conf = mi_direct(
                    data,
                    x=raw_vars_pair,
                    y=target_indices,
                    factors_nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )

                if mi > ind_elems_mi_sum * fe_min_pair_mi_prevalence:
                    cached_MIs[raw_vars_pair] = mi

    return cached_MIs


def get_existing_feature_name(fe_tuple: tuple, cols_names: Sequence) -> str:
    fname = cols_names[fe_tuple[0]]
    if fe_tuple[1] == "identity":
        return fname
    else:
        return f"{fe_tuple[1]}({fname})"


def get_new_feature_name(fe_tuple: tuple, cols_names: Sequence) -> str:
    return f"{fe_tuple[1]}({get_existing_feature_name(fe_tuple=fe_tuple[0][0],cols_names=cols_names)},{get_existing_feature_name(fe_tuple=fe_tuple[0][1],cols_names=cols_names)})"  # (((2, 'log'), (3, 'sin')), 'mul', 1016)


def njit_functions_dict(dict, exceptions: Sequence = ("grad1", "grad2", "sinc", "log", "logn", "greater", "less", "equal")):
    """Tries replacing funcs in the dict with their njitted equivqlents, caring for exceptions."""
    for key, func in dict.items():
        if key not in exceptions:
            try:
                dict[key] = njit(func)
            except Exception as e:
                pass


def smart_log(x: np.ndarray) -> np.ndarray:
    x_min = np.float32(np.nanmin(x))
    if x_min > 0:
        return np.log(x)
    else:
        return np.log(x + 1e-5 - x_min)


def create_unary_transformations(preset: str = "minimal"):
    unary_constraints = {
        # reverse trigonometric
        "arccos": "-1to1",
        "arcsin": "-1to1",
        "arctan": "-pi/2topi/2",
        # reverse hyperbolic
        "arccosh": "1toinf",
        "arctanh": "-0.(9)to0.(9)",
        # powers
        "sqrt": "pos",
        "log": "pos",
        "reciproc": "nonzero",
        "invsquared": "nonzero",
        "invqubed": "nonzero",
        "invcbrt": "nonzero",
        "invsqrt": "nonzero",
    }

    unary_transformations = {
        # simplest
        "identity": lambda x: x,
    }
    if preset != "minimal":
        unary_transformations.update(
            {
                "sign": np.sign,
                "neg": np.negative,
                "abs": np.abs,
                # outliers removal
                # Rounding
                "rint": np.rint,
                # np.modf Return the fractional and integral parts of an array, element-wise.
                # clip
                # powers
                "squared": lambda x: np.power(x, 2),
                "qubed": lambda x: np.power(x, 3),
                "reciproc": lambda x: np.power(x, -1),
                "invsquared": lambda x: np.power(x, -2),
                "invqubed": lambda x: np.power(x, -3),
                "cbrt": np.cbrt,
                "sqrt": lambda x: np.sqrt(np.abs(x)),
                "invcbrt": lambda x: np.power(x, -1 / 3),
                "invsqrt": lambda x: np.power(x, -1 / 2),
                # logarithms
                "log": smart_log,
                "exp": np.exp,
                # trigonometric
                "sin": np.sin,
            }
        )

    if preset == "maximal":
        unary_transformations.update(
            {
                # math an
                "grad1": np.gradient,
                "grad2": lambda x: np.gradient(x, edge_order=2),
                # trigonometric
                "sinc": np.sinc,
                "cos": np.cos,
                "tan": np.tan,
                # reverse trigonometric
                "arcsin": np.arcsin,
                "arccos": np.arccos,
                "arctan": np.arctan,
                # hyperbolic
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                # reverse hyperbolic
                "arcsinh": np.arcsinh,
                "arccosh": np.arccosh,
                "arctanh": np.arctanh,
                # special
                #'psi':sp.psi, polygamma(0,x) is same as psi
                "erf": sp.erf,
                "dawsn": sp.dawsn,
                "gammaln": sp.gammaln,
                #'spherical_jn':sp.spherical_jn
            }
        )

    njit_functions_dict(unary_transformations)

    if preset == "maximal":
        for order in range(3):
            unary_transformations["polygamma_" + str(order)] = lambda x: sp.polygamma(order, x)
            unary_transformations["struve" + str(order)] = lambda x: sp.struve(order, x)
            unary_transformations["jv" + str(order)] = lambda x: sp.jv(order, x)

        """j0
        faster version of this function for order 0.

        j1
        faster version of this function for order 1.
        """

    return unary_transformations


def create_binary_transformations(preset: str = "minimal"):

    binary_transformations = {
        # Basic
        "mul": np.multiply,
        "add": np.add,
        # Extrema
        "max": np.maximum,
        "min": np.minimum,
    }

    if preset == "maximal":
        binary_transformations.update(
            {
                # All kinds of averages
                "hypot": np.hypot,
                "logaddexp": np.logaddexp,
                "agm": sp.agm,  # Compute the arithmetic-geometric mean of a and b.
                # Rational routines
                #'lcm':np.lcm, # requires int arguments #  ufunc 'lcm' did not contain a loop with signature matching types (<class 'numpy.dtype[float32]'>, <class 'numpy.dtype[float32]'>) -> None
                #'gcd':np.gcd, # requires int arguments
                #'mod':np.remainder, # requires int arguments
                # Powers
                "pow": np.power,  # non-symmetrical! may required dtype=complex for arbitrary numbers
                # Logarithms
                "logn": lambda x, y: np.emath.logn(x - np.min(x) + 0.1, y - np.min(y) + 0.1),  # non-symmetrical!
                # DSP
                # 'convolve':np.convolve, # symmetrical wrt args. scipy.signal.fftconvolve should be faster? SLOW?
                # Linalg
                "heaviside": np.heaviside,  # non-symmetrical!
                #'cross':np.cross, # symmetrical # incompatible dimensions for cross product (dimension must be 2 or 3)
                # Comparison
                "greater": lambda x, y: np.greater(x, y).astype(int),
                "less": lambda x, y: np.less(x, y).astype(int),
                "equal": lambda x, y: np.equal(x, y).astype(int),
                # special
                "beta": sp.beta,  # symmetrical
                "binom": sp.binom,  # non-symmetrical! binomial coefficient considered as a function of two real variables.
            }
        )

    njit_functions_dict(binary_transformations)

    return binary_transformations
