"""Pair-discovery feature engineering used by ``MRMR.fit``.

Public functions:

* ``check_prospective_fe_pairs`` -- given high-MI raw pairs, enumerate unary x binary transformation candidates, evaluate each via ``mi_direct``, recommend top-N replacements.
* ``compute_pairs_mis`` -- precompute MI for every pair candidate so the screening loop has a fast lookup.
* ``create_unary_transformations`` / ``create_binary_transformations`` -- preset registries of numpy / scipy.special / hermval functions.
* ``get_existing_feature_name`` / ``get_new_feature_name`` -- string formatters for engineered-feature names.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermval
from scipy import special as sp

from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu


# Domain-validity tags for unary transforms produced by ``create_unary_transformations(preset="maximal")``. Consumers that need to clip / reject inputs before
# applying these transforms can look them up here. Tag vocabulary: ``-1to1``, ``-pi/2topi/2``, ``1toinf``, ``-0.(9)to0.(9)``, ``pos``, ``nonzero``. Transforms not
# listed are unconstrained on real inputs (e.g. ``sin``, ``exp``, ``squared``).
UNARY_INPUT_CONSTRAINTS: dict[str, str] = {
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

from ._internals import njit_functions_dict, smart_log
from .discretization import discretize_array
from .permutation import mi_direct

logger = logging.getLogger(__name__)


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
    # Starting from the most heavily connected pairs, create a big pool of original features + their unary transforms. Individual vars referenced more than once go
    # to the global pool, the rest to the local (not stored)?

    res = {}

    # Exact preallocation. ``n_pairs * n_unary * 2`` over-counts because (var, tr_name) keys are de-duplicated in ``vars_transformations``; the unique-key set is the
    # true upper bound.
    unique_keys: set = set()
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        for var in raw_vars_pair:
            for tr_name in unary_transformations.keys():
                unique_keys.add((var, tr_name))

    if verbose >= 2:
        logger.info(
            "Creating a pool of %d unary transformations for feature engineering "
            "(legacy upper bound was %d).",
            len(unique_keys),
            len(prospective_pairs) * len(unary_transformations) * 2,
        )

    transformed_vars = np.empty(shape=(len(X), len(unique_keys)), dtype=np.float32)

    # Hoist ``final_transformed_vals`` outside the per-pair loop: precompute each pair's ``combs``, find the max length, allocate one shared buffer. Each pair writes
    # then reads the same ``[:, i]`` slice so stale tail data is never observed.
    pair_combs: dict = {}
    max_n_combs = 0
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        combs = list(
            combinations(
                [(raw_vars_pair[0], k) for k in unary_transformations.keys()]
                + [(raw_vars_pair[1], k) for k in unary_transformations.keys()],
                2,
            )
        )
        combs = [tp for tp in combs if tp[0][0] != tp[1][0]]
        pair_combs[raw_vars_pair] = combs
        if len(combs) > max_n_combs:
            max_n_combs = len(combs)

    final_transformed_vals_shared = np.empty(
        shape=(len(X), max_n_combs * len(binary_transformations)),
        dtype=np.float32,
    ) if max_n_combs > 0 else None

    vars_transformations = {}
    i = 0
    for (raw_vars_pair, _pair_mi), _uplift in prospective_pairs.items():
        for var in raw_vars_pair:
            # Polars vs pandas int-column indexing: ``X[:, idx].to_numpy()`` (polars, zero-copy for numerics) vs ``X.iloc[:, idx].values`` (pandas).
            if isinstance(X, pd.DataFrame):
                vals = X.iloc[:, original_cols[var]].values
            else:
                vals = X[:, original_cols[var]].to_numpy()
            for tr_name, tr_func in unary_transformations.items():
                key = (var, tr_name)
                if key not in vars_transformations:
                    try:
                        if "poly_" in tr_name:
                            transformed_vars[:, i] = hermval(vals, c=tr_func)
                        else:
                            transformed_vars[:, i] = tr_func(vals)
                    except Exception as e:
                        # ``np.isnan`` / ``np.isinf`` / ``np.nanmin`` only work on float dtypes. When ``vals`` is object/string (e.g. a polars Utf8 cat column not encoded
                        # before reaching FE), calling them inside the error-log formatter itself raises -- masking the real transformation error and aborting MRMR
                        # entirely. Compute numeric-only diagnostics conditionally.
                        if np.issubdtype(vals.dtype, np.floating):
                            _diag = (
                                f", isnan={np.isnan(vals).sum()}, "
                                f"isinf={np.isinf(vals).sum()}, nanmin={np.nanmin(vals)}"
                            )
                        else:
                            _diag = f", dtype={vals.dtype} (numeric diagnostics skipped)"
                        logger.error(
                            f"Error when performing {tr_name} on array {vals[:5]}, "
                            f"var={cols[var]}: {str(e)}{_diag}"
                        )
                    else:
                        vars_transformations[key] = i
                        i += 1

    if verbose >= 2:
        logger.info("Created. For every pair from the pool, trying all known functions...")

    # For every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM). Record best pairs.
    for (
        raw_vars_pair,
        pair_mi,
    ), _uplift in tqdmu(
        prospective_pairs.items(), desc="pair", leave=False
    ):  # better to start considering form the most prospective pairs with highest mis ratio!

        messages = []

        combs = pair_combs[raw_vars_pair]

        best_config, best_mi = None, -1
        this_pair_features = set()
        var_pairs_perf = {}

        # Shared buffer; this pair uses only the first ``len(combs) * len(binary_transformations)`` columns.
        final_transformed_vals = final_transformed_vals_shared

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
                except Exception:
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
                        x=np.array([0], dtype=np.int64),
                        y=None,
                        factors_nbins=np.array([quantization_nbins], dtype=np.int64),
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

            # If there is a group of leaders with almost the same performance, approve them through one of the other variables.
            # если будут возникать такие группы примерно одинаковых по силе лидеров, их придётся разрешать с помощью одного из других влияющих факторов
            leading_features = []
            for next_config, next_mi in sort_dict_by_value(var_pairs_perf).items():
                if next_mi > best_mi * fe_good_to_best_feature_mi_threshold:
                    leading_features.append(next_config)

            if len(leading_features) > 1:
                if len(numeric_vars_to_consider) > 2:

                    if verbose > 2:
                        print(f"Taking {len(leading_features)} new features for a separate validation step!")

                    # Test all candidates as-is against the rest of the approved factors (also as-is). Candidates significantly outstanding (in terms of MI with target)
                    # against any other approved factor are kept.
                    valid_pairs_perf = {}

                    for transformations_pair, bin_func_name, i in leading_features:
                        param_a = final_transformed_vals[:, i]

                        best_valid_mi = -1
                        config = (transformations_pair, bin_func_name, i)

                        external_factors = list(set(numeric_vars_to_consider) - set(raw_vars_pair))
                        if fe_max_external_validation_factors and len(external_factors) > fe_max_external_validation_factors:
                            external_factors = np.random.choice(external_factors, fe_max_external_validation_factors)

                        for external_factor in tqdmu(external_factors, desc="external validation factor", leave=False):
                            if isinstance(X, pd.DataFrame):
                                param_b = X.iloc[:, original_cols[external_factor]].values
                            else:
                                param_b = X[:, original_cols[external_factor]].to_numpy()

                            for valid_bin_func_name, valid_bin_func in binary_transformations.items():

                                valid_vals = valid_bin_func(param_a, param_b)

                                discretized_transformed_values = discretize_array(
                                    arr=valid_vals, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                                )
                                fe_mi, fe_conf = mi_direct(
                                    discretized_transformed_values.reshape(-1, 1),
                                    x=np.array([0], dtype=np.int64),
                                    y=None,
                                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
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

                    # Recommend proceeding with top N best transformations.
                    for j, (config, _valid_mi) in enumerate(sort_dict_by_value(valid_pairs_perf, reverse=True).items()):
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

                # Bulk add the found & checked best features.
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

                # T3#24 2026-05-18 Pack #5 pair-MI cache: ALWAYS cache the computed pair MI, not only when it passes ``fe_min_pair_mi_prevalence``.
                # Pre-fix this branch dropped sub-threshold MIs on the floor; adaptive retry with relaxed prevalence (Pack #5) then re-computed them.
                # The downstream consumer in MRMR.fit applies the prevalence gate on the cached value, so caching unconditionally preserves filtering behaviour.
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


# GPU broadcast: per-function gpu_compatible flag. Many transformations have direct CuPy equivalents (``np.log`` -> ``cp.log`` etc.); some require ``scipy.special``
# (erfinv, gammaln, ...) which CuPy lacks. The FE driver groups columns into a CuPy tensor and applies the GPU-compatible transformation in one launch, falling back
# to per-column CPU dispatch for the rest.


def gpu_compatible_unary_names() -> set:
    """Names of unary transformations with a direct CuPy equivalent. Anything outside this set falls back to per-column CPU dispatch."""
    return {
        "identity", "sign", "neg", "abs", "rint",
        "squared", "qubed", "reciproc", "invsquared", "invqubed",
        "cbrt", "sqrt", "invcbrt", "invsqrt",
        "log", "exp",
        "sin", "cos", "tan",
        "sinh", "cosh", "tanh",
    }


def gpu_compatible_binary_names() -> set:
    """Same idea for binary transformations."""
    return {
        "add", "sub", "mul", "div", "pow", "min", "max",
        "hypot", "atan2",
    }


def apply_gpu_unary_batched(
    cols_data,
    column_indices: Sequence[int],
    transformation_name: str,
):
    """Apply a unary transformation to a batch of columns on GPU. Returns a CuPy 2-D array with shape ``(n_samples, len(column_indices))``. Only safe for names in
    ``gpu_compatible_unary_names()``.

    Raises:
        ValueError: if the transformation is not GPU-compatible.
        ImportError: if CuPy is not installed.
    """
    import cupy as cp

    name = transformation_name
    if name not in gpu_compatible_unary_names():
        raise ValueError(f"transformation {name!r} is not GPU-compatible")

    if hasattr(cols_data, "to_numpy"):
        cols_np = cols_data.to_numpy() if hasattr(cols_data, "to_numpy") else np.asarray(cols_data)
    else:
        cols_np = np.asarray(cols_data)

    # Slice the requested columns.
    sub = cols_np[:, list(column_indices)] if cols_np.ndim == 2 else cols_np.reshape(-1, 1)
    sub_gpu = cp.asarray(sub.astype(np.float32))

    if name == "identity":
        return sub_gpu
    if name == "sign":
        return cp.sign(sub_gpu)
    if name == "neg":
        return -sub_gpu
    if name == "abs":
        return cp.abs(sub_gpu)
    if name == "rint":
        return cp.rint(sub_gpu)
    if name == "squared":
        return cp.power(sub_gpu, 2)
    if name == "qubed":
        return cp.power(sub_gpu, 3)
    if name == "reciproc":
        return cp.power(sub_gpu, -1)
    if name == "invsquared":
        return cp.power(sub_gpu, -2)
    if name == "invqubed":
        return cp.power(sub_gpu, -3)
    if name == "cbrt":
        return cp.cbrt(sub_gpu)
    if name == "sqrt":
        return cp.sqrt(cp.abs(sub_gpu))
    if name == "invcbrt":
        return cp.power(sub_gpu, -1.0 / 3.0)
    if name == "invsqrt":
        return cp.power(sub_gpu, -0.5)
    if name == "log":
        # Smart log: avoid log of non-positive.
        x_min = cp.nanmin(sub_gpu)
        if x_min > 0:
            return cp.log(sub_gpu)
        return cp.log(sub_gpu + (1e-5 - x_min))
    if name == "exp":
        return cp.exp(sub_gpu)
    if name == "sin":
        return cp.sin(sub_gpu)
    if name == "cos":
        return cp.cos(sub_gpu)
    if name == "tan":
        return cp.tan(sub_gpu)
    if name == "sinh":
        return cp.sinh(sub_gpu)
    if name == "cosh":
        return cp.cosh(sub_gpu)
    if name == "tanh":
        return cp.tanh(sub_gpu)
    raise ValueError(f"GPU dispatch missing for {name!r} despite being in compatible set")


def apply_gpu_binary_batched(
    a_gpu,
    b_gpu,
    transformation_name: str,
):
    """Apply a binary transformation element-wise to two CuPy arrays already on the GPU. Names must be in ``gpu_compatible_binary_names()``."""
    import cupy as cp

    name = transformation_name
    if name not in gpu_compatible_binary_names():
        raise ValueError(f"transformation {name!r} is not GPU-compatible")

    if name == "add":
        return a_gpu + b_gpu
    if name == "sub":
        return a_gpu - b_gpu
    if name == "mul":
        return a_gpu * b_gpu
    if name == "div":
        return a_gpu / b_gpu
    if name == "pow":
        return cp.power(a_gpu, b_gpu)
    if name == "min":
        return cp.minimum(a_gpu, b_gpu)
    if name == "max":
        return cp.maximum(a_gpu, b_gpu)
    if name == "hypot":
        return cp.hypot(a_gpu, b_gpu)
    if name == "atan2":
        return cp.arctan2(a_gpu, b_gpu)
    raise ValueError(f"GPU dispatch missing for binary {name!r}")


def create_unary_transformations(preset: str = "minimal"):
    # Domain-validity tags for each transform live in the module-level ``UNARY_INPUT_CONSTRAINTS`` dict so callers that need to clip / reject inputs can look them up
    # by transform name (e.g. ``arccos`` requires ``-1to1``).
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
            unary_transformations["polygamma_" + str(order)] = lambda x, _order=order: sp.polygamma(_order, x)
            unary_transformations["struve" + str(order)] = lambda x, _order=order: sp.struve(_order, x)
            unary_transformations["jv" + str(order)] = lambda x, _order=order: sp.jv(_order, x)

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
