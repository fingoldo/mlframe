"""Information-theoretic primitives: entropy, mutual information, conditional MI.

All functions are ``@njit``-compiled and operate on integer-encoded
arrays produced upstream by :mod:`.discretization`.

Contents
--------
* ``merge_vars``    -- collapse multiple ordinal-encoded variables into a
  single 1-D class array (used as the histogram building block).
* ``entropy``       -- Shannon entropy ``-sum(p * log p)``.
* ``mi``            -- mutual information ``I(X; Y) = H(X) + H(Y) - H(X, Y)``
  computed via entropy decomposition.
* ``conditional_mi`` -- ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)``
  with a pluggable entropy cache. **B4 fix (etap 3)**: previously a single
  ``key`` variable was reused across four cache branches, which only
  happened to be safe by accidental ordering. Each branch now has its own
  ``key_z`` / ``key_xz`` / ``key_yz`` / ``key_xyz`` local. No semantic
  change; the test
  ``mlframe/tests/feature_selection/test_info_theory_cache.py`` enumerates
  all four ``(can_use_x_cache, can_use_y_cache)`` combos and pins down
  exactly which keys land in the cache.
* ``compute_mi_from_classes`` -- mutual information directly from two
  pre-computed class vectors and their marginals (used by the permutation
  loop where ``classes_y`` is shuffled in place).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._numba_utils import arr2str, unpack_and_sort


@njit()
def merge_vars(
    factors_data: np.ndarray,
    vars_indices,
    var_is_nominal,
    factors_nbins,
    dtype=np.int32,
    min_occupancy: int = None,
    current_nclasses: int = 1,
    final_classes: np.ndarray = None,
    verbose: bool = False,
) -> tuple:
    """Melt multiple ordinal-encoded variables into a single 1-D class array.

    ``factors_data`` is laid out as ``(n_samples, n_features)`` (sklearn
    convention). Empty bins are pruned and the class indices renumbered so
    the returned ``final_classes`` is dense (no skipped integers).
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

        nzeros = 0
        lookup_table = np.empty(expected_nclasses, dtype=dtype)
        for oldclass, npoints in enumerate(freqs):
            if npoints == 0:
                nzeros += 1
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


@njit()
def entropy(freqs: np.ndarray, min_occupancy: float = 0) -> float:
    """Shannon entropy in nats. Bins below ``min_occupancy`` (or zero by
    default) are filtered out before the log to avoid ``0 * log(0)``."""
    if min_occupancy:
        freqs = freqs[freqs >= min_occupancy]
    else:
        freqs = freqs[freqs > 0]
    return -(np.log(freqs) * freqs).sum()


@njit()
def entropy_miller_madow(freqs: np.ndarray, n_samples: int, min_occupancy: float = 0) -> float:
    """Miller-Madow bias-corrected Shannon entropy.

    Plug-in entropy ``H_plugin = -sum p log p`` is biased downward when
    the empirical distribution has many bins relative to ``n_samples``
    (the joint X-Y-Z space inflates fast). The Miller-Madow correction
    adds ``(k - 1) / (2 * n_samples)`` where ``k`` is the number of
    non-empty bins. Cheap, asymptotically unbiased, no extra RAM.

    Why it matters for mRMR: in ``conditional_mi(X, Y, Z)`` the
    ``H(X, Y, Z)`` term has the most bins and the steepest bias. Without
    correction, ``I(X;Y|Z)`` is biased *toward zero*, causing false
    rejection of weakly-conditional features. The correction is opt-in
    via ``MRMR(use_miller_madow=True)`` -- default off so the legacy
    plug-in estimator stays bit-exact.

    References:
    * Miller (1955) "Note on the bias of information estimates."
    * Paninski (2003) "Estimation of entropy and mutual information."
    """
    if min_occupancy:
        freqs = freqs[freqs >= min_occupancy]
    else:
        freqs = freqs[freqs > 0]
    h_plugin = -(np.log(freqs) * freqs).sum()
    k = len(freqs)
    return h_plugin + (k - 1) / (2.0 * n_samples)


@njit()
def mi(
    factors_data,
    x: np.ndarray,
    y: np.ndarray,
    factors_nbins: np.ndarray,
    verbose: bool = False,
    dtype=np.int32,
) -> float:
    """Mutual information ``I(X; Y) = H(X) + H(Y) - H(X, Y)`` via entropy."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    factors_nbins = np.asarray(factors_nbins, dtype=np.int64)

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_x = entropy(freqs=freqs_x)
    if verbose:
        print(f"entropy_x={entropy_x}, nclasses_x={len(freqs_x)} ({classes_x.min()} to {classes_x.max()})")

    _, freqs_y, _ = merge_vars(
        factors_data=factors_data, vars_indices=y, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_y = entropy(freqs=freqs_y)
    if verbose:
        print(f"entropy_y={entropy_y}, nclasses_y={len(freqs_y)}")

    vars_xy = np.unique(np.concatenate((x, y)))

    classes_xy, freqs_xy, _ = merge_vars(
        factors_data=factors_data, vars_indices=vars_xy, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_xy = entropy(freqs=freqs_xy)
    if verbose:
        print(f"entropy_xy={entropy_xy}, nclasses_x={len(freqs_xy)} ({classes_xy.min()} to {classes_xy.max()})")

    return entropy_x + entropy_y - entropy_xy


@njit()
def conditional_mi(
    factors_data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    var_is_nominal: np.ndarray,
    factors_nbins: np.ndarray,
    entropy_z: float = -1.0,
    entropy_xz: float = -1.0,
    entropy_yz: float = -1.0,
    entropy_xyz: float = -1.0,
    entropy_cache: dict = None,
    can_use_x_cache: bool = False,
    can_use_y_cache: bool = False,
    dtype=np.int32,
) -> float:
    """Conditional mutual information ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)``.

    B4 fix: each cache branch owns its own key local (``key_z`` / ``key_xz``
    / ``key_yz`` / ``key_xyz``). The legacy code reused a single ``key``
    across all four branches; while accidentally safe, that pattern was a
    foot-gun for future edits.
    """
    key_z = ""
    key_xz = ""
    key_yz = ""
    key_xyz = ""

    if entropy_z < 0:
        if entropy_cache is not None:
            key_z = arr2str(sorted(z))
            entropy_z = entropy_cache.get(key_z, -1)
        if entropy_z < 0:
            _, freqs_z, _ = merge_vars(
                factors_data=factors_data, vars_indices=z, var_is_nominal=None,
                factors_nbins=factors_nbins, dtype=dtype,
            )
            entropy_z = entropy(freqs=freqs_z)
            if entropy_cache is not None:
                entropy_cache[key_z] = entropy_z

    if entropy_xz < 0:
        indices = unpack_and_sort(x, z)
        if can_use_x_cache and entropy_cache is not None:
            key_xz = arr2str(indices)
            entropy_xz = entropy_cache.get(key_xz, -1)
        if entropy_xz < 0:
            _, freqs_xz, _ = merge_vars(
                factors_data=factors_data, vars_indices=indices, var_is_nominal=None,
                factors_nbins=factors_nbins, dtype=dtype,
            )
            entropy_xz = entropy(freqs=freqs_xz)
            if can_use_x_cache and entropy_cache is not None:
                entropy_cache[key_xz] = entropy_xz

    current_nclasses_yz = 1
    if can_use_y_cache:
        if entropy_yz < 0:
            indices = unpack_and_sort(y, z)
            if entropy_cache is not None:
                key_yz = arr2str(indices)
                entropy_yz = entropy_cache.get(key_yz, -1)
            if entropy_yz < 0:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
                    factors_data=factors_data, vars_indices=indices, var_is_nominal=None,
                    factors_nbins=factors_nbins, dtype=dtype,
                )
                entropy_yz = entropy(freqs=freqs_yz)
                if entropy_cache is not None:
                    entropy_cache[key_yz] = entropy_yz
    else:
        classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
            factors_data=factors_data, vars_indices=unpack_and_sort(y, z),
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )
        entropy_yz = entropy(freqs=freqs_yz)

    if entropy_xyz < 0:
        if can_use_y_cache and can_use_x_cache:
            indices = unpack_and_sort(x, y)
            indices = unpack_and_sort(indices, z)
            if entropy_cache is not None:
                key_xyz = arr2str(indices)
                entropy_xyz = entropy_cache.get(key_xyz, -1)
        if entropy_xyz < 0:
            if current_nclasses_yz == 1:
                classes_yz, freqs_yz, current_nclasses_yz = merge_vars(
                    factors_data=factors_data, vars_indices=unpack_and_sort(y, z),
                    var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
                )

            _, freqs_xyz, _ = merge_vars(
                factors_data=factors_data,
                vars_indices=x,
                var_is_nominal=None,
                factors_nbins=factors_nbins,
                current_nclasses=current_nclasses_yz,
                final_classes=classes_yz,
                dtype=dtype,
            )
            entropy_xyz = entropy(freqs=freqs_xyz)
            if entropy_cache is not None and can_use_y_cache and can_use_x_cache:
                entropy_cache[key_xyz] = entropy_xyz

    res = entropy_xz + entropy_yz - entropy_z - entropy_xyz
    if res < 0.0:
        res = 0.0
    return res


@njit(cache=True)
def compute_mi_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:
    """Mutual information from two pre-computed class arrays + their marginals.

    Used by the permutation loop where ``classes_y`` is shuffled in place
    and we don't want to re-bin from scratch each time.
    """
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
