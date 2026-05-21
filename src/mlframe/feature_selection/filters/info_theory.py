"""Information-theoretic primitives: entropy, mutual information, conditional MI.

All functions are ``@njit``-compiled and operate on integer-encoded arrays produced upstream by :mod:`.discretization`.

Contents
--------
* ``merge_vars``    -- collapse multiple ordinal-encoded variables into a single 1-D class array (used as the histogram building block).
* ``entropy``       -- Shannon entropy ``-sum(p * log p)``.
* ``mi``            -- mutual information ``I(X; Y) = H(X) + H(Y) - H(X, Y)`` computed via entropy decomposition.
* ``conditional_mi`` -- ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)`` with a pluggable entropy cache. Each cache branch owns its own
  ``key_z`` / ``key_xz`` / ``key_yz`` / ``key_xyz`` local; ``test_info_theory_cache.py`` enumerates all four ``(can_use_x_cache, can_use_y_cache)`` combos and
  pins down exactly which keys land in the cache.
* ``compute_mi_from_classes`` -- mutual information directly from two pre-computed class vectors and their marginals (used by the permutation loop where
  ``classes_y`` is shuffled in place).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._numba_utils import arr2str, unpack_and_sort


@njit(cache=True)
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

    ``factors_data`` is laid out as ``(n_samples, n_features)`` (sklearn convention). Empty bins are pruned and the class indices renumbered so the returned
    ``final_classes`` is dense (no skipped integers).

    bench-attempt-rejected (2026-05-21, c0148 / iter136): 1-var
    fast-path replacing the per-row loop with np.bincount + cast runs
    15-20% SLOWER at n=50k..1M -- numba already compiles the manual
    loop into tight machine code; the bincount path pays an int64->
    dtype cast plus an extra .copy() for the no-prune branch. Bench:
    profiling/bench_merge_vars_1var_fastpath.py.
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


@njit(cache=True)
def entropy(freqs: np.ndarray, min_occupancy: float = 0) -> float:
    """Shannon entropy in nats. Bins below ``min_occupancy`` (or zero by default) are filtered out before the log to avoid ``0 * log(0)``."""
    if min_occupancy:
        freqs = freqs[freqs >= min_occupancy]
    else:
        freqs = freqs[freqs > 0]
    return -(np.log(freqs) * freqs).sum()


@njit(cache=True)
def entropy_miller_madow(freqs: np.ndarray, n_samples: int, min_occupancy: float = 0) -> float:
    """Miller-Madow bias-corrected Shannon entropy.

    Plug-in entropy ``H_plugin = -sum p log p`` is biased downward when the empirical distribution has many bins relative to ``n_samples`` (the joint X-Y-Z space inflates fast). The Miller-Madow correction adds ``(k - 1) / (2 * n_samples)`` where ``k`` is the number of non-empty bins. Cheap, asymptotically unbiased, no extra RAM.

    Why it matters for mRMR: in ``conditional_mi(X, Y, Z)`` the ``H(X, Y, Z)`` term has the most bins and the steepest bias. Without correction, ``I(X;Y|Z)`` is biased *toward zero*, causing false rejection of weakly-conditional features. Opt-in via ``MRMR(use_miller_madow=True)`` -- default off so the legacy plug-in estimator stays bit-exact.

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


@njit(cache=True)
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


@njit(cache=True)
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

    Each cache branch owns its own key local (``key_z`` / ``key_xz`` / ``key_yz`` / ``key_xyz``) to avoid cross-branch aliasing.
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
    """Mutual information from two pre-computed class arrays + their marginals. Used by the permutation loop where ``classes_y`` is shuffled in place and we don't want to re-bin from scratch each time.

    Indexed range-loop over zip-iter: ~25% faster at n=50k..1M (numba's
    zip pair unboxing adds per-iteration overhead vs the indexed form).
    On-the-fly freq computation via inv_n also avoids the (K_x, K_y)
    float64 joint_freqs intermediate allocation. Bench:
    profiling/bench_compute_mi_from_classes_no_zip.py.
    """
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    joint_counts = np.zeros((K_x, K_y), dtype=dtype)
    for k in range(n):
        joint_counts[classes_x[k], classes_y[k]] += 1
    inv_n = 1.0 / n

    total = 0.0
    for i in range(K_x):
        prob_x = freqs_x[i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                total += jf * math.log(jf / (prob_x * prob_y))
    return total


@njit(parallel=True, nogil=True, cache=True)
def batch_pair_mi_prange(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> np.ndarray:
    """Vectorised batch MI computation over an array of (a, b) variable-index pairs.

    Replaces the ``joblib.Parallel(delayed(compute_pairs_mis)(...))`` outer loop in
    :meth:`MRMR._run_fe_step` with a single numba ``prange`` over all pairs. Each
    iteration:

      1. Builds the joint class encoding for ``(factors_data[:, a], factors_data[:, b])``
         in-place into a thread-local buffer (no Python objects, no dict lookups,
         GIL released throughout).
      2. Counts joint-class frequencies.
      3. Calls :func:`compute_mi_from_classes` with the just-computed marginal.

    Pre-fix (iter-371 1M cb multiclass + MRMR + binary=medium) used joblib loky
    with per-worker mmap of the entire ``factors_data`` array; threading (Layer 1)
    already removed the copy cost, this kernel removes the joblib threadpool
    dispatch overhead AND the Python wrapper GIL contention between numba calls.
    Pair MI now scales bit-for-bit linearly across physical cores.

    Notes:
      * No permutation testing is run here. Callers that need a confidence gate
        should fall back to per-pair ``mi_direct`` afterwards for the surviving
        candidates -- the bench harness shows that's a small fraction of pairs.
      * ``factors_data`` MUST be pre-binned (categorical / ordinal int dtype);
        ``categorize_dataset`` produces this format. Passing a float matrix
        gives undefined results.
      * The joint encoding uses ``a * nbins[b] + b`` which is monotone in
        ``(a, b)`` -- the same convention :func:`merge_vars` uses, so this
        kernel and the legacy path produce numerically identical MIs on the
        same inputs (verified by ``test_batch_pair_mi_prange_matches_merge_vars``).
    """
    n_samples = factors_data.shape[0]
    n_pairs = pair_a.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    n_classes_y = freqs_y.shape[0]

    # Wave 47 (2026-05-20): empty factors_data divides by zero in `1/n_samples`
    # inside the per-pair MI inner loop; return zeros (no-information baseline).
    if n_samples == 0:
        out[:] = 0.0
        return out

    for p in prange(n_pairs):
        a = pair_a[p]
        b = pair_b[p]
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])
        joint_card = nb_a * nb_b

        # Thread-local buffers. numba allocates these per-prange-iteration on the
        # worker thread's stack so there's no cross-thread aliasing.
        joint_counts = np.zeros((joint_card, n_classes_y), dtype=np.int64)
        freqs_x_int = np.zeros(joint_card, dtype=np.int64)

        for i in range(n_samples):
            va = int(factors_data[i, a])
            vb = int(factors_data[i, b])
            cls_x = va * nb_b + vb
            cls_y = int(classes_y[i])
            joint_counts[cls_x, cls_y] += 1
            freqs_x_int[cls_x] += 1

        # Direct MI computation (inlined compute_mi_from_classes core to avoid a
        # cross-kernel call that would require classes_x to be materialised as a
        # 1-D array).
        total = 0.0
        inv_n = 1.0 / n_samples
        for i in range(joint_card):
            fx = freqs_x_int[i]
            if fx == 0:
                continue
            prob_x = fx * inv_n
            for j in range(n_classes_y):
                jc = joint_counts[i, j]
                if jc == 0:
                    continue
                jf = jc * inv_n
                prob_y = freqs_y[j]
                if prob_y > 0.0:
                    total += jf * math.log(jf / (prob_x * prob_y))
        out[p] = total

    return out
