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


@njit(nogil=True, cache=True)
def symmetric_uncertainty(
    factors_data,
    x: np.ndarray,
    y: np.ndarray,
    factors_nbins: np.ndarray,
    verbose: bool = False,
    dtype=np.int32,
) -> float:
    """Symmetric Uncertainty (Witten-Frank-Hall) — cardinality-normalised MI.

    ``SU(X, Y) := 2 * I(X; Y) / (H(X) + H(Y))`` lives in [0, 1] independently of
    the cardinality of X or Y. Raw ``I(X; Y)`` is bounded by ``min(H(X), H(Y))``
    so high-cardinality features (zip codes, hash IDs, decile-binned continuous
    columns with many bins) get inflated relevance scores under the bare ``mi``
    estimator. SU divides by the sum of marginal entropies, scrubbing the bias.

    Two same-entropy features keep the same MRMR ordering under SU vs MI (both
    get divided by the same denominator). Cross-cardinality comparisons are
    where SU bites: a 2-level binary feature with strong signal beats a
    1000-level hash that happens to have higher raw MI from sheer entropy.

    Used when ``MRMR(mi_normalization='su')``. Same numba cache as ``mi``.
    Reference: Witten, Frank, Hall (2011) "Data Mining: Practical ML Tools and
    Techniques", section on feature selection.
    """
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    factors_nbins = np.asarray(factors_nbins, dtype=np.int64)

    _, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_x = entropy(freqs=freqs_x)

    _, freqs_y, _ = merge_vars(
        factors_data=factors_data, vars_indices=y, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_y = entropy(freqs=freqs_y)

    vars_xy = np.unique(np.concatenate((x, y)))
    _, freqs_xy, _ = merge_vars(
        factors_data=factors_data, vars_indices=vars_xy, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_xy = entropy(freqs=freqs_xy)

    mi_xy = entropy_x + entropy_y - entropy_xy
    denom = entropy_x + entropy_y
    if denom <= 1e-12:
        return 0.0
    return 2.0 * mi_xy / denom


@njit(nogil=True, cache=True)
def conditional_symmetric_uncertainty(
    factors_data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    factors_nbins: np.ndarray,
    dtype=np.int32,
) -> float:
    """Conditional Symmetric Uncertainty (CSU): ``2 * I(X; Y | Z) / (H(X|Z) + H(Y|Z))``.

    Normalised conditional MI used by the Fleuret-criterion redundancy step
    when ``MRMR(mi_normalization='su')`` is active. Reduces cardinality bias
    in the conditional information path the same way ``symmetric_uncertainty``
    does for the unconditional path: a high-cardinality Z silently inflates
    raw ``I(X; Y | Z)`` because ``H(X|Z)`` and ``H(Y|Z)`` are still bounded
    by their unconditional H counterparts.

    Formula: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
             H(X|Z)   = H(X,Z) - H(Z)
             H(Y|Z)   = H(Y,Z) - H(Z)
             denom    = H(X|Z) + H(Y|Z) = H(X,Z) + H(Y,Z) - 2*H(Z)
    """
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    z = np.asarray(z, dtype=np.int64)
    factors_nbins = np.asarray(factors_nbins, dtype=np.int64)

    _, freqs_z, _ = merge_vars(
        factors_data=factors_data, vars_indices=z, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_z = entropy(freqs=freqs_z)

    xz = np.unique(np.concatenate((x, z)))
    _, freqs_xz, _ = merge_vars(
        factors_data=factors_data, vars_indices=xz, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_xz = entropy(freqs=freqs_xz)

    yz = np.unique(np.concatenate((y, z)))
    _, freqs_yz, _ = merge_vars(
        factors_data=factors_data, vars_indices=yz, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_yz = entropy(freqs=freqs_yz)

    xyz = np.unique(np.concatenate((x, y, z)))
    _, freqs_xyz, _ = merge_vars(
        factors_data=factors_data, vars_indices=xyz, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_xyz = entropy(freqs=freqs_xyz)

    cmi = h_xz + h_yz - h_z - h_xyz
    denom = h_xz + h_yz - 2.0 * h_z
    if denom <= 1e-12:
        return 0.0
    return 2.0 * cmi / denom


# 2026-05-28: thread-local SU toggle. Set by MRMR.fit when mi_normalization='su',
# read by evaluation.py / fleuret.py at the scoring sites. Pure functions ``mi``
# and ``conditional_mi`` stay legacy bit-for-bit so cached entropy numbers
# across the rest of the project don't shift.
import threading as _threading
_SU_STATE = _threading.local()


def use_su_normalization() -> bool:
    return bool(getattr(_SU_STATE, "active", False))


def set_su_normalization(active: bool) -> None:
    _SU_STATE.active = bool(active)


# 2026-05-30 Wave 8 — JMIM / BUR thread-local toggles. Set by MRMR.fit when
# ``redundancy_aggregator='jmim'`` or ``bur_lambda > 0``; read at the
# evaluation hot-path. Same pattern as SU above. Independent of SU so the
# three can compose freely.
_JMIM_STATE = _threading.local()
_BUR_STATE = _threading.local()


def use_jmim_aggregator() -> bool:
    return bool(getattr(_JMIM_STATE, "active", False))


def set_jmim_aggregator(active: bool) -> None:
    _JMIM_STATE.active = bool(active)


def get_bur_lambda() -> float:
    """Returns the current thread-local BUR weight (0.0 = off)."""
    return float(getattr(_BUR_STATE, "weight", 0.0))


def set_bur_lambda(weight: float) -> None:
    _BUR_STATE.weight = float(weight)


def mi_or_su(factors_data, x, y, factors_nbins, verbose=False, dtype=np.int32) -> float:
    """Dispatch raw MI or SU based on the thread-local toggle. Cheap path
    when SU is off: a one-call delegation to ``mi`` (which is njit-cached)."""
    if use_su_normalization():
        return symmetric_uncertainty(factors_data, x, y, factors_nbins, verbose=verbose, dtype=dtype)
    return mi(factors_data, x, y, factors_nbins, verbose=verbose, dtype=dtype)


def cmi_or_csu(factors_data, x, y, z, factors_nbins, dtype=np.int32, **mi_kwargs) -> float:
    """Dispatch conditional MI or CSU based on the thread-local toggle. ``conditional_mi``
    accepts a richer kwarg surface (cache, can_use_x_cache, ...); when SU is on those caches
    are bypassed because the SU denominator is path-dependent on the same entropies, so
    caching the unconditional CMI would silently desync."""
    if use_su_normalization():
        return conditional_symmetric_uncertainty(factors_data, x, y, z, factors_nbins, dtype=dtype)
    return conditional_mi(factors_data, x, y, z, factors_nbins=factors_nbins, dtype=dtype, **mi_kwargs)


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


@njit(nogil=True, cache=True)
def compute_su_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:
    """Symmetric Uncertainty from pre-computed class arrays + marginals.

    SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y)). Built atop the same joint-counts
    pass as ``compute_mi_from_classes`` so the permutation loop in
    ``permutation.py`` can swap to this scorer when ``mi_normalization='su'``
    without recomputing classes/freqs. Reuses the freqs_x / freqs_y arrays
    to compute H(X), H(Y) -- one log-pass per marginal, O(K_x + K_y).
    """
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    joint_counts = np.zeros((K_x, K_y), dtype=dtype)
    for k in range(n):
        joint_counts[classes_x[k], classes_y[k]] += 1
    inv_n = 1.0 / n
    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_x[i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))
    h_x = 0.0
    for i in range(K_x):
        p = freqs_x[i]
        if p > 0:
            h_x -= p * math.log(p)
    h_y = 0.0
    for j in range(K_y):
        p = freqs_y[j]
        if p > 0:
            h_y -= p * math.log(p)
    denom = h_x + h_y
    if denom <= 1e-12:
        return 0.0
    return 2.0 * mi_xy / denom


@njit(nogil=True, cache=True)
def compute_relevance_score(
    use_su: bool,
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:
    """njit-callable dispatcher between raw MI and Symmetric Uncertainty.

    ``permutation.py``'s njit kernels cannot read the Python-level thread-local
    SU toggle directly; this branch-on-flag helper lets the joblib entry point
    (``mi_direct``) thread the SU mode down once per call, and the njit kernel
    selects the scorer at runtime with a single bool check.

    Both branches share the same dtype + array contracts as
    ``compute_mi_from_classes`` so the existing permutation loops stay
    byte-for-byte stable in the SU-off code path.
    """
    if use_su:
        return compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype)
    return compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype)


def mi_or_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=np.int32) -> float:
    """Dispatch raw MI or SU from pre-computed classes based on the thread-local toggle.

    Cheap when SU is off: one Python-call delegation to the njit ``compute_mi_from_classes``.
    Used by ``permutation.py`` so the relevance gate in MRMR's simple-mode path picks up
    the cardinality-bias-corrected scorer when ``MRMR(mi_normalization='su')``.
    """
    if use_su_normalization():
        return compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype)
    return compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype)


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
