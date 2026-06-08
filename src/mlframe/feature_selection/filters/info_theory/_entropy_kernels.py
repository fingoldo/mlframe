"""Core entropy / mutual-information kernels: Shannon entropy, Miller-Madow bias correction, MI, Symmetric Uncertainty, and the conditional variants."""
from __future__ import annotations

import numpy as np
from numba import njit

from .._numba_utils import arr2str, unpack_and_sort
from ._class_encoding import merge_vars


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
    # 2026-05-30 Wave 9.1 fix (loop iter 20): guard against k <= 1.
    # When freqs is empty (k=0) or single-bin (k=1) the Miller-Madow
    # correction term ``(k - 1) / (2 * n_samples)`` is either negative
    # (-1/(2n)) or zero. Negative entropy violates the H >= 0
    # invariant and propagates as NEGATIVE conditional MI through
    # ``conditional_mi = H(XZ) + H(YZ) - H(Z) - H(XYZ)`` on degenerate
    # Z-conditioning slices (empty support after MDLP filtering, etc.).
    # Plug-in entropy is exact at k <= 1 (deterministic distribution),
    # so return h_plugin (which is 0) directly.
    if k <= 1:
        return h_plugin
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
