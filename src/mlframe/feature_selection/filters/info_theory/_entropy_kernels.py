"""Core entropy / mutual-information kernels: Shannon entropy, Miller-Madow bias correction, MI, Symmetric Uncertainty, and the conditional variants."""
from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

from .._numba_utils import arr2str, unpack_and_sort
from ._class_encoding import merge_vars, joint_freqs_2var


@njit(cache=True)
def entropy(freqs: np.ndarray, min_occupancy: float = 0) -> float:
    """Shannon entropy in nats. Bins below ``min_occupancy`` (or zero by default) are filtered out before the log to avoid ``0 * log(0)``."""
    if min_occupancy:
        freqs = freqs[freqs >= min_occupancy]
    else:
        freqs = freqs[freqs > 0]
    return float(-(np.log(freqs) * freqs).sum())


@njit(cache=True)
def _merge_vars_freqs(factors_data, vars_indices, factors_nbins, dtype=np.int32) -> np.ndarray:
    """Normalized NONZERO joint-class frequencies of ``vars_indices`` -- BIT-IDENTICAL to
    ``merge_vars(factors_data, vars_indices, None, factors_nbins, dtype)[1]`` but WITHOUT allocating a
    caller-visible ``final_classes`` array and WITHOUT the LAST variable's ``final_classes`` relabel pass.

    ``conditional_mi``'s H(X,Z) melt discards ``merge_vars``'s ``final_classes`` output entirely -- only the
    ``freqs`` feed ``entropy``. ``merge_vars`` still writes ``final_classes`` on the final variable and then
    remaps it (a full extra length-``n`` pass) purely to hand back a relabel array the CMI path throws away.
    This kernel keeps the incremental encoding + inter-variable pruning IDENTICAL (so the produced ``freqs``
    order/values are byte-for-byte those of ``merge_vars``, preserving the hoist's merge-order contract) but
    skips the final variable's ``final_classes`` write + remap. The freqs order (ascending class id, empty
    bins pruned) is unchanged, so ``entropy(...)`` over it is bit-identical.
    """
    n_rows = len(factors_data)
    if n_rows == 0:
        return np.zeros(0, dtype=np.float64)
    nvars = len(vars_indices)
    final_classes = np.zeros(n_rows, dtype=dtype)
    current_nclasses = 1
    freqs = np.zeros(1, dtype=np.int64)
    for var_number in range(nvars):
        var_index = vars_indices[var_number]
        expected_nclasses = current_nclasses * factors_nbins[var_index]
        freqs = np.zeros(expected_nclasses, dtype=np.int64)
        values = factors_data[:, var_index].astype(dtype)
        is_last = var_number == nvars - 1
        if is_last:
            # H(.) only needs the pruned freqs; the final relabel array is discarded by the caller, so skip it.
            for sample_row in range(n_rows):
                newclass = final_classes[sample_row] + values[sample_row] * current_nclasses
                freqs[newclass] += 1
            nzeros = 0
            for oldclass in range(expected_nclasses):
                if freqs[oldclass] == 0:
                    nzeros += 1
            if nzeros:
                freqs = freqs[freqs > 0]
        else:
            for sample_row in range(n_rows):
                newclass = final_classes[sample_row] + values[sample_row] * current_nclasses
                freqs[newclass] += 1
                final_classes[sample_row] = newclass
            nzeros = 0
            lookup_table = np.empty(expected_nclasses, dtype=np.int64)
            for oldclass in range(expected_nclasses):
                if freqs[oldclass] == 0:
                    nzeros += 1
                lookup_table[oldclass] = oldclass - nzeros
            if nzeros:
                for sample_row in range(n_rows):
                    final_classes[sample_row] = lookup_table[final_classes[sample_row]]
            current_nclasses = expected_nclasses - nzeros
    return freqs / n_rows


@njit(cache=True)
def _entropy_xz_fused(factors_data, indices, factors_nbins, dtype=np.int32) -> float:
    """``H`` of the joint over ``indices`` (the sorted X u Z union), fused freqs-only. For the common
    2-variable union it uses the single-pass ``joint_freqs_2var`` (no ``final_classes`` at all); otherwise the
    general ``_merge_vars_freqs`` fast path. Bit-identical to ``entropy(merge_vars(...)[1])`` by construction
    (same freqs, same ``entropy`` reduction)."""
    if len(indices) == 2:
        return float(entropy(joint_freqs_2var(factors_data, indices[0], indices[1], factors_nbins[indices[0]], factors_nbins[indices[1]])))
    return float(entropy(_merge_vars_freqs(factors_data, indices, factors_nbins, dtype)))


@njit(cache=True)
def _entropy_x_onto_classes(factors_data, xi, final_classes, current_nclasses, nb_x) -> float:
    """``H(X, Y, Z)`` melting a single candidate column ``xi`` onto the precomputed (Y,Z) class labels
    ``final_classes`` (``current_nclasses`` distinct labels). BIT-IDENTICAL to
    ``entropy(merge_vars(factors_data, [xi], None, factors_nbins, current_nclasses=current_nclasses,
    final_classes=final_classes)[1])`` but WITHOUT mutating ``final_classes`` and WITHOUT the discarded
    relabel/remap passes: it histograms ``final_classes[row] + x[row]*current_nclasses`` once, prunes empty
    bins (ascending class id -- the same order ``merge_vars`` produces), and reduces to entropy.

    ``final_classes`` is read-only here (``merge_vars`` overwrites it in place; the CMI caller discards it
    afterwards, so not mutating it is a strict correctness gain as well as a saved length-n remap pass)."""
    n_rows = len(factors_data)
    if n_rows == 0:
        return 0.0
    expected = current_nclasses * nb_x
    freqs = np.zeros(expected, dtype=np.int64)
    for sample_row in range(n_rows):
        newclass = final_classes[sample_row] + factors_data[sample_row, xi] * current_nclasses
        freqs[newclass] += 1
    nz = freqs[freqs > 0]
    return float(entropy(nz / n_rows))


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
        return float(h_plugin)
    return float(h_plugin + (k - 1) / (2.0 * n_samples))


@njit(cache=True)
def mi_miller_madow_correct(mi_plugin: float, k_x: int, k_y: int, n_samples: int) -> float:
    """Miller-Madow bias-correct a PLUG-IN mutual-information estimate.

    The plug-in MI ``I_hat(X;Y) = H_hat(X) + H_hat(Y) - H_hat(X,Y)`` carries a
    POSITIVE finite-sample bias because the joint ``H(X,Y)`` term is the most
    over-binned. Carrying the Miller-Madow ``(k-1)/(2n)`` entropy correction
    through ``I = H(X)+H(Y)-H(X,Y)`` gives the closed-form MI correction
    ``bias = (k_x - 1)(k_y - 1)/(2n)`` (the marginal corrections cancel against the
    joint one down to the product term). Subtracting it yields the Miller-Madow MI
    estimate ``I_mm = I_plugin - (k_x-1)(k_y-1)/(2n)``.

    ``k_x`` / ``k_y`` are the OCCUPIED (non-empty) bin counts of X and Y -- the SAME
    ``k = #{bins with count>0}`` that :func:`entropy_miller_madow` uses internally
    (backlog #4: nominal ``nbins`` over-corrects heavy-tailed columns that collapse
    to few occupied bins). The bias term is zero when either variable is degenerate
    (``k <= 1``), so the plug-in value passes through unchanged there.

    Used by the FE joint-prevalence gate: the numerator (1-D engineered MI over
    ~``nbins`` bins) and the denominator (2-D joint MI over ~``nbins^2`` bins) carry
    bias terms differing by ~``nbins``x, so the RAW ratio ``best_mi/pair_mi`` is
    structurally depressed below 1.0 even when the 1-D feature captures all the joint
    information -- worst at small/moderate ``n``. MM-correcting BOTH sides before the
    ratio removes that asymmetry. ``->0`` as ``n -> inf``, so large-n selection is
    byte-untouched. References: Miller (1955); Paninski (2003).
    """
    if k_x <= 1 or k_y <= 1:
        return mi_plugin
    return mi_plugin - (k_x - 1) * (k_y - 1) / (2.0 * n_samples)


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

    return float(entropy_x + entropy_y - entropy_xy)


@njit(cache=True)
def mi_miller_madow(
    factors_data,
    x: np.ndarray,
    y: np.ndarray,
    factors_nbins: np.ndarray,
    verbose: bool = False,
    dtype=np.int32,
) -> float:
    """Miller-Madow bias-corrected mutual information ``I_mm(X; Y) = I_plugin - (k_x-1)(k_y-1)/(2n)``.

    The plug-in ``I(X;Y) = H(X)+H(Y)-H(X,Y)`` carries a POSITIVE finite-sample bias dominated by the over-binned joint term, scaling with the OCCUPIED bin
    counts ``k_x``/``k_y``. This kernel computes the plug-in MI from the SAME occupied-bin frequencies and subtracts the closed-form Miller-Madow bias so a
    high-cardinality NOISE feature no longer out-ranks a low-cardinality TRUE-relevant feature by sheer entropy at small n. ``-> I_plugin`` as ``n -> inf``.
    """
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    factors_nbins = np.asarray(factors_nbins, dtype=np.int64)

    _, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    _, freqs_y, _ = merge_vars(
        factors_data=factors_data, vars_indices=y, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    vars_xy = np.unique(np.concatenate((x, y)))
    _, freqs_xy, _ = merge_vars(
        factors_data=factors_data, vars_indices=vars_xy, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    entropy_x = entropy(freqs=freqs_x)
    entropy_y = entropy(freqs=freqs_y)
    entropy_xy = entropy(freqs=freqs_xy)
    mi_plugin = entropy_x + entropy_y - entropy_xy
    k_x = len(freqs_x[freqs_x > 0])
    k_y = len(freqs_y[freqs_y > 0])
    n_samples = factors_data.shape[0] if factors_data.ndim > 1 else len(factors_data)
    return float(mi_miller_madow_correct(mi_plugin, k_x, k_y, n_samples))


@njit(cache=True)
def entropy_chao_shen(freqs: np.ndarray, n: int) -> float:
    """Chao & Shen (2003) coverage-adjusted entropy from a normalized frequency vector ``freqs``
    (``freqs[i] = count_i / n``, as returned by ``merge_vars``) and the sample size ``n``. Recovers
    integer per-bin counts via ``round(freqs * n)`` (exact for the ``count/n`` divisions ``merge_vars``
    produces) and re-estimates entropy with the coverage correction ``C_hat = 1 - f1/n`` (``f1`` = number
    of singleton bins) instead of the plug-in estimator's implicit ``C_hat=1``. See finding #7,
    05_concurrency_and_statistics.md."""
    occ = freqs[freqs > 0]
    counts = np.rint(occ * n).astype(np.int64)
    f1 = 0
    for i in range(len(counts)):
        if counts[i] == 1:
            f1 += 1
    if f1 >= n:
        c_hat = (n - 1.0) / n if n > 1 else 1.0
    else:
        c_hat = 1.0 - f1 / n
    h = 0.0
    for i in range(len(counts)):
        ni = counts[i]
        if ni <= 0:
            continue
        p_tilde = c_hat * ni / n
        if p_tilde <= 0.0:
            continue
        lam = 1.0 - (1.0 - p_tilde) ** n
        term = -p_tilde * np.log(p_tilde)
        if lam > 1e-12:
            term = term / lam
        h += term
    return float(h)


@njit(cache=True)
def mi_chao_shen(
    factors_data,
    x: np.ndarray,
    y: np.ndarray,
    factors_nbins: np.ndarray,
    verbose: bool = False,
    dtype=np.int32,
) -> float:
    """Chao-Shen (2003) coverage-adjusted mutual information ``I_cs(X;Y) = H_cs(X) + H_cs(Y) -
    H_cs(X,Y)``, floored at 0. Unlike ``mi_miller_madow``'s closed-form additive bias term (a function
    only of occupied-bin COUNTS), Chao-Shen re-estimates each entropy term from its own observed-category
    coverage -- better tracks bias on sparse high-cardinality joints (many singleton cells) where
    Miller-Madow's bias term systematically under-corrects."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    factors_nbins = np.asarray(factors_nbins, dtype=np.int64)

    _, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    _, freqs_y, _ = merge_vars(
        factors_data=factors_data, vars_indices=y, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    vars_xy = np.unique(np.concatenate((x, y)))
    _, freqs_xy, _ = merge_vars(
        factors_data=factors_data, vars_indices=vars_xy, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=verbose, dtype=dtype,
    )
    n_samples = factors_data.shape[0] if factors_data.ndim > 1 else len(factors_data)
    h_x = entropy_chao_shen(freqs_x, n_samples)
    h_y = entropy_chao_shen(freqs_y, n_samples)
    h_xy = entropy_chao_shen(freqs_xy, n_samples)
    mi_cs = h_x + h_y - h_xy
    return float(mi_cs) if mi_cs > 0.0 else 0.0


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
    estimator. SU divides by the sum of marginal entropies, normalising AWAY the
    cardinality-driven magnitude inflation. (This rescales for cardinality only --
    it does NOT remove the finite-sample plug-in MI bias in the numerator; for that
    use a Miller-Madow / debiased MI estimator.)

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
    # Floor the plug-in numerator at 0: on near-deterministic columns float round-off in ``H(X)+H(Y)-H(XY)`` can leave ``mi_xy`` slightly negative, yielding a tiny negative SU treated as a valid low relevance instead of 0.
    if mi_xy < 0.0:
        mi_xy = 0.0
    return float(2.0 * mi_xy / denom)


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

    n_samples = factors_data.shape[0] if factors_data.ndim > 1 else len(factors_data)

    # Both the CMI numerator and the conditional-entropy normalizer are built from Miller-Madow-corrected entropy
    # terms. The plug-in entropies over-bin worst on the high-cardinality joints (X,Z), (Y,Z), (X,Y,Z); a plug-in
    # numerator over a plug-in denominator does NOT cancel that bias (the joint terms carry steeper bias than H(Z)),
    # so on an independent-given-Z pair the bare ratio sits well above 0. Routing every entropy through
    # ``entropy_miller_madow`` debiases both sides consistently so SU(X;Y|Z) -> 0 there.
    _, freqs_z, _ = merge_vars(
        factors_data=factors_data, vars_indices=z, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_z = entropy_miller_madow(freqs_z, n_samples)

    xz = np.unique(np.concatenate((x, z)))
    _, freqs_xz, _ = merge_vars(
        factors_data=factors_data, vars_indices=xz, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_xz = entropy_miller_madow(freqs_xz, n_samples)

    yz = np.unique(np.concatenate((y, z)))
    _, freqs_yz, _ = merge_vars(
        factors_data=factors_data, vars_indices=yz, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_yz = entropy_miller_madow(freqs_yz, n_samples)

    xyz = np.unique(np.concatenate((x, y, z)))
    _, freqs_xyz, _ = merge_vars(
        factors_data=factors_data, vars_indices=xyz, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    h_xyz = entropy_miller_madow(freqs_xyz, n_samples)

    cmi = h_xz + h_yz - h_z - h_xyz
    denom = h_xz + h_yz - 2.0 * h_z
    if denom <= 1e-12:
        return 0.0
    # Floor the conditional-MI numerator at 0 (matching the ``conditional_mi`` clamp): float round-off in ``H(XZ)+H(YZ)-H(Z)-H(XYZ)`` on near-deterministic slices can leave ``cmi`` slightly negative, yielding a tiny negative CSU instead of 0.
    if cmi < 0.0:
        cmi = 0.0
    csu = 2.0 * cmi / denom
    # CSU is bounded by [0, 1] in exact arithmetic (I(X;Y|Z) <= min(H(X|Z), H(Y|Z)) <= denom/2). A value above 1
    # is a numerical artifact -- a tiny-but-positive denom just above the 1e-12 guard divided into an
    # MM-correction-inflated numerator -- which would spuriously dominate the redundancy ranking. Clamp it.
    return csu if csu <= 1.0 else 1.0


@njit(cache=True)
def _cmi_miller_madow_bias(factors_data, x, y, z, factors_nbins, dtype) -> float:
    """Analytic Miller-Madow bias of the plug-in CMI: ``(k_xyz + k_z - k_xz - k_yz) / (2n)`` where ``k_*`` are the
    OCCUPIED-cell counts of the four joint distributions. This is exactly the bias term ``_cmi_from_binned`` /
    ``_fe_cmi_redundancy_null`` use; subtracting it makes the Fleuret redundancy carry the SAME bias correction as the
    Miller-Madow relevance (critique N-F2 part 1). Fully isolated: only called when ``use_mm`` is set, so the default
    plug-in redundancy path is untouched."""
    n = factors_data.shape[0]
    if n <= 0:
        return 0.0
    _, fz, _ = merge_vars(factors_data, z, None, factors_nbins, dtype=dtype)
    _, fxz, _ = merge_vars(factors_data, unpack_and_sort(x, z), None, factors_nbins, dtype=dtype)
    _, fyz, _ = merge_vars(factors_data, unpack_and_sort(y, z), None, factors_nbins, dtype=dtype)
    _, fxyz, _ = merge_vars(factors_data, unpack_and_sort(unpack_and_sort(x, y), z), None, factors_nbins, dtype=dtype)
    k_z = np.count_nonzero(fz)
    k_xz = np.count_nonzero(fxz)
    k_yz = np.count_nonzero(fyz)
    k_xyz = np.count_nonzero(fxyz)
    return float((k_xyz + k_z - k_xz - k_yz) / (2.0 * n))


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
    entropy_cache: Optional[dict] = None,
    can_use_x_cache: bool = False,
    can_use_y_cache: bool = False,
    dtype=np.int32,
    use_mm: bool = False,
) -> float:
    """Conditional mutual information ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)``.

    ``use_mm`` (critique N-F2 part 1): subtract the analytic Miller-Madow CMI bias so the Fleuret redundancy carries
    the SAME bias correction as the Miller-Madow relevance. No-op (bit-identical) when False, the default -- the MM
    path recomputes the four occupied-cell counts (the entropy_cache stores only the entropy scalar, not the counts).

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
            # Fused freqs-only melt: the H(X,Z) path discards merge_vars' final_classes relabel array; compute
            # entropy straight from the pruned joint freqs (bit-identical, ~5-6x on the common 2-var X u Z union).
            entropy_xz = _entropy_xz_fused(factors_data, indices, factors_nbins, dtype)
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

            if len(x) == 1:
                # Fused single-candidate melt onto the precomputed (Y,Z) classes: histogram + entropy in one
                # pass, no final_classes mutation/remap (all discarded by this path). Bit-identical, ~2.5-4.5x.
                entropy_xyz = _entropy_x_onto_classes(factors_data, x[0], classes_yz, current_nclasses_yz, factors_nbins[x[0]])
            else:
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
    if use_mm:
        res = res - _cmi_miller_madow_bias(factors_data, x, y, z, factors_nbins, dtype)
        if res < 0.0:
            res = 0.0
    return res


def conditional_mi_redundancy_batched(
    factors_data,
    cand_indices,
    y_index,
    z_index,
    factors_nbins,
    dtype=np.int32,
    force=None,
):
    """Batched ``I(X_j; Y | Z)`` over all candidate columns for the greedy Fleuret redundancy round.

    DISPATCH POINT for the MRMR redundancy loop's dominant cost: instead of calling the serial njit
    ``conditional_mi`` once per candidate (``Z`` = the just-selected feature, ``Y`` = target), this
    routes the whole candidate sweep through the batched CUDA kernel when ``(n, p)`` is large enough
    to win (size/HW gate via ``kernel_tuning_cache``), and to the exact CPU ``conditional_mi`` loop
    otherwise / when no GPU. Bit-parity (<1e-9) with the per-candidate CPU path -- see
    ``tests/feature_selection/test_cmi_cuda_kernel.py``.

    NOTE on the integration site: the actual redundancy loop (``filters/evaluation.py``) runs inside
    an ``@njit`` function with per-Z early-exit + entropy caching, so it cannot itself call cupy.
    This module-level wrapper is the single Python-reachable dispatch hook; a Python-level caller
    that materialises the candidate set per greedy round can swap its per-candidate ``conditional_mi``
    loop for this one call. Lazy import keeps cupy off the njit import path.

    Returns a ``(p,)`` float64 array of raw CMI (clamped at 0), aligned with ``cand_indices``.
    """
    from ._cmi_cuda import conditional_mi_batched_dispatch

    return conditional_mi_batched_dispatch(
        factors_data=factors_data,
        cand_indices=cand_indices,
        y_index=y_index,
        z_index=z_index,
        factors_nbins=factors_nbins,
        dtype=dtype,
        force=force,
    )
