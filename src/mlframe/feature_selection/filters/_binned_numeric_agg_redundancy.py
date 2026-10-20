"""Redundancy-gate builder of the binned numeric aggregation family: bins each candidate (on the device when resident) and keeps those whose CMI about y given their sources clears the permutation-null ceiling (carved from _binned_numeric_agg_fe; re-exported there)."""

from __future__ import annotations

import logging
import math

from mlframe.utils.log_throttle import log_throttle
import numpy as np

from ._binned_agg_linear_usability import linear_usability_keeps

logger = logging.getLogger(__name__)


def _candidate_codes(col: np.ndarray, nbins: int, quantile_bin):
    """Equi-frequency codes of one binagg candidate column. Under the strict-resident path an all-finite column is binned on the device and its codes STAY
    there (the CMI scorers below read resident codes in place, so a candidate no longer costs a copy back per column); otherwise the host binner runs."""
    try:
        from mlframe.feature_selection.filters._gpu_strict_fe import fe_gpu_strict_resident_enabled

        if fe_gpu_strict_resident_enabled() and np.isfinite(col).all():
            from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin_gpu_resident

            dev = _quantile_bin_gpu_resident(col, nbins)
            if dev is not None:
                return dev
    except Exception as e:
        logger.debug("resident binagg candidate binning failed, using the host binner: %s", e)
    return quantile_bin(col, nbins=nbins)


def _null_tail(null: np.ndarray, z: float) -> float:
    """Upper tail point of a positive, right-skewed null at the one-sided level of the normal z: the quantile of a Gamma fitted to the null by moments, never below the null mean.

    ``null`` is the vector of permutation-null CMI values; ``z`` the family-wise z. With a degenerate null (zero mean or variance) the normal bound ``mean + z * std`` is returned.
    """
    mean, var = float(null.mean()), float(null.var())
    normal_bound = mean + float(z) * math.sqrt(var)
    if not (mean > 0.0 and var > 0.0):
        return normal_bound
    from scipy.stats import gamma, norm

    shape, scale = mean * mean / var, var / mean
    return max(float(gamma.isf(float(norm.sf(float(z))), a=shape, scale=scale)), mean)


def _strata_permuted_labels(y_cls, strata, n_perm: int, seed: int) -> np.ndarray:
    """An (n, n_perm) matrix whose columns are ``y_cls`` permuted WITHIN the strata of ``strata`` (all rows in one stratum when it is None).

    The null of a conditional MI, ``CMI(cand; y | z)``, must keep the dependence of ``y`` on ``z``: shuffling ``y`` over all rows removes it, the
    plug-in estimate under that null sits far lower than the estimate for a candidate that is merely a function of ``z``'s partition (the null
    mean was 0.0008 against 0.0032 within strata on a noise aggregate of an informative group), and every such candidate then looks significant.
    A random permutation followed by a stable sort on the stratum code gives, per stratum, its rows in random order; assigning the labels of those
    rows to the strata's rows in natural order is a uniform permutation within each stratum, in linear time for small integer codes.
    """
    y = np.asarray(y_cls)
    n = y.shape[0]
    if strata is None:
        code = np.zeros(n, dtype=np.uint8)
    else:
        code = np.asarray(strata.get() if hasattr(strata, "get") else strata)
        code = code.astype(np.uint16 if code.size and int(code.max()) < 65535 else np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    natural = np.argsort(code, kind="stable")
    out = np.empty((n, int(n_perm)), dtype=y.dtype)
    for i in range(int(n_perm)):
        shuffled = rng.permutation(n)
        out[natural, i] = y[shuffled[np.argsort(code[shuffled], kind="stable")]]
    return out


def _labels_for_pair(cache: dict, key: tuple, y_cls, strata, n_perm: int, seed: int) -> np.ndarray:
    """The within-strata permuted labels of one (group, agg) pair, built from the seed of its first candidate and reused by the pair's other statistics (``cache`` holds the last pair only)."""
    if cache.get("key") != key:
        cache["key"] = key
        cache["matrix"] = _strata_permuted_labels(y_cls, strata, n_perm, seed)
    return cache["matrix"]


def _build_binned_agg_recipes(cands, raw, X, _src_bins, nbins_base, y_cls, min_cmi_gain, _rng, _n_perm, _NULL_Z, kept_cols, reject_sink, linear_usability=True):
    """Build the replayable recipe of every binned-aggregate column; with ``linear_usability`` a column the CMI gate rejects is still kept when it lifts a linear fit."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _cmi_gpu_enabled, _renumber_joint

    rejected: dict = {}
    labels_cache: dict = {}
    for nm in cands.names:
        srcs = [c for c in (raw[nm].get("group_col"), raw[nm].get("agg_col")) if c in X.columns]
        z_joint = _renumber_joint(*[_src_bins(c) for c in srcs])[0] if srcs else None
        # bench-attempt-rejected (2026-07-02): routing this binning through batched_quantile_bin_gpu (with a
        # _renumber_joint_gpu joint) FAILED the redundancy suite - its partition differs from _quantile_bin
        # (a genuinely redundant binagg column survived). Unnecessary anyway: _quantile_bin itself already
        # routes large columns to its device twin under the STRICT-resident path (size-gated
        # _quantile_bin_gpu), so this gate's binning is device-backed without a partition change.
        cand_bin = cands.codes(nm, nbins_base)
        cmi = _cmi_from_binned(cand_bin, y_cls, z_joint)
        null_ceiling = 0.0
        if np.isfinite(cmi) and cmi >= float(min_cmi_gain):
            # BATCHED null (launch-reduction): cand_bin / z_joint are FIXED across the _n_perm shuffles;
            # only the permuted y varies. Plug-in CMI is symmetric in X and Y, so CMI(cand; yp | z) ==
            # CMI(yp; cand | z) - stack the SAME _rng-drawn permuted-y columns into one (n, nperm) matrix
            # and score them all in ONE batched_cmi_gpu workload (cand as the fixed 'y', z as support),
            # replacing _n_perm per-perm _cmi_from_binned calls. Identical permutations -> the null ceiling
            # is selection-equivalent; falls back to the per-perm loop on any error / when GPU is off.
            # ONE draw from `_rng` for the permutation seed, then a child generator both paths rebuild
            # identically. The GPU path used to consume `_n_perm` draws BEFORE the call that can fail, and
            # the host fallback then drew `_n_perm` MORE from the now-advanced generator -- so a GPU failure
            # silently produced a DIFFERENT null ceiling and therefore a different keep/reject verdict, which
            # is exactly the selection-equivalence the comment above claims. Seeding a child also leaves
            # `_rng` equally advanced whichever path runs, so everything downstream is unaffected too, and it
            # costs no extra memory (the fallback still permutes one column at a time).
            _perm_seed = int(_rng.integers(0, 2**63 - 1))
            _pair_key = (raw[nm].get("group_col"), raw[nm].get("agg_col"))
            _null = None
            try:
                if _cmi_gpu_enabled(n=int(y_cls.shape[0]), p=int(_n_perm), min_p=2) and int(_n_perm) > 1:
                    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu

                    _null = np.asarray(batched_cmi_gpu(_labels_for_pair(labels_cache, _pair_key, y_cls, z_joint, int(_n_perm), _perm_seed), cand_bin, z_joint), dtype=np.float64)  # Intentional float64: permutation-null MI statistic
                    _null = np.where(np.isfinite(_null), _null, 0.0)
            except Exception as e:
                log_throttle(
                    logger,
                    "binagg_gpu_perm_null_fallback",
                    logging.WARNING,
                    "GPU permutation-null batch failed (%s: %s); recomputing the null on the host path. The "
                    "permutations are seeded identically, so the verdict is unchanged -- the GPU cost is not.",
                    type(e).__name__,
                    e,
                )
                _null = None
            if _null is None:
                _null = np.empty(_n_perm, dtype=np.float64)  # Intentional float64: permutation-null MI statistic
                _labels = _labels_for_pair(labels_cache, _pair_key, y_cls, z_joint, int(_n_perm), _perm_seed)
                for _i in range(_n_perm):
                    c0 = _cmi_from_binned(cand_bin, _labels[:, _i], z_joint)
                    _null[_i] = c0 if np.isfinite(c0) else 0.0
            # One-sided upper tail of the null at the family-wise level. The plug-in CMI under the null is positive and skewed to the right (close to a
            # scaled chi-square), so a normal bound ``mean + z * std`` sits below its true tail: a Gamma fitted to the null's mean and variance is used,
            # and the normal bound stays as the floor when the null is degenerate.
            null_ceiling = _null_tail(_null, _NULL_Z)
        if np.isfinite(cmi) and cmi >= float(min_cmi_gain) and cmi > null_ceiling:
            kept_cols.append(nm)
        else:
            rejected[nm] = (srcs, cmi, null_ceiling)
    if linear_usability and rejected:
        _keep_linearly_usable(cands, raw, X, _src_bins, nbins_base, y_cls, rejected, kept_cols)
        order = {name: i for i, name in enumerate(cands.names)}
        kept_cols.sort(key=order.__getitem__)
    if reject_sink is None:
        return
    kept_set = set(kept_cols)
    for nm, (srcs, cmi, null_ceiling) in rejected.items():
        if nm in kept_set:
            continue
        try:
            reject_sink(
                gate="binagg_source_redundancy", candidate=str(nm),
                operands=tuple(srcs), operator="binned_numeric_agg_redundancy_gate",
                observed=float(cmi) if np.isfinite(cmi) else 0.0,
                threshold=max(float(min_cmi_gain), float(null_ceiling)),
                reason="binagg CMI about y given its sources does not clear the permutation-null ceiling",
            )
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass


def _keep_linearly_usable(cands, raw, X, _src_bins, nbins_base, y_cls, rejected, kept_cols) -> None:
    """Add to ``kept_cols`` the columns the CMI gate rejected that lift an out-of-fold linear fit over their raw sources.

    A binned-aggregate column is a function of its group's cell, which is the partition the CMI test conditions on, so it reads as redundant to
    any model that can use the raw group. A linear model cannot. The columns of one (group, agg) pair are tested together and the family-wise
    level 0.05 is split over the pairs (see ``_binned_agg_linear_usability``).
    """
    by_pair: dict = {}
    for nm in rejected:
        group_col, agg_col = raw[nm].get("group_col"), raw[nm].get("agg_col")
        if rejected[nm][0] and group_col in X.columns:
            by_pair.setdefault((group_col, agg_col), []).append(nm)
    level = 0.05 / max(1, len({(raw[n].get("group_col"), raw[n].get("agg_col")) for n in cands.names}))
    for (group_col, agg_col), names in by_pair.items():
        srcs = rejected[names[0]][0]
        codes = {nm: cands.codes(nm, nbins_base) for nm in names}
        kept_cols.extend(sorted(linear_usability_keeps(X, srcs, codes, y_cls, _src_bins(group_col), level, f"{group_col}|{agg_col}"), key=names.index))
