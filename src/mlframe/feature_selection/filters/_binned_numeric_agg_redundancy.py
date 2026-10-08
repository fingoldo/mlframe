"""Redundancy-gate builder of the binned numeric aggregation family: bins each candidate (on the device when resident) and keeps those whose CMI about y given their sources clears the permutation-null ceiling (carved from _binned_numeric_agg_fe; re-exported there)."""

from __future__ import annotations

import logging
from mlframe.utils.log_throttle import log_throttle
import numpy as np

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


def _build_binned_agg_recipes(cands, raw, X, _src_bins, nbins_base, y_cls, min_cmi_gain, _rng, _n_perm, _NULL_Z, kept_cols, reject_sink):
    """Build the replayable recipe of every binned-aggregate column."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _cmi_gpu_enabled, _renumber_joint

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
            _null = None
            try:
                if _cmi_gpu_enabled(n=int(y_cls.shape[0]), p=int(_n_perm), min_p=2) and int(_n_perm) > 1:
                    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu

                    _prng = np.random.default_rng(_perm_seed)
                    _Yp = np.empty((y_cls.shape[0], int(_n_perm)), dtype=np.int64)
                    for _i in range(int(_n_perm)):
                        _Yp[:, _i] = y_cls[_prng.permutation(y_cls.shape[0])]
                    _null = np.asarray(batched_cmi_gpu(_Yp, cand_bin, z_joint), dtype=np.float64)
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
                _prng = np.random.default_rng(_perm_seed)
                _null = np.empty(_n_perm, dtype=np.float64)
                for _i in range(_n_perm):
                    yp = y_cls[_prng.permutation(y_cls.shape[0])]
                    c0 = _cmi_from_binned(cand_bin, yp, z_joint)
                    _null[_i] = c0 if np.isfinite(c0) else 0.0
            # Robust one-sided null upper tail (mean + z*std), not the noisy raw max.
            null_ceiling = float(_null.mean() + _NULL_Z * _null.std())
        keep = np.isfinite(cmi) and cmi >= float(min_cmi_gain) and cmi > null_ceiling
        if keep:
            kept_cols.append(nm)
        elif reject_sink is not None:
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
