"""Analytic large-n null fast path of ``mi_direct`` and its fused single-variable relevance kernel.

Re-exported from ``permutation`` so existing ``from ...permutation import X`` call sites keep working.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from numba import njit

from .info_theory import (
    merge_vars, compute_relevance_score, use_su_normalization, use_mi_miller_madow, use_mi_chao_shen,
)

logger = logging.getLogger(__name__)


@njit(nogil=True, cache=True)
def _relevance_mi_1var_fused(
    factors_data: np.ndarray,
    ix: int,
    nb_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> tuple:
    """Fused single-variable raw MI I(X_ix; Y) + occupied-x-bin count in ONE O(n) pass.

    Returns ``(mi, bx)`` where ``mi`` is BIT-IDENTICAL to
    ``compute_mi_from_classes(*merge_vars(factors_data, [ix], ...)[:2], classes_y, freqs_y)``
    and ``bx`` equals the pruned ``len(freqs_x)`` (occupied x-bins) that
    ``merge_vars`` returns - exactly the two quantities the ANALYTIC-null branch of
    ``mi_direct`` consumes (``_ax_freqs.shape[0]`` and the MI). The legacy path built the
    length-n ``classes_x`` relabel array via ``merge_vars`` (one O(n) accumulate pass plus a
    second O(n) lookup-remap pass when any x-bin is empty) and then walked all n samples AGAIN
    inside ``compute_mi_from_classes`` to build the joint histogram - and the analytic branch
    DISCARDS ``classes_x`` entirely (it needs only ``len(freqs_x)`` and the MI scalar). This
    kernel builds the ``(nb_x, K_y)`` joint histogram directly from ``factors_data[:, ix]`` and
    ``classes_y`` in a SINGLE row pass, then derives ``freqs_x`` from the row sums - eliminating
    the separate ``merge_vars(x)`` pass + its ``final_classes``/lookup allocations. Mirrors the
    ``joint_freqs_2var`` / ``joint_entropy_2var`` pruned-fast-path wins on the DCD pairwise path.

    BIT-IDENTITY to ``compute_mi_from_classes(merge_vars(...))`` (raw MI, no SU/MM):
      * ``merge_vars`` renumbers OCCUPIED x-bins densely in ASCENDING original-bin order, so
        ``compute_mi_from_classes`` visits x-classes ``i in range(len(freqs_x))`` in that same
        ascending order. Iterating ``i in range(nb_x)`` here and SKIPPING empty rows
        (``rowcount == 0``) visits the identical occupied bins in the identical order.
      * ``prob_x = rowcount / n`` reproduces ``merge_vars``'s ``freqs / n_rows`` (int64 count,
        float64 true division); ``prob_y = freqs_y[j]`` is passed through unchanged; and
        ``jf = jc * inv_n`` with ``inv_n = 1.0 / n`` matches ``compute_mi_from_classes`` exactly
        - same operands, same order, so the ``jf*log(jf/(prob_x*prob_y))`` terms accumulate
        bit-for-bit.
      * Empty x-bins (and empty joint cells) contribute nothing to MI in EITHER path, so pruning
        vs skipping is numerically inert.
    """
    n = factors_data.shape[0]
    K_y = len(freqs_y)
    if n == 0:
        return 0.0, 0
    joint = np.zeros((nb_x, K_y), dtype=np.int64)
    for k in range(n):
        joint[factors_data[k, ix], classes_y[k]] += 1
    inv_n = 1.0 / n
    total = 0.0
    bx = 0
    for i in range(nb_x):
        rowcount = 0
        for j in range(K_y):
            rowcount += joint[i, j]
        if rowcount == 0:
            continue
        bx += 1
        prob_x = rowcount / n
        for j in range(K_y):
            jc = joint[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                total += jf * math.log(jf / (prob_x * prob_y))
    return total, bx


def _mi_direct_analytic_null(
    factors_data: np.ndarray,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    dtype: type,
    classes_y: Optional[np.ndarray],
    freqs_y: Optional[np.ndarray],
    return_null_mean: bool,
) -> tuple[Optional[tuple], Optional[np.ndarray], Optional[np.ndarray]]:
    """Analytic large-n null for ``mi_direct``: returns ``(result_or_None, classes_y, freqs_y)``.

    ``result`` is the finished 2- or 4-tuple when the analytic path applies, else ``None`` (caller falls through to the permutation path). ``classes_y``/``freqs_y`` are
    returned because this step may have merged ``y`` and the caller reuses them."""
    try:
        from ._analytic_mi_null import (
            analytic_mi_null, analytic_null_enabled, analytic_null_min_n, analytic_null_applicable,
        )

        # The analytic-null fast path returned RAW plug-in
        # MI unconditionally (compute_relevance_score(False, ...), no use_mm/use_cs) while the sub-threshold
        # permutation path below always applies the active mi_correction - a silent formula discontinuity
        # in the reported relevance value exactly at the analytic_null_min_n() threshold whenever
        # mi_correction='miller_madow'/'chao_shen' is active. Fall through to the exact permutation path
        # (unchanged) when either correction is active, matching the existing SU/sparsity fall-through pattern.
        _analytic_ok = (
            analytic_null_enabled()
            and int(factors_data.shape[0]) >= analytic_null_min_n()
            and not use_su_normalization()
            and not use_mi_miller_madow()
            and not use_mi_chao_shen()
        )
    except Exception as e:
        logger.warning("analytic-null availability check failed, defaulting to False: %s", e)
        _analytic_ok = False
    if _analytic_ok:
        if classes_y is None:
            classes_y, freqs_y, _ = merge_vars(
                factors_data=factors_data, vars_indices=y,
                var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
            )
        assert freqs_y is not None  # the None branch above always sets classes_y/freqs_y together
        _n_rows = int(factors_data.shape[0])
        _by = int(freqs_y.shape[0])
        # FUSED single-var fast path (wasted-per-call-work audit): the analytic branch
        # needs only the MI scalar + the occupied-x-bin count - it DISCARDS the length-n classes_x
        # that merge_vars(x) builds. For the single-variable relevance x (every MRMR/FE caller passes
        # x=(var,) / x=[0]), ``_relevance_mi_1var_fused`` builds the joint histogram + MI + bx in ONE
        # O(n) pass, skipping the separate merge_vars(x) accumulate (+ remap) pass entirely.
        # Bit-identical to the legacy merge_vars + compute_relevance_score(False, ...) below (proven
        # in test_mi_direct_fused_1var_relevance). Mirrors the joint_freqs_2var DCD win.
        if len(x) == 1:
            _ix = int(x[0])
            _nb_x = int(factors_nbins[_ix])
            _ax_mi, _ax_bx = _relevance_mi_1var_fused(factors_data, _ix, _nb_x, classes_y, freqs_y)
            if analytic_null_applicable(_n_rows, _ax_bx, _by):
                _ax_nm, _ax_p = analytic_mi_null(_ax_mi, _n_rows, _ax_bx, _by)
                if return_null_mean:
                    return (_ax_mi, 1.0 - _ax_p, _ax_nm, _ax_p), classes_y, freqs_y
                return (_ax_mi, 1.0 - _ax_p), classes_y, freqs_y
            # Sparse cells -> analytic null unreliable; FALL THROUGH to the permutation path below
            # (which rebuilds merge_vars(x); rare high-cardinality case, not the dense hot path).
        else:
            _ax_classes, _ax_freqs, _ = merge_vars(
                factors_data=factors_data, vars_indices=x,
                var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
            )
            # Occupancy safe-condition: the chi-square / Miller-Madow approximation is only trustworthy
            # when the contingency cells are not sparse (a high-cardinality x can have sparse cells even at
            # large n). When sparse, FALL THROUGH to the sparsity-correct permutation path below.
            if analytic_null_applicable(int(_ax_classes.shape[0]), int(_ax_freqs.shape[0]), _by):
                _ax_mi = compute_relevance_score(False, _ax_classes, _ax_freqs, classes_y, freqs_y, dtype=dtype)
                _ax_nm, _ax_p = analytic_mi_null(
                    _ax_mi, int(_ax_classes.shape[0]), int(_ax_freqs.shape[0]), _by,
                )
                if return_null_mean:
                    return (_ax_mi, 1.0 - _ax_p, _ax_nm, _ax_p), classes_y, freqs_y
                return (_ax_mi, 1.0 - _ax_p), classes_y, freqs_y
    return None, classes_y, freqs_y
