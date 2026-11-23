"""Rank-aware tail-concentration verdicts for the usability admission gate, batched over pairs.

Moved out of ``batch_pair_usability_corr_gpu`` (which had reached the 1000-line budget); it reuses that module's
dispatcher and form ids, so the values are the ones the batched correlation path produces.
"""
from __future__ import annotations

import numpy as np

from .batch_pair_usability_corr_gpu import (
    ALL_PAIR_FORM_IDS,
    ALL_SINGLE_FORM_IDS,
    FORM_X0SQ_DIV_X1,
    FORM_X0_DIV_X1,
    FORM_X1_DIV_X0,
    FORM_X1SQ_DIV_X0,
    _EPS_DENOM_FLOOR,
    dispatch_batch_pair_usability_corr,
)

def _eval_pair_form_numpy(form_id: int, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Vectorised (numpy, not numba) evaluation of ONE of the 5 ``ALL_PAIR_FORM_IDS`` forms across a whole
    array - mirrors ``_eval_form``'s per-row logic and ``_fe_usability_signal.usability_form_corrs``'s own
    pair-form construction (the eps-floored ratio denominator -> NaN on a near-zero divisor) EXACTLY, so a
    caller reconstructing the winning form's VALUES (needed for the rank-correlation leg - a |corr| alone
    is not enough) gets bit-identical values to what the serial reference would have used. Only called on
    the small subset of pairs that clear :func:`batch_pair_tail_concentration_rankaware`'s cheap min_corr/
    pairness gate - never on the full candidate pool (the whole point of the two-stage split)."""
    _eps = _EPS_DENOM_FLOOR
    if form_id == FORM_X0_DIV_X1:
        x1f = np.where(np.abs(x1) < _eps, np.nan, x1)
        return np.asarray(x0 / x1f)
    if form_id == FORM_X1_DIV_X0:
        x0f = np.where(np.abs(x0) < _eps, np.nan, x0)
        return np.asarray(x1 / x0f)
    if form_id == FORM_X0SQ_DIV_X1:
        x1f = np.where(np.abs(x1) < _eps, np.nan, x1)
        return np.asarray((x0 * x0) / x1f)
    if form_id == FORM_X1SQ_DIV_X0:
        x0f = np.where(np.abs(x0) < _eps, np.nan, x0)
        return np.asarray((x1 * x1) / x0f)
    # FORM_X0_MUL_X1
    return np.asarray(x0 * x1)


def batch_pair_tail_concentration_rankaware(
    y: np.ndarray,
    operand_matrix: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    *,
    min_corr: float,
    pairness_margin: float,
    max_rank_frac: float = 0.7,
    single_corr: np.ndarray | None = None,
) -> np.ndarray:
    """Batched equivalent of calling ``_fe_usability_signal.pair_is_tail_concentrated_rankaware`` once per
    pair in ``(pair_a, pair_b)`` - returns a bool array, one verdict per pair. ``y``/``operand_matrix`` must
    already be subsampled + cast to the SAME dtype the serial reference uses internally (``_crit_np_dtype()``)
    - this function does not re-subsample; the caller applies ``_corr_stride``/``_crit_np_dtype()`` ONCE for
    the whole pool (mirrors the existing dominant-pair prescan's pattern) instead of once per pair.

    ``single_corr``: optional array of each OPERAND-MATRIX ROW's own single-form |corr| (from
    ``_fe_usability_signal._single_operand_usability_corr``, aligned to ``operand_matrix`` rows via the
    caller's own row-index mapping) - when given, pair i's ``_cs`` is ``max(single_corr[pair_a[i]],
    single_corr[pair_b[i]])`` (bit-identical to the serial per-pair ``precomputed_single_corr`` path); when
    omitted, computed via this same batched kernel over the 4 single forms.

    TWO-STAGE, avoiding materializing all 5 pair-forms for every pair (the whole point of batching - most
    candidate pairs reaching this gate are noise and fail the cheap gate immediately): stage 1 batches ONLY
    the |corr| reduction (streamed on-core, no full form array ever stored) for every pair to get the best
    pair-form's value AND which form won; stage 2 rebuilds the actual winning-form VALUES (a plain numpy
    elementwise op, via :func:`_eval_pair_form_numpy`) only for the SUBSET that clears stage 1's min_corr/
    pairness gate, to run the rank-correlation leg - mirrors the serial reference's own control flow
    (``_cp >= min_corr and _cp >= margin*_cs`` gates BEFORE the rank transform is ever computed)."""
    from ._fe_usability_signal import _rank_transform, abs_pearson

    n_pairs = int(pair_a.shape[0])
    verdict = np.zeros(n_pairs, dtype=bool)
    if n_pairs == 0:
        return verdict

    pair_corrs, _ = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, form_ids=ALL_PAIR_FORM_IDS)
    best_form_i = np.argmax(pair_corrs, axis=1)
    cp = pair_corrs[np.arange(n_pairs), best_form_i]

    if single_corr is not None:
        cs = np.maximum(single_corr[pair_a], single_corr[pair_b])
    else:
        single_corrs, _ = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, form_ids=ALL_SINGLE_FORM_IDS)
        cs = single_corrs.max(axis=1)

    _gate = (cp >= float(min_corr)) & (cp >= float(pairness_margin) * cs)
    _cand_idx = np.flatnonzero(_gate)
    if _cand_idx.size == 0:
        return verdict

    y_arr = np.asarray(y)
    for _i in _cand_idx:
        _form_id = int(ALL_PAIR_FORM_IDS[best_form_i[_i]])
        _x0 = operand_matrix[pair_a[_i]]
        _x1 = operand_matrix[pair_b[_i]]
        _form_vals = np.asarray(_eval_pair_form_numpy(_form_id, _x0, _x1))
        _m = np.isfinite(_form_vals) & np.isfinite(y_arr)
        if int(_m.sum()) < 3:
            continue
        _rank_corr = abs_pearson(_rank_transform(y_arr[_m]), _rank_transform(_form_vals[_m]))
        verdict[_i] = bool(_rank_corr <= float(max_rank_frac) * cp[_i])
    return verdict
