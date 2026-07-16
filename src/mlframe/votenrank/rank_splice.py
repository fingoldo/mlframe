"""``segment_rank_splice``: re-order only a segment's rows using an alternate model, all else fixed.

Source: 2nd_amex-default-prediction.md -- a specialist "sparse-history" model's predictions are merged into
a main ranked list by RE-SORTING ONLY WITHIN the sparse subgroup, preserving overall rank order for the
majority. Extracted from ``training.composite.segment_routed.SegmentRoutedEstimator``'s private
``_rank_splice`` helper (the same algorithm, embedded in a full sklearn-style fit/predict estimator there) as
a standalone, general-purpose votenrank utility callable directly on raw score arrays -- useful whenever the
"specialist model, splice into a global ranking" pattern is needed without wanting the full estimator's
fit/predict/feature-subsetting machinery.

Why re-ranking, not blending: a ranking/AUC-style metric only cares about relative order. The specialist and
main model can have arbitrarily different score calibration/scale -- averaging raw scores for the segment
would inject an uncontrolled distortion into the full population's ranking. Rank-splicing instead keeps every
segment row's exact numeric score VALUE from the multiset the main model already assigned to that segment (so
the segment's aggregate position in the global distribution is untouched), and only permutes WHICH row gets
which value, using the specialist's own within-segment ranking to decide the order.
"""
from __future__ import annotations

import numpy as np


def segment_rank_splice(
    main_scores: np.ndarray,
    specialist_scores: np.ndarray,
    segment_mask: np.ndarray,
    *,
    blend_weight: float = 0.0,
) -> np.ndarray:
    """Re-order rows within ``segment_mask`` by ``specialist_scores``' rank; every other row is untouched.

    Parameters
    ----------
    main_scores
        ``(n,)`` the full population's main-model scores.
    specialist_scores
        ``(n,)`` OR ``(n_segment,)`` -- the specialist model's scores. If ``(n,)``, only the ``segment_mask``
        positions are read; if ``(n_segment,)``, it's treated as already segment-subset (in the same row
        order as ``main_scores[segment_mask]``).
    segment_mask
        ``(n,)`` boolean mask selecting the rows to re-order.
    blend_weight
        ``0.0`` (default) is a hard cutover -- the segment order is decided purely by the specialist's
        within-segment rank, matching the original behavior bit-for-bit. ``1.0`` would ignore the specialist
        entirely and keep the main model's own within-segment order. Values in between linearly blend the two
        rank positions before re-sorting, which matters when the specialist's ranking is genuinely informative
        but noisy: averaging two independently-noisy rank estimates (main and specialist) reduces the combined
        rank variance versus either alone, the same effect an ensemble gets from averaging predictions. Must
        be in ``[0.0, 1.0]``.

    Returns
    -------
    np.ndarray
        ``(n,)`` -- a copy of ``main_scores`` with the segment's row values PERMUTED (the multiset of values
        within the segment is unchanged; only their row assignment changes, ordered by the blended rank).
        Non-segment rows are identical to ``main_scores``.
    """
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError(f"segment_rank_splice: blend_weight must be in [0.0, 1.0], got {blend_weight}")

    main_arr = np.asarray(main_scores, dtype=np.float64)
    mask = np.asarray(segment_mask, dtype=bool)
    if mask.shape != main_arr.shape:
        raise ValueError(f"segment_rank_splice: segment_mask shape {mask.shape} must match main_scores shape {main_arr.shape}")

    specialist_arr = np.asarray(specialist_scores, dtype=np.float64)
    specialist_segment = specialist_arr[mask] if specialist_arr.shape == main_arr.shape else specialist_arr
    if specialist_segment.shape[0] != int(mask.sum()):
        raise ValueError(f"segment_rank_splice: specialist_scores segment length {specialist_segment.shape[0]} must match segment_mask.sum() {int(mask.sum())}")

    def _ranks(v: np.ndarray) -> np.ndarray:
        """0-based ordinal ranks via single argsort + scatter (bit-identical to argsort(argsort(v)),
        ~1.7-1.9x faster -- the second sort was pure waste, the inverse permutation of the first argsort
        IS the rank vector)."""
        order = np.argsort(v)
        r = np.empty(v.size, dtype=order.dtype)
        r[order] = np.arange(v.size)
        return r

    out = main_arr.copy()
    main_segment = main_arr[mask]
    sorted_main_vals = np.sort(main_segment)
    specialist_rank = _ranks(specialist_segment)  # 0-based rank of each row within the segment
    if blend_weight == 0.0:
        # Hard-cutover default: identical to the pre-blend implementation, byte-for-byte.
        final_rank = specialist_rank
    else:
        main_rank = _ranks(main_segment)
        blended_key = (1.0 - blend_weight) * specialist_rank.astype(np.float64) + blend_weight * main_rank.astype(np.float64)
        final_rank = _ranks(blended_key)
    out[mask] = sorted_main_vals[final_rank]
    return np.asarray(out)


__all__ = ["segment_rank_splice"]
