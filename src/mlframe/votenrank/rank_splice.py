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


def segment_rank_splice(main_scores: np.ndarray, specialist_scores: np.ndarray, segment_mask: np.ndarray) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        ``(n,)`` -- a copy of ``main_scores`` with the segment's row values PERMUTED (the multiset of values
        within the segment is unchanged; only their row assignment changes, ordered by the specialist's
        within-segment rank). Non-segment rows are identical to ``main_scores``.
    """
    main_arr = np.asarray(main_scores, dtype=np.float64)
    mask = np.asarray(segment_mask, dtype=bool)
    if mask.shape != main_arr.shape:
        raise ValueError(f"segment_rank_splice: segment_mask shape {mask.shape} must match main_scores shape {main_arr.shape}")

    specialist_arr = np.asarray(specialist_scores, dtype=np.float64)
    specialist_segment = specialist_arr[mask] if specialist_arr.shape == main_arr.shape else specialist_arr
    if specialist_segment.shape[0] != int(mask.sum()):
        raise ValueError(f"segment_rank_splice: specialist_scores segment length {specialist_segment.shape[0]} must match segment_mask.sum() {int(mask.sum())}")

    out = main_arr.copy()
    main_segment = main_arr[mask]
    sorted_main_vals = np.sort(main_segment)
    specialist_rank = np.argsort(np.argsort(specialist_segment))  # 0-based rank of each row within the segment
    out[mask] = sorted_main_vals[specialist_rank]
    return np.asarray(out)


__all__ = ["segment_rank_splice"]
