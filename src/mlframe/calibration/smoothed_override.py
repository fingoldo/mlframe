"""``apply_smoothed_override``: convex-blend a high-confidence known label with the model's prediction.

Source: 4th_mechanisms-of-action-moa-prediction.md, comment (Theo Viel) -- "Not using a hard label but a
smoothed version worked though ... ``a * hard_target + (1 - a) * original prediction``, with a = 0.9 or
0.99." When a rule/lookup is highly confident about an entity's true label (a known user-ID match, a known
drug/compound match, a deterministic business rule), the tempting move is to REPLACE the model's prediction
outright -- but a hard override risks a private-LB-style drop whenever the "known" label is occasionally
wrong (mislabeled ground truth, a stale lookup, an edge case the rule didn't anticipate). A convex blend
keeps the override's benefit on the (common) case where it's right while bounding the damage on the (rare)
case where it's wrong, since the model's own prediction never gets fully zeroed out.
"""
from __future__ import annotations

import numpy as np


def apply_smoothed_override(prediction: np.ndarray, known_label: np.ndarray, override_mask: np.ndarray, a: float = 0.9) -> np.ndarray:
    """Blend ``known_label`` into ``prediction`` wherever ``override_mask`` is True, at strength ``a``.

    Parameters
    ----------
    prediction
        ``(n,)`` the model's own predictions.
    known_label
        ``(n,)`` the high-confidence known/rule-derived label (only read where ``override_mask`` is True).
    override_mask
        ``(n,)`` boolean -- rows where the confident-override rule fired.
    a
        Blend strength in ``[0, 1]``: ``a=1.0`` is a hard override (NOT recommended, per the source's own
        finding), ``a=0.0`` is a no-op. ``0.9``-``0.99`` matches the source's own tuned range.

    Returns
    -------
    np.ndarray
        ``prediction`` with overridden rows replaced by ``a * known_label + (1 - a) * prediction``; rows
        outside ``override_mask`` are returned unchanged.
    """
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"apply_smoothed_override: a must be in [0, 1], got {a}")

    pred_arr = np.asarray(prediction, dtype=np.float64)
    known_arr = np.asarray(known_label, dtype=np.float64)
    mask = np.asarray(override_mask, dtype=bool)

    # Boolean fancy-indexing (out[mask] = ...) pays a gather (known_arr[mask]/pred_arr[mask], each a new
    # allocation) plus a scatter (the assignment back) -- measured ~2x slower than computing the blend for
    # EVERY row and selecting via np.where (a single elementwise pass, no gather/scatter), which is
    # mathematically identical since a=0 outside the mask makes non-overridden rows compute an unused value
    # cheaply rather than not compute it at all.
    blended = a * known_arr + (1.0 - a) * pred_arr
    return np.asarray(np.where(mask, blended, pred_arr))


__all__ = ["apply_smoothed_override"]
