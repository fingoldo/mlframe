"""Offset-product candidates for the usability pool (the linear-downstream selection list).

``select``-ing for a linear model happens in a separate pool (``build_usability_candidate_pool``) that does not see the columns the main FE stages engineered, so a sign-crossing interaction
``(u + s) * (v + t)`` found by ``_offset_product_fe`` helped only the MI list. This adds the family's accepted columns to that pool as replayable candidates; the usability greedy then keeps
one only if it improves the held-out linear fit.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["offset_product_candidates"]


def offset_product_candidates(df: Any, y_cont: np.ndarray, base_names: Sequence[str], feature_dtype: Any, quantization_nbins: int) -> list:
    """``UsableCandidate`` objects for the offset products the family accepts on the raw numeric frame ``df`` (empty on any failure: the pool simply stays as it was)."""
    from ._mi_greedy_cmi_fe import _quantile_bin, precompute_marginal_y_terms
    from ._offset_product_fe import hybrid_offset_product_fe
    from ._usability_aware_selection import UsableCandidate, _binned_mi, _scrub

    try:
        _, appended, recipes, enc = hybrid_offset_product_fe(df, y_cont, num_cols=list(base_names))
    except Exception as e:  # the family is an optional enrichment of the pool; never sink the usability pass
        logger.debug("offset-product usability candidates skipped: %s", e)
        return []
    if not appended:
        return []
    y_codes = _quantile_bin(y_cont, quantization_nbins, host_only=True)
    y_terms = precompute_marginal_y_terms(y_codes)
    out = []
    for rec in recipes:
        values = _scrub(enc[rec.name].to_numpy(), feature_dtype)
        if float(np.std(values)) <= 1e-9:
            continue
        out.append(UsableCandidate(rec.name, values, _binned_mi(values, y_codes, quantization_nbins, y_terms), rec, tuple(rec.src_names), ()))
    return out
