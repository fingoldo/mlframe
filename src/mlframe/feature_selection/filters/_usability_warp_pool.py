"""Out-of-fold warp candidates for the usability pool (the linear-downstream selection list).

A linear model cannot use a column that matters through a non-monotone effect, and the warp ``E[rank(y) | x]`` makes that effect linear in the target. The MI-list stage of the warp family
accepts only warps whose MI gain over the raw column is significant; for the linear list every usable column's warp is offered (the best ``MAX_OFFERED`` by held-out MI gain) and the usability
greedy keeps one only if it improves the held-out linear fit.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["warp_pool_candidates"]

MAX_OFFERED = 8  # warps offered to the linear pool: bounds the extra cross-validated candidates


def warp_pool_candidates(df: Any, y_cont: np.ndarray, base_names: Sequence[str], feature_dtype: Any, quantization_nbins: int) -> list:
    """``UsableCandidate`` objects for the warps of the numeric ``base_names`` columns of ``df`` (empty on any failure: the pool simply stays as it was)."""
    from ._mi_greedy_cmi_fe import _quantile_bin, precompute_marginal_y_terms
    from ._oof_warp_fe import build_oof_warp1d_recipe, training_column, warp_candidates
    from ._usability_aware_selection import UsableCandidate, _binned_mi, _scrub

    try:
        cands = warp_candidates(df, y_cont, list(base_names))
    except Exception as e:  # an optional enrichment of the pool; never sink the usability pass
        logger.debug("warp usability candidates skipped: %s", e)
        return []
    if not cands:
        return []
    y_codes = _quantile_bin(y_cont, quantization_nbins, host_only=True)
    y_terms = precompute_marginal_y_terms(y_codes)
    out = []
    for cand in sorted(cands, key=lambda c: -c["gain"])[:MAX_OFFERED]:
        col = training_column(cand)
        values = _scrub(col, feature_dtype)
        if float(np.std(values)) <= 1e-9:
            continue
        fit = cand["fit"]
        name = f"oofwarp({cand['col']})"
        recipe = build_oof_warp1d_recipe(name=name, src=cand["col"], cx=fit["cx"], cy=fit["cy"], fill=fit["fill"], lo=float(col.min()), hi=float(col.max()))
        out.append(UsableCandidate(name, values, _binned_mi(values, y_codes, quantization_nbins, y_terms), recipe, (str(cand["col"]),), ()))
    return out
