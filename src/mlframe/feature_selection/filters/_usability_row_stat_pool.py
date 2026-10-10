"""Row-statistic candidates for the usability pool (the linear-downstream selection list).

A linear model cannot form the range, the extreme or the spread of several columns, so the accepted row statistics are offered to the pool of the linear list as replayable candidates; the
usability greedy keeps one only if it improves the held-out linear fit.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["row_stat_pool_candidates"]


def row_stat_pool_candidates(df: Any, y_cont: np.ndarray, base_names: Sequence[str], feature_dtype: Any, quantization_nbins: int) -> list:
    """``UsableCandidate`` objects for the row statistics the family accepts on the numeric ``base_names`` columns of ``df`` (empty on any failure: the pool simply stays as it was)."""
    from ._mi_greedy_cmi_fe import _quantile_bin, precompute_marginal_y_terms
    from ._row_stat_fe import hybrid_row_stat_fe
    from ._usability_aware_selection import UsableCandidate, _binned_mi, _scrub

    try:
        _, appended, recipes, enc = hybrid_row_stat_fe(df, y_cont, num_cols=list(base_names))
    except Exception as e:  # an optional enrichment of the pool; never sink the usability pass
        logger.debug("row-statistic usability candidates skipped: %s", e)
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
