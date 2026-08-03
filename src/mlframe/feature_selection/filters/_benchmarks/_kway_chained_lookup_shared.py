"""Shared pre-fix reference implementation for the kway-chained-lookup bench cluster
(``bench_kway_chained_lookup_unique.py`` / ``bench_kway_chained_lookup_njit_renumber.py``):
independently duplicated across those scripts, consolidated here so a fix can't silently
drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory import merge_vars


def chain_old(factors_data, idx_tuple, nbins, dtype):
    """Pre-fix reference: merge_vars over the full growing prefix at every step."""
    results = []
    for step in range(1, len(idx_tuple)):
        vi_prefix = np.array(idx_tuple[: step + 1], dtype=np.int64)
        cls_next, _, n_uniq_next = merge_vars(
            factors_data=factors_data, vars_indices=vi_prefix,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        results.append((cls_next.astype(np.int64, copy=False), int(n_uniq_next)))
    return results
