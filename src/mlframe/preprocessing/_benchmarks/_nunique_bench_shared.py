"""Shared pre-fix reference implementation for the ``_get_nunique`` A/B benches
(``bench_paired74.py`` / ``bench_nunique74.py``): the exact ``np.unique``-based OLD impl,
independently duplicated across those scripts, consolidated here so a fix can't silently drift
out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def get_nunique_old(vals, skip_nan=True, skip_vals=None):
    """Pre-fix reference: ``np.unique`` full sort + post-filter, for the OLD side of the A/B."""
    unique_vals = np.unique(vals)
    if skip_nan:
        if unique_vals.dtype.kind in ("f", "c"):
            unique_vals = unique_vals[~np.isnan(unique_vals)]
        else:
            import pandas as pd

            unique_vals = unique_vals[~pd.isna(unique_vals)]
    if skip_vals:
        for val in skip_vals:
            unique_vals = unique_vals[unique_vals != val]
    return len(unique_vals)
