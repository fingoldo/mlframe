"""Shared ``CompositeTargetDiscovery.fit`` driver for the iter9x cProfile bench cluster: independently
duplicated across those scripts, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def run_discovery_fit(make_discovery, Xy, feature_cols, n):
    """Build a fresh discovery instance via ``make_discovery()`` and fit it on the first ``n`` rows of ``Xy``."""
    disco = make_discovery()
    train_idx = np.arange(n)
    return disco.fit(Xy, "y", feature_cols, train_idx)
