"""Shared scenario-building helpers for the early-stopping-family benches
(``bench_val_size_earlystop.py`` / ``bench_early_stopping_rounds_rule.py``): independently
duplicated across those scripts, consolidated here so a fix can't silently drift out of sync
across copies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fte(regression):
    """Canonical test extractor (8-tuple transform) shared by every scenario, so the suite contract is exercised exactly."""
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    return SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression)


def split(X, y, p, seed):
    """25/75 holdout/train split into named-column frames with a ``target`` column, seeded off ``seed + 777``."""
    rng = np.random.RandomState(seed + 777)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_hold = n // 4
    hold, train = idx[:n_hold], idx[n_hold:]
    cols = [f"f_{i}" for i in range(p)]
    tr = pd.DataFrame(X[train], columns=cols)
    tr["target"] = y[train]
    ho = pd.DataFrame(X[hold], columns=cols)
    ho["target"] = y[hold]
    return tr, ho
