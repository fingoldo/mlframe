"""Shared synthetic-dataset builder used by several ``_benchmarks/`` scripts in this directory:
independently duplicated across those scripts, consolidated here so a fix can't silently drift out
of sync across copies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_gaussian_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    """Synthetic (n_rows, n_cols) Gaussian frame, columns named ``f0``..``f{n_cols-1}``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])
