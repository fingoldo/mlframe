"""Shared helper for the ``_integer_lattice_fe_proto.py`` / ``_conditional_gate_fe_proto.py`` sibling
prototypes: independently duplicated across those modules, consolidated here so a fix can't silently
drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np

from ._pairwise_modular_fe import _mi


def perm_null_hi(feat: np.ndarray, yi: np.ndarray, nbins: int, n_perm: int, rng: np.random.Generator, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of the fixed feature's MI under y permutation - the noise reference the
    feature must clear. The feature is fixed; only y is shuffled (cheap, n_perm small)."""
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(feat, yi[rng.permutation(yi.size)], nbins=nbins)
    return float(vals.mean() + z * vals.std())
